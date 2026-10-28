import argparse
import json
import logging
import os
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from env_config import load_env_file
import s3_store
from segment_detector import detect_racing_segments

log = logging.getLogger("pipeline")


def setup_logging(log_path):
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if log_path:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)


def load_processed_keys(manifest_path):
    done = set()
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["s3_key"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def append_manifest(manifest_path, record):
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _merge_spans(spans, gap=1.0):
    spans = sorted(spans)
    merged = []
    for s, e in spans:
        if merged and s - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return [[round(s, 2), round(e, 2)] for s, e in merged]


def process_one(client, bucket, key, local_path, verify_uncertain):
    s3_store.download(client, bucket, key, local_path)
    try:
        det = detect_racing_segments(local_path)
        racing_segments = list(det["confirmed_segments"])
        verification_log = []

        if verify_uncertain and det["uncertain_segments"]:
            import verification
            for s, e in det["uncertain_segments"]:
                is_racing, resolved_by, vehicle_type = verification.verify_segment(local_path, s, e)
                verification_log.append({
                    "span": [s, e], "racing": is_racing,
                    "resolved_by": resolved_by, "vehicle_type": vehicle_type,
                })
                if is_racing:
                    racing_segments.append([s, e])

        racing_segments = _merge_spans(racing_segments)
        needs_paid_verification = bool(det["uncertain_segments"]) and not verify_uncertain

        return {
            "s3_key": key,
            "filename": os.path.basename(key),
            "duration_sec": det["duration_sec"],
            "fps": det["fps"],
            "racing_segments_sec": racing_segments,
            "uncertain_segments_sec": det["uncertain_segments"],
            "needs_paid_verification": needs_paid_verification,
            "verification": verification_log,
            "status": "ok",
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


def _run_one(client, bucket, key, size, tmp_dir, task_id, verify_uncertain):
    local_path = os.path.join(tmp_dir, f"w{task_id}_{os.path.basename(key)}")
    try:
        record = process_one(client, bucket, key, local_path, verify_uncertain)
    except Exception as e:
        record = {
            "s3_key": key,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)
    return key, size, record


def main():
    ap = argparse.ArgumentParser(description="Download clips from S3, detect racing-car segments, record timestamps, delete local copy.")
    ap.add_argument("--manifest", default="manifest.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="process at most N new videos (for testing)")
    ap.add_argument("--verify-uncertain", action="store_true", help="resolve uncertain segments via SAM3.1/gpt6-luna (paid)")
    ap.add_argument("--tmp-dir", default=None)
    ap.add_argument("--log-file", default="pipeline.log")
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8, help="concurrent download+detect workers")
    ap.add_argument("--time-limit-sec", type=float, default=None, help="stop submitting new work after this many seconds; in-flight videos finish")
    args = ap.parse_args()

    setup_logging(args.log_file)
    load_env_file()
    bucket = os.environ["S3_BUCKET"]
    prefix = os.environ.get("S3_PREFIX", "")
    client = s3_store.make_client()

    log.info("listing objects under s3://%s/%s ...", bucket, prefix)
    all_videos = list(s3_store.list_videos(client, bucket, prefix))
    done = load_processed_keys(args.manifest)
    todo = [(k, sz) for k, sz in all_videos if k not in done]
    if args.limit:
        todo = todo[: args.limit]
    total = len(todo)
    log.info("found %d videos total, %d already in manifest, %d to process now (workers=%d, verify_uncertain=%s)",
              len(all_videos), len(done), total, args.workers, args.verify_uncertain)

    n_done, n_ok, n_err, n_needs_verify, n_with_racing = 0, 0, 0, 0, 0
    start = time.monotonic()
    lock = threading.Lock()

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir, \
         ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(_run_one, client, bucket, key, size, tmp_dir, i, args.verify_uncertain)
            for i, (key, size) in enumerate(todo)
        ]
        time_up = False
        for fut in as_completed(futures):
            key, size, record = fut.result()
            with lock:
                n_done += 1
                if record["status"] == "ok":
                    n_ok += 1
                    if record["racing_segments_sec"]:
                        n_with_racing += 1
                    if record["needs_paid_verification"]:
                        n_needs_verify += 1
                        log.info("[%d/%d] %s (%.1f MB) -> racing=%s, UNVERIFIED uncertain=%s",
                                  n_done, total, key, size / 1e6, record["racing_segments_sec"], record["uncertain_segments_sec"])
                    else:
                        log.info("[%d/%d] %s (%.1f MB) -> racing=%s", n_done, total, key, size / 1e6, record["racing_segments_sec"])
                else:
                    n_err += 1
                    log.error("[%d/%d] %s -> ERROR: %s", n_done, total, key, record.get("error"))
                append_manifest(args.manifest, record)

                elapsed = time.monotonic() - start
                if n_done % args.progress_every == 0 or n_done == total:
                    rate = n_done / elapsed if elapsed > 0 else 0
                    eta_sec = (total - n_done) / rate if rate > 0 else 0
                    log.info(
                        "PROGRESS %d/%d | ok=%d err=%d with_racing=%d needs_paid_verification=%d | %.2fs/video (wall, %d workers) | ETA %.0fmin",
                        n_done, total, n_ok, n_err, n_with_racing, n_needs_verify, elapsed / max(1, n_done), args.workers, eta_sec / 60,
                    )

            if args.time_limit_sec and elapsed >= args.time_limit_sec:
                time_up = True
                log.info("time limit of %.0fs reached after %d/%d videos, cancelling remaining queued work ...",
                          args.time_limit_sec, n_done, total)
                pool.shutdown(wait=True, cancel_futures=True)
                break

    log.info(
        "%s. processed=%d/%d ok=%d err=%d with_racing=%d needs_paid_verification=%d manifest=%s",
        "STOPPED (time limit)" if time_up else "DONE",
        n_done, total, n_ok, n_err, n_with_racing, n_needs_verify, args.manifest,
    )


if __name__ == "__main__":
    main()
