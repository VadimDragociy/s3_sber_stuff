import argparse
import json
import logging
import os
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import clip_export
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
    """Only status="ok" counts as done, so a transient failure (a crashed
    ffmpeg call, an API timeout, ...) gets retried on the next run instead
    of silently and permanently skipping that video - confirmed this
    mattered: 97 videos errored out in one run before the cause was fixed,
    and without this they'd never have been attempted again."""
    done = set()
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "ok":
                        done.add(rec["s3_key"])
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


def process_one(client, bucket, key, local_path, prefix, clips_prefix, tmp_dir, task_id):
    s3_store.download(client, bucket, key, local_path)
    try:
        det = detect_racing_segments(local_path)
        racing_segments = det["confirmed_segments"]

        clips = []
        for i, (start, end) in enumerate(racing_segments):
            clip_local = os.path.join(tmp_dir, f"w{task_id}_clip{i}_{os.path.basename(key)}")
            try:
                clip_export.cut_clip(local_path, start, end, clip_local)
                clip_key = clip_export.clip_s3_key(key, prefix, clips_prefix, start, end, det["fps"])
                s3_store.upload(client, bucket, clip_key, clip_local)
                clips.append({"start": start, "end": end, "s3_key": clip_key})
            finally:
                if os.path.exists(clip_local):
                    os.remove(clip_local)

        return {
            "s3_key": key,
            "filename": os.path.basename(key),
            "duration_sec": det["duration_sec"],
            "fps": det["fps"],
            "racing_segments_sec": racing_segments,
            "clips": clips,
            "borderline_resolved": det.get("borderline_resolved", []),
            "status": "ok",
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


def _run_one(client, bucket, key, size, tmp_dir, task_id, prefix, clips_prefix):
    local_path = os.path.join(tmp_dir, f"w{task_id}_{os.path.basename(key)}")
    try:
        record = process_one(client, bucket, key, local_path, prefix, clips_prefix, tmp_dir, task_id)
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
    ap = argparse.ArgumentParser(description="Download clips from S3, detect racing-car segments locally (no paid vision model calls), cut them out, upload the clips to S3, and delete local copies.")
    ap.add_argument("--manifest", default="manifest.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="process at most N new videos (for testing)")
    ap.add_argument("--clips-prefix", default=None, help="S3 prefix to upload cut racing clips under (default: $S3_CLIPS_PREFIX or 'autosport_racing_clips/')")
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
    clips_prefix = args.clips_prefix or os.environ.get("S3_CLIPS_PREFIX", "autosport_racing_clips/")
    client = s3_store.make_client()

    log.info("listing objects under s3://%s/%s ...", bucket, prefix)
    all_videos = list(s3_store.list_videos(client, bucket, prefix))
    done = load_processed_keys(args.manifest)
    todo = [(k, sz) for k, sz in all_videos if k not in done]
    if args.limit:
        todo = todo[: args.limit]
    total = len(todo)
    log.info("found %d videos total, %d already in manifest, %d to process now (workers=%d, clips_prefix=%s)",
              len(all_videos), len(done), total, args.workers, clips_prefix)

    n_done, n_ok, n_err, n_with_racing, n_clips = 0, 0, 0, 0, 0
    start = time.monotonic()
    lock = threading.Lock()

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir, \
         ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(_run_one, client, bucket, key, size, tmp_dir, i, prefix, clips_prefix)
            for i, (key, size) in enumerate(todo)
        ]
        time_up = False
        for fut in as_completed(futures):
            key, size, record = fut.result()
            with lock:
                n_done += 1
                if record["status"] == "ok":
                    n_ok += 1
                    n_clips += len(record.get("clips", []))
                    if record["racing_segments_sec"]:
                        n_with_racing += 1
                    log.info("[%d/%d] %s (%.1f MB) -> racing=%s clips=%d",
                              n_done, total, key, size / 1e6, record["racing_segments_sec"], len(record.get("clips", [])))
                else:
                    n_err += 1
                    log.error("[%d/%d] %s -> ERROR: %s", n_done, total, key, record.get("error"))
                append_manifest(args.manifest, record)

                elapsed = time.monotonic() - start
                if n_done % args.progress_every == 0 or n_done == total:
                    rate = n_done / elapsed if elapsed > 0 else 0
                    eta_sec = (total - n_done) / rate if rate > 0 else 0
                    log.info(
                        "PROGRESS %d/%d | ok=%d err=%d with_racing=%d clips=%d | %.2fs/video (wall, %d workers) | ETA %.0fmin",
                        n_done, total, n_ok, n_err, n_with_racing, n_clips, elapsed / max(1, n_done), args.workers, eta_sec / 60,
                    )

            if args.time_limit_sec and elapsed >= args.time_limit_sec:
                time_up = True
                log.info("time limit of %.0fs reached after %d/%d videos, cancelling remaining queued work ...",
                          args.time_limit_sec, n_done, total)
                pool.shutdown(wait=True, cancel_futures=True)
                break

    log.info(
        "%s. processed=%d/%d ok=%d err=%d with_racing=%d clips=%d manifest=%s",
        "STOPPED (time limit)" if time_up else "DONE",
        n_done, total, n_ok, n_err, n_with_racing, n_clips, args.manifest,
    )


if __name__ == "__main__":
    main()
