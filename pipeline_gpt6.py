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
import verification
from segment_detector import detect_racing_segments
from pipeline import load_processed_keys, append_manifest

log = logging.getLogger("pipeline_gpt6")


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


def process_one(client, bucket, key, local_path, prefix, clips_prefix, tmp_dir, task_id, token_totals, token_lock):
    s3_store.download(client, bucket, key, local_path)
    try:
        det = detect_racing_segments(local_path, borderline_resolver=verification.verify_segment_resolver)
        racing_segments = det["confirmed_segments"]

        video_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
        for b in det.get("borderline_resolved", []):
            usage = b.get("usage") or {}
            if usage:
                video_tokens["calls"] += 1
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                video_tokens[k] += usage.get(k, 0)
        with token_lock:
            for k in video_tokens:
                token_totals[k] += video_tokens[k]

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
            "token_usage": video_tokens,
            "status": "ok",
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


def _run_one(client, bucket, key, size, tmp_dir, task_id, prefix, clips_prefix, token_totals, token_lock):
    local_path = os.path.join(tmp_dir, f"w{task_id}_{os.path.basename(key)}")
    try:
        record = process_one(client, bucket, key, local_path, prefix, clips_prefix, tmp_dir, task_id, token_totals, token_lock)
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
    ap = argparse.ArgumentParser(
        description="Same pipeline as pipeline.py, but borderline segments are "
                     "resolved with real SAM3.1/gpt6-luna calls instead of the "
                     "classical heuristic - for measuring real token spend on a "
                     "timed test run, not for production use."
    )
    ap.add_argument("--manifest", default="manifest_gpt6.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--clips-prefix", default=None)
    ap.add_argument("--tmp-dir", default=None)
    ap.add_argument("--log-file", default="pipeline_gpt6.log")
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--time-limit-sec", type=float, default=None)
    args = ap.parse_args()

    setup_logging(args.log_file)
    load_env_file()
    bucket = os.environ["S3_BUCKET"]
    prefix = os.environ.get("S3_PREFIX", "")
    clips_prefix = args.clips_prefix or os.environ.get("S3_CLIPS_PREFIX", "autosport_racing_clips_gpt6_test/")
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
    token_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
    start = time.monotonic()
    lock = threading.Lock()
    token_lock = threading.Lock()

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir, \
         ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(_run_one, client, bucket, key, size, tmp_dir, i, prefix, clips_prefix, token_totals, token_lock)
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
                    log.info("[%d/%d] %s (%.1f MB) -> racing=%s clips=%d vision_calls=%d tokens=%d",
                              n_done, total, key, size / 1e6, record["racing_segments_sec"],
                              len(record.get("clips", [])), record.get("token_usage", {}).get("calls", 0),
                              record.get("token_usage", {}).get("total_tokens", 0))
                else:
                    n_err += 1
                    log.error("[%d/%d] %s -> ERROR: %s", n_done, total, key, record.get("error"))
                append_manifest(args.manifest, record)

                elapsed = time.monotonic() - start
                if n_done % args.progress_every == 0 or n_done == total:
                    rate = n_done / elapsed if elapsed > 0 else 0
                    eta_sec = (total - n_done) / rate if rate > 0 else 0
                    with token_lock:
                        tt = dict(token_totals)
                    log.info(
                        "PROGRESS %d/%d | ok=%d err=%d with_racing=%d clips=%d | %.2fs/video (wall, %d workers) | "
                        "ETA %.0fmin | vision_calls=%d prompt_tokens=%d completion_tokens=%d total_tokens=%d",
                        n_done, total, n_ok, n_err, n_with_racing, n_clips, elapsed / max(1, n_done), args.workers, eta_sec / 60,
                        tt["calls"], tt["prompt_tokens"], tt["completion_tokens"], tt["total_tokens"],
                    )

            if args.time_limit_sec and elapsed >= args.time_limit_sec:
                time_up = True
                log.info("time limit of %.0fs reached after %d/%d videos, cancelling remaining queued work ...",
                          args.time_limit_sec, n_done, total)
                pool.shutdown(wait=True, cancel_futures=True)
                break

    log.info(
        "%s. processed=%d/%d ok=%d err=%d with_racing=%d clips=%d vision_calls=%d prompt_tokens=%d "
        "completion_tokens=%d total_tokens=%d manifest=%s",
        "STOPPED (time limit)" if time_up else "DONE",
        n_done, total, n_ok, n_err, n_with_racing, n_clips,
        token_totals["calls"], token_totals["prompt_tokens"], token_totals["completion_tokens"], token_totals["total_tokens"],
        args.manifest,
    )


if __name__ == "__main__":
    main()
