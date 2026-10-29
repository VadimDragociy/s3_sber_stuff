import argparse
import json
import os
import tempfile
import time

import clip_export
from env_config import load_env_file
import s3_store
from pipeline import setup_logging, _merge_spans, log
from segment_detector import classify_span


def main():
    ap = argparse.ArgumentParser(
        description="Migrate an OLD manifest (from before the SAM3.1/gpt6-luna "
                     "vision calls were removed) to the new, fully classical "
                     "pipeline: resolve any leftover needs_paid_verification/"
                     "uncertain_segments_sec spans locally, then cut and upload "
                     "racing clips to S3 for every record that doesn't have "
                     "clips yet."
    )
    ap.add_argument("--manifest", default="manifest.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log-file", default="reverify.log")
    ap.add_argument("--clips-prefix", default=None)
    args = ap.parse_args()

    setup_logging(args.log_file)
    load_env_file()

    bucket = os.environ["S3_BUCKET"]
    prefix = os.environ.get("S3_PREFIX", "")
    clips_prefix = args.clips_prefix or os.environ.get("S3_CLIPS_PREFIX", "autosport_racing_clips/")
    client = s3_store.make_client()

    with open(args.manifest, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    pending = [r for r in records if r.get("status") == "ok" and not r.get("clips")]
    if args.limit:
        pending = pending[: args.limit]
    log.info("found %d records without uploaded clips yet (of %d total in manifest)", len(pending), len(records))

    by_key = {r["s3_key"]: r for r in records}

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, rec in enumerate(pending, start=1):
            key = rec["s3_key"]
            local_path = os.path.join(tmp_dir, rec["filename"])
            log.info("[%d/%d] %s", i, len(pending), key)
            try:
                s3_store.download(client, bucket, key, local_path)
                racing_segments = list(rec.get("racing_segments_sec", []))
                resolved_log = list(rec.get("borderline_resolved", []))
                for s, e in rec.get("uncertain_segments_sec", []):
                    is_racing, metrics = classify_span(local_path, s, e)
                    resolved_log.append({"span": [s, e], "racing": is_racing, **metrics})
                    if is_racing:
                        racing_segments.append([s, e])
                racing_segments = _merge_spans(racing_segments)

                fps = rec.get("fps") or 25.0
                clips = []
                for j, (start, end) in enumerate(racing_segments):
                    clip_local = os.path.join(tmp_dir, f"clip{j}_{rec['filename']}")
                    try:
                        clip_export.cut_clip(local_path, start, end, clip_local)
                        clip_key = clip_export.clip_s3_key(key, prefix, clips_prefix, start, end, fps)
                        s3_store.upload(client, bucket, clip_key, clip_local)
                        clips.append({"start": start, "end": end, "s3_key": clip_key})
                    finally:
                        if os.path.exists(clip_local):
                            os.remove(clip_local)

                rec["racing_segments_sec"] = racing_segments
                rec["borderline_resolved"] = resolved_log
                rec["clips"] = clips
                rec.pop("needs_paid_verification", None)
                rec.pop("uncertain_segments_sec", None)
                rec.pop("verification", None)
                rec["reverified_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                log.info("  -> racing=%s clips=%d", racing_segments, len(clips))
            except Exception as e:
                log.error("  ERROR on %s: %s", key, e)
            finally:
                if os.path.exists(local_path):
                    os.remove(local_path)

            by_key[key] = rec
            tmp_manifest = args.manifest + ".tmp"
            with open(tmp_manifest, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(by_key[r["s3_key"]], ensure_ascii=False) + "\n")
            os.replace(tmp_manifest, args.manifest)

    log.info("DONE reverify. manifest=%s", args.manifest)


if __name__ == "__main__":
    main()
