import argparse
import json
import logging
import os
import tempfile
import time

from env_config import load_env_file
import s3_store
from pipeline import setup_logging, _merge_spans, log


def main():
    ap = argparse.ArgumentParser(description="Resolve manifest entries flagged needs_paid_verification=true via SAM3.1/gpt6-luna, once API access is available.")
    ap.add_argument("--manifest", default="manifest.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log-file", default="reverify.log")
    args = ap.parse_args()

    setup_logging(args.log_file)
    load_env_file()
    import verification

    with open(args.manifest, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    pending = [r for r in records if r.get("status") == "ok" and r.get("needs_paid_verification")]
    if args.limit:
        pending = pending[: args.limit]
    log.info("found %d records pending paid verification (of %d total in manifest)", len(pending), len(records))

    bucket = os.environ["S3_BUCKET"]
    client = s3_store.make_client()
    by_key = {r["s3_key"]: r for r in records}

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, rec in enumerate(pending, start=1):
            key = rec["s3_key"]
            local_path = os.path.join(tmp_dir, rec["filename"])
            log.info("[%d/%d] %s", i, len(pending), key)
            try:
                s3_store.download(client, bucket, key, local_path)
                racing_segments = list(rec["racing_segments_sec"])
                verification_log = list(rec.get("verification", []))
                for s, e in rec["uncertain_segments_sec"]:
                    is_racing, resolved_by, vehicle_type = verification.verify_segment(local_path, s, e)
                    verification_log.append({
                        "span": [s, e], "racing": is_racing,
                        "resolved_by": resolved_by, "vehicle_type": vehicle_type,
                    })
                    if is_racing:
                        racing_segments.append([s, e])
                rec["racing_segments_sec"] = _merge_spans(racing_segments)
                rec["verification"] = verification_log
                rec["needs_paid_verification"] = False
                rec["reverified_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                log.info("  -> racing=%s", rec["racing_segments_sec"])
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
