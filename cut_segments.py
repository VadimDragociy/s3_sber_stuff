import argparse
import json
import os
import shutil
import subprocess
import tempfile

import cv2

from env_config import load_env_file
import s3_store


def cut_with_ffmpeg(src, start, end, dst):
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", src,
         "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", dst],
        check=True, capture_output=True,
    )


def cut_with_opencv(src, start, end, dst):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame, end_frame = int(start * fps), int(end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    idx = start_frame
    while idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        idx += 1
    writer.release()
    cap.release()


def cut_clip(src, start, end, dst):
    if shutil.which("ffmpeg"):
        cut_with_ffmpeg(src, start, end, dst)
    else:
        cut_with_opencv(src, start, end, dst)


def main():
    ap = argparse.ArgumentParser(description="Read a pipeline manifest and cut out the racing segments into a sibling output folder.")
    ap.add_argument("--manifest", default="manifest.jsonl")
    ap.add_argument("--output-dir", default="../autosport_cut")
    ap.add_argument("--source-dir", default=None, help="look here for already-downloaded videos before hitting S3 again")
    ap.add_argument("--delete-source-after", action="store_true")
    args = ap.parse_args()

    load_env_file()
    bucket = os.environ["S3_BUCKET"]
    client = s3_store.make_client()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.manifest, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    with tempfile.TemporaryDirectory() as tmp_dir:
        for rec in records:
            if rec.get("status") != "ok" or not rec.get("racing_segments_sec"):
                continue

            key = rec["s3_key"]
            filename = rec["filename"]
            local_path = os.path.join(args.source_dir, filename) if args.source_dir else None
            downloaded_here = False
            if not local_path or not os.path.exists(local_path):
                local_path = os.path.join(tmp_dir, filename)
                print(f"downloading {key}")
                s3_store.download(client, bucket, key, local_path)
                downloaded_here = True

            base, ext = os.path.splitext(filename)
            for i, (start, end) in enumerate(rec["racing_segments_sec"]):
                dst = os.path.join(args.output_dir, f"{base}__seg{i:02d}_{start:.1f}-{end:.1f}{ext}")
                if os.path.exists(dst):
                    continue
                print(f"  cutting [{start:.1f}, {end:.1f}] -> {dst}")
                cut_clip(local_path, start, end, dst)

            if downloaded_here and args.delete_source_after:
                os.remove(local_path)

    print("done")


if __name__ == "__main__":
    main()
