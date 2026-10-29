import os
import shutil
import subprocess

import cv2


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


def clip_s3_key(original_key, source_prefix, clips_prefix, start_sec):
    """autosport_racing_filtered/Track/Cam/file.mp4, start=3.45
    -> <clips_prefix>/Track/Cam/file_3.45.mp4 (prev_filename + timestamp_start)."""
    rel = original_key
    if source_prefix and original_key.startswith(source_prefix):
        rel = original_key[len(source_prefix):]
    rel_dir, filename = os.path.split(rel)
    base, ext = os.path.splitext(filename)
    clip_name = f"{base}_{start_sec:.2f}{ext}"
    prefix = clips_prefix.rstrip("/")
    return f"{prefix}/{rel_dir}/{clip_name}" if rel_dir else f"{prefix}/{clip_name}"
