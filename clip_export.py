import os
import re
import shutil
import subprocess

import cv2

# Source filenames look like:
#   IgoraDrive_07_08_26__C-03__2026_08_07-09_45_26_014_00410_00445_000_00000_00012.mp4
#   <place>_<date DD_MM_YY>__<camera>__<recording start YYYY_MM_DD-HH_MM_SS>_<...>
# Only the place/date/camera/recording-start prefix is reused for the clip name;
# everything after it (segment index, source frame range, sub-clip range) is
# specific to how the source pipeline chopped up the day's footage and is
# replaced wholesale by our own first/last frame numbers.
_SOURCE_NAME_RE = re.compile(
    r"^(?P<place>.+)_(?P<date>\d{2}_\d{2}_\d{2})"
    r"__(?P<camera>[A-Za-z]+-\d+)"
    r"__(?P<start_time>\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})"
)


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


def clip_filename(source_filename, start_sec, end_sec, fps):
    """{date}_{place}_{camera}_{recording_start}_{first_frame}_{last_frame}.ext

    first_frame/last_frame are this clip's own start/end, as frame numbers
    into the SOURCE file (the mp4 the pipeline downloaded and is cutting
    from) - not the day's full continuous recording, and not seconds.
    Everything else is carried over unchanged from the source filename.
    Falls back to a plain "<source base>_<start>-<end>s<ext>" name if the
    source filename doesn't match the expected pattern, so an unexpected
    naming convention never breaks the upload.
    """
    base, ext = os.path.splitext(source_filename)
    m = _SOURCE_NAME_RE.match(base)
    first_frame = round(start_sec * fps)
    last_frame = round(end_sec * fps)
    if not m:
        return f"{base}_{start_sec:.2f}-{end_sec:.2f}s{ext}"
    return (
        f"{m['date']}_{m['place']}_{m['camera']}_{m['start_time']}"
        f"_{first_frame:05d}_{last_frame:05d}{ext}"
    )


def clip_s3_key(original_key, source_prefix, clips_prefix, start_sec, end_sec, fps):
    """autosport_racing_filtered/Track/Cam/file.mp4, start=3.0, end=7.27
    -> <clips_prefix>/Track/Cam/<date>_<place>_<camera>_<rec_start>_<first_frame>_<last_frame>.mp4"""
    rel = original_key
    if source_prefix and original_key.startswith(source_prefix):
        rel = original_key[len(source_prefix):]
    rel_dir, filename = os.path.split(rel)
    clip_name = clip_filename(filename, start_sec, end_sec, fps)
    prefix = clips_prefix.rstrip("/")
    return f"{prefix}/{rel_dir}/{clip_name}" if rel_dir else f"{prefix}/{clip_name}"
