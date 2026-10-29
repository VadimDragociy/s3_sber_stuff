import os
import re
import shutil
import subprocess

import cv2

# See the matching comment in segment_detector.py: caps OpenCV's own
# internal per-call parallelism, and - via CAP_PROP_N_THREADS passed at
# VideoCapture-open time below (the only thing that actually worked) -
# FFmpeg's internal decode-thread pool, so N worker threads don't multiply
# into far more OS threads than --workers would suggest.
cv2.setNumThreads(1)

_SINGLE_THREAD_DECODE = [cv2.CAP_PROP_N_THREADS, 1]

# Source filenames look like:
#   IgoraDrive_07_08_26__C-03__2026_08_07-09_45_26_014_00410_00445_000_00000_00012.mp4
#   <place>_<date DD_MM_YY>__<camera>__<recording start YYYY_MM_DD-HH_MM_SS>_<seg_idx>_<day_sec_start>_<day_sec_end>_<...>
# <day_sec_start>/<day_sec_end> is this short (12-70s) source file's own
# position, in SECONDS, within the full continuous per-camera recording
# session named by <recording start> - confirmed empirically (their
# difference always matches the clip's duration_sec). That field is what
# makes a clip name unique, NOT <recording start> alone: every one of the
# ~10-20 short segments cut from one continuous recording session shares
# the exact same <recording start>, camera and date. A first version of
# this function built frame numbers from start_sec/end_sec alone (offsets
# into the just-downloaded short segment) and silently collided - two
# different segments of the same session producing the same output
# filename, one overwriting the other in S3 - confirmed on real output:
# 149 of 9372 distinct clip keys in a production run turned out to be
# claimed by more than one source video. <day_sec_start> anchors the frame
# numbers to the full session instead, which is unique per segment.
_SOURCE_NAME_RE = re.compile(
    r"^(?P<place>.+)_(?P<date>\d{2}_\d{2}_\d{2})"
    r"__(?P<camera>[A-Za-z]+-\d+)"
    r"__(?P<start_time>\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})"
    r"_(?P<seg_idx>\d+)_(?P<day_sec_start>\d+)_(?P<day_sec_end>\d+)"
)


def cut_with_ffmpeg(src, start, end, dst):
    # -threads 1: libx264 defaults to one encoder thread per CPU core, so
    # without this every one of the 8 concurrent workers' ffmpeg calls (one
    # per clip, several clips per video) spawns its own large thread pool -
    # the same multiplication problem already fixed for cv2's decoder.
    # Confirmed on the real deploy target: without it, the cgroup's task
    # count hit 360+ (vs. ~55 expected for 8 workers) and ffmpeg itself
    # started failing outright (exit 187/245) under the resulting CPU-quota
    # starvation - 97 of the first 1725 videos errored out before this was
    # caught and the run was stopped.
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", src,
         "-c:v", "libx264", "-preset", "veryfast", "-threads", "1", "-c:a", "aac", dst],
        check=True, capture_output=True,
    )


def cut_with_opencv(src, start, end, dst):
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG, _SINGLE_THREAD_DECODE)
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

    first_frame/last_frame are this clip's start/end as frame numbers into
    the full continuous recording SESSION (anchored via the source
    filename's own day_sec_start - see the regex comment above), not just
    the short file the pipeline happened to download - that's what keeps
    them unique across the session's other segments. Falls back to a
    plain "<source base>_<start>-<end>s<ext>" name (which keeps the whole
    original, already-unique filename) if the source filename doesn't
    match the expected pattern, so an unexpected naming convention never
    breaks the upload - and never silently collides either.
    """
    base, ext = os.path.splitext(source_filename)
    m = _SOURCE_NAME_RE.match(base)
    if not m:
        return f"{base}_{start_sec:.2f}-{end_sec:.2f}s{ext}"
    day_sec_start = int(m["day_sec_start"])
    first_frame = round((day_sec_start + start_sec) * fps)
    last_frame = round((day_sec_start + end_sec) * fps)
    return (
        f"{m['date']}_{m['place']}_{m['camera']}_{m['start_time']}"
        f"_{first_frame:06d}_{last_frame:06d}{ext}"
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
