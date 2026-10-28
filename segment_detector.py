import cv2
import numpy as np


def _read_sampled_gray_frames(video_path, sample_hz, max_dim):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    scale = min(1.0, max_dim / max(w, h)) if max(w, h) > 0 else 1.0
    W, H = max(1, int(w * scale)), max(1, int(h * scale))
    step = max(1, int(round(fps / sample_hz)))

    times, frames = [], []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            small = cv2.resize(frame, (W, H))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            times.append(idx / fps)
        idx += 1
    cap.release()

    duration = (nframes / fps) if fps else (times[-1] if times else 0.0)
    return frames, times, fps, duration, (W, H)


def _corner_roi_mask(W, H, corner_frac):
    mask = np.full((H, W), 255, dtype=np.uint8)
    bw, bh = int(W * corner_frac), int(H * corner_frac)
    for x0, y0 in [(0, 0), (W - bw, 0), (0, H - bh), (W - bw, H - bh)]:
        mask[y0:y0 + bh, x0:x0 + bw] = 0
    return mask


def _blob_candidates(gray, bg, roi_mask, diff_thresh, min_area):
    diff = cv2.absdiff(gray, bg)
    diff = cv2.bitwise_and(diff, diff, mask=roi_mask)
    diff = cv2.GaussianBlur(diff, (3, 3), 0)
    _, th = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
    th = cv2.dilate(th, np.ones((2, 2), np.uint8), iterations=1)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        blobs.append((m["m10"] / m["m00"], m["m01"] / m["m00"], area))
    return blobs


def detect_racing_segments(
    video_path,
    sample_hz=4.0,
    max_dim=960,
    corner_frac=0.09,
    diff_thresh=22,
    min_blob_area=10,
    max_link_dist_frac=0.15,
    max_area_ratio_step=3.0,
    min_track_samples=3,
    strong_area_ratio=3.0,
    strong_net_disp_frac=0.20,
    weak_net_disp_frac=0.05,
    weak_area_ratio=1.3,
    merge_gap_sec=1.0,
    pad_sec=0.5,
):
    """Zero-cost background-subtraction detector, tracking only the single
    largest ROI blob per sample (avoids chaining unrelated noise specks).

    A track is classified by two physically meaningful signals instead of
    raw motion: how far the object's centroid travelled relative to the
    frame (net_disp_frac) and how much its apparent size changed
    (area_ratio) - a real car passing a fixed camera does one or both
    (approaches/recedes = area changes a lot; crosses the frame = position
    changes a lot). A static artifact (stuck glare/shadow) does neither.

    Returns three timelines in seconds:
      - confirmed: strong evidence of a moving car -> keep, no paid model needed
      - uncertain: persists for a few samples but weak evidence -> verify
        with one SAM3.1 call per segment before deciding
      - everything else is implicitly "empty" -> drop, no paid model needed
    """
    frames, times, fps, duration, (W, H) = _read_sampled_gray_frames(video_path, sample_hz, max_dim)
    result = {"confirmed_segments": [], "uncertain_segments": [], "fps": fps,
              "duration_sec": round(duration, 2), "n_samples": len(frames)}
    if len(frames) < 2:
        return result

    stack = np.stack(frames, axis=0).astype(np.float32)
    bg = np.median(stack, axis=0).astype(np.uint8)
    roi_mask = _corner_roi_mask(W, H, corner_frac)
    diag = (W ** 2 + H ** 2) ** 0.5
    max_link_dist = diag * max_link_dist_frac

    tracks = []
    cur = None
    for t, gray in zip(times, frames):
        blobs = _blob_candidates(gray, bg, roi_mask, diff_thresh, min_blob_area)
        top = max(blobs, key=lambda b: b[2]) if blobs else None

        linked = False
        if top is not None and cur is not None:
            x, y, area = top
            lx, ly = cur["last"]
            d = ((x - lx) ** 2 + (y - ly) ** 2) ** 0.5
            ratio = max(area, cur["last_area"]) / max(1.0, min(area, cur["last_area"]))
            if d < max_link_dist and ratio < max_area_ratio_step:
                cur["last"] = (x, y)
                cur["last_t"] = t
                cur["n"] += 1
                cur["last_area"] = area
                cur["area_min"] = min(cur["area_min"], area)
                cur["area_max"] = max(cur["area_max"], area)
                linked = True

        if not linked:
            if cur is not None:
                tracks.append(cur)
            cur = None
            if top is not None:
                x, y, area = top
                cur = {"first": (x, y), "last": (x, y), "start_t": t, "last_t": t,
                       "n": 1, "last_area": area, "area_min": area, "area_max": area}
    if cur is not None:
        tracks.append(cur)

    def net_disp(tr):
        fx, fy = tr["first"]
        lx, ly = tr["last"]
        return ((lx - fx) ** 2 + (ly - fy) ** 2) ** 0.5

    confirmed, uncertain = [], []
    for tr in tracks:
        if tr["n"] < min_track_samples:
            continue
        span = (tr["start_t"], tr["last_t"])
        nd_frac = net_disp(tr) / diag
        ar = tr["area_max"] / max(1.0, tr["area_min"])

        if ar >= strong_area_ratio or nd_frac >= strong_net_disp_frac:
            confirmed.append(span)
        elif nd_frac < weak_net_disp_frac and ar < weak_area_ratio:
            continue  # jitter/artifact, drop for free
        else:
            uncertain.append(span)

    def _merge(spans):
        spans = sorted(spans)
        merged = []
        for s, e in spans:
            if merged and s - merged[-1][1] <= merge_gap_sec:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return [[round(max(0.0, s - pad_sec), 2), round(min(duration, e + pad_sec), 2)] for s, e in merged]

    result["confirmed_segments"] = _merge(confirmed)
    result["uncertain_segments"] = _merge(uncertain)
    return result
