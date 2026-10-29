import cv2
import numpy as np


def _read_sampled_gray_frames(video_path, sample_hz, max_dim, start_sec=None, end_sec=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    scale = min(1.0, max_dim / max(w, h)) if max(w, h) > 0 else 1.0
    W, H = max(1, int(w * scale)), max(1, int(h * scale))
    step = max(1, int(round(fps / sample_hz)))

    idx = 0
    if start_sec is not None:
        idx = max(0, int(start_sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    base_idx = idx

    times, frames = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = idx / fps
        if end_sec is not None and t > end_sec:
            break
        if (idx - base_idx) % step == 0:
            small = cv2.resize(frame, (W, H))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            times.append(t)
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
        bbox = cv2.boundingRect(c)
        blobs.append((m["m10"] / m["m00"], m["m01"] / m["m00"], area, bbox))
    return blobs


def _patch_texture(gray, bbox):
    """Laplacian variance inside a blob's bounding box on the ORIGINAL
    (non-diff) frame: a real car's livery/numbers/shadows under it read as
    textured/sharp, while a soft shadow, sun glare or specular reflection
    reads as flat/blurry. Zero-cost, purely local signal."""
    x, y, w, h = bbox
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(gray.shape[1], x + w), min(gray.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = gray[y0:y1, x0:x1]
    if patch.size < 4:
        return 0.0
    return float(cv2.Laplacian(patch, cv2.CV_64F).var())


def _median_shape_ratio(bboxes):
    ratios = [max(w, 1) / max(h, 1) for (_, _, w, h) in bboxes if h > 0]
    return float(np.median(ratios)) if ratios else 0.0


def _motion_consistency(centroids):
    """Average cosine similarity between consecutive displacement vectors.
    Close to 1 = steady one-directional travel (a car driving through
    frame); close to 0/negative = jitter (flicker, noise). Tracks too
    short to judge a direction change are not penalized."""
    if len(centroids) < 3:
        return 1.0
    vecs = []
    for (x0, y0), (x1, y1) in zip(centroids[:-1], centroids[1:]):
        dx, dy = x1 - x0, y1 - y0
        norm = (dx ** 2 + dy ** 2) ** 0.5
        if norm > 1e-6:
            vecs.append((dx / norm, dy / norm))
    if len(vecs) < 2:
        return 1.0
    sims = [ax * bx + ay * by for (ax, ay), (bx, by) in zip(vecs[:-1], vecs[1:])]
    return float(np.mean(sims))


def _secondary_decision(texture_score, shape_ratio, motion_consistency,
                         texture_min, shape_ratio_min, shape_ratio_max, motion_consistency_min):
    return (
        texture_score >= texture_min
        and shape_ratio_min <= shape_ratio <= shape_ratio_max
        and motion_consistency >= motion_consistency_min
    )


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
    texture_min=18.0,
    shape_ratio_min=0.8,
    shape_ratio_max=6.0,
    motion_consistency_min=0.3,
):
    """Zero-cost background-subtraction detector, tracking only the single
    largest ROI blob per sample (avoids chaining unrelated noise specks).

    A track is first classified by two physically meaningful motion signals:
    how far the object's centroid travelled relative to the frame
    (net_disp_frac) and how much its apparent size changed (area_ratio) - a
    real car passing a fixed camera does one or both (approaches/recedes =
    area changes a lot; crosses the frame = position changes a lot). A
    static artifact (stuck glare/shadow) does neither, so it's dropped for
    free.

    Tracks that are too ambiguous for that first pass (persist for a few
    samples but with weak motion evidence either way) used to be punted to
    a paid vision model (SAM3.1 / gpt6-luna). That round-trip is gone: they
    are instead resolved locally with three extra classical signals that
    are cheap to compute from the frames already sampled:
      - texture_score: Laplacian variance inside the blob's bounding box on
        the real frame (a car's livery/numbers/shadow reads as textured;
        a sun-glare streak or soft shadow reads as flat).
      - shape_ratio: median width/height of the blob's bounding box (a
        trackside camera sees cars as wider-than-tall blobs within a
        plausible range; a thin light streak or near-square speck isn't).
      - motion_consistency: how steadily the blob travels in one direction
        (a driving car is steady; flicker/noise is not).
    These thresholds are heuristic defaults, not tuned against labeled
    ground truth - spot-check a sample batch's "borderline_resolved" log
    before trusting this at scale, and adjust the *_min/*_max kwargs above
    if it's over- or under-including.

    Returns:
      - confirmed_segments: everything accepted as a moving racing vehicle,
        in seconds - no paid model involved at any point.
      - borderline_resolved: diagnostic log of every ambiguous track and
        how the secondary heuristic scored/decided it, for spot-checking.
    """
    frames, times, fps, duration, (W, H) = _read_sampled_gray_frames(video_path, sample_hz, max_dim)
    result = {"confirmed_segments": [], "fps": fps,
              "duration_sec": round(duration, 2), "n_samples": len(frames),
              "borderline_resolved": []}
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
            x, y, area, bbox = top
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
                cur["centroids"].append((x, y))
                cur["bboxes"].append(bbox)
                cur["textures"].append(_patch_texture(gray, bbox))
                linked = True

        if not linked:
            if cur is not None:
                tracks.append(cur)
            cur = None
            if top is not None:
                x, y, area, bbox = top
                cur = {"first": (x, y), "last": (x, y), "start_t": t, "last_t": t,
                       "n": 1, "last_area": area, "area_min": area, "area_max": area,
                       "centroids": [(x, y)], "bboxes": [bbox], "textures": [_patch_texture(gray, bbox)]}
    if cur is not None:
        tracks.append(cur)

    def net_disp(tr):
        fx, fy = tr["first"]
        lx, ly = tr["last"]
        return ((lx - fx) ** 2 + (ly - fy) ** 2) ** 0.5

    confirmed = []
    for tr in tracks:
        if tr["n"] < min_track_samples:
            continue
        span = (tr["start_t"], tr["last_t"])
        nd_frac = net_disp(tr) / diag
        ar = tr["area_max"] / max(1.0, tr["area_min"])

        if ar >= strong_area_ratio or nd_frac >= strong_net_disp_frac:
            confirmed.append(span)
            continue
        if nd_frac < weak_net_disp_frac and ar < weak_area_ratio:
            continue  # jitter/artifact, drop for free

        texture_score = float(np.median(tr["textures"])) if tr["textures"] else 0.0
        shape_ratio = _median_shape_ratio(tr["bboxes"])
        motion_consistency = _motion_consistency(tr["centroids"])
        is_racing = _secondary_decision(
            texture_score, shape_ratio, motion_consistency,
            texture_min, shape_ratio_min, shape_ratio_max, motion_consistency_min,
        )
        result["borderline_resolved"].append({
            "span": [round(span[0], 2), round(span[1], 2)],
            "texture_score": round(texture_score, 1),
            "shape_ratio": round(shape_ratio, 2),
            "motion_consistency": round(motion_consistency, 2),
            "racing": is_racing,
        })
        if is_racing:
            confirmed.append(span)

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
    return result


def classify_span(
    video_path, start_sec, end_sec,
    sample_hz=6.0, max_dim=960, pad_sec=1.0,
    corner_frac=0.09, diff_thresh=22, min_blob_area=10,
    texture_min=18.0, shape_ratio_min=0.8, shape_ratio_max=6.0, motion_consistency_min=0.3,
):
    """Classify one already-recorded 'uncertain' span from an OLD manifest
    (collected before this rewrite, back when such spans were deferred to
    SAM3.1/gpt6-luna) by re-reading just that window from the video and
    applying the same classical shape/texture/motion signals as
    detect_racing_segments - no network or vision-model calls.

    Returns (is_racing: bool, metrics: dict).
    """
    win_start = max(0.0, start_sec - pad_sec)
    win_end = end_sec + pad_sec
    frames, _times, _fps, _duration, (W, H) = _read_sampled_gray_frames(
        video_path, sample_hz, max_dim, start_sec=win_start, end_sec=win_end)
    if len(frames) < 3:
        return False, {"reason": "too_few_frames"}

    stack = np.stack(frames, axis=0).astype(np.float32)
    bg = np.median(stack, axis=0).astype(np.uint8)
    roi_mask = _corner_roi_mask(W, H, corner_frac)

    centroids, bboxes, textures = [], [], []
    for gray in frames:
        blobs = _blob_candidates(gray, bg, roi_mask, diff_thresh, min_blob_area)
        if not blobs:
            continue
        x, y, area, bbox = max(blobs, key=lambda b: b[2])
        centroids.append((x, y))
        bboxes.append(bbox)
        textures.append(_patch_texture(gray, bbox))

    if len(centroids) < 2:
        return False, {"reason": "no_persistent_blob"}

    texture_score = float(np.median(textures))
    shape_ratio = _median_shape_ratio(bboxes)
    motion_consistency = _motion_consistency(centroids)
    is_racing = _secondary_decision(
        texture_score, shape_ratio, motion_consistency,
        texture_min, shape_ratio_min, shape_ratio_max, motion_consistency_min,
    )
    return is_racing, {
        "texture_score": round(texture_score, 1),
        "shape_ratio": round(shape_ratio, 2),
        "motion_consistency": round(motion_consistency, 2),
    }
