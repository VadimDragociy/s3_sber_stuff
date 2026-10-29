import base64
import json
import os

import cv2
import requests

PROMPT = (
    "You are given up to 4 frames sampled a moment apart from a motorsport "
    "track camera. Decide if a RACING VEHICLE is actually present and moving "
    "on track in these frames. This includes BOTH: (a) regular racing/touring "
    "cars (closed-cockpit, sedan- or GT-shaped), and (b) open-wheel single-seater "
    "formula cars, specifically Formula 4 style - low, narrow chassis, exposed "
    "wheels, front/rear wings, open cockpit; these can look very small and thin "
    "at a distance, do not mistake them for debris or track markings. "
    "Answer false for: an empty track, a static/parked vehicle, a non-racing "
    "service/safety/marshal vehicle, pedestrians, or a camera/lighting artifact. "
    "Reply with strict JSON only: "
    '{"racing": true|false, "vehicle_type": "touring"|"formula"|"none", "confidence": 0-1}.'
)


def extract_frames_b64(video_path, timestamps_sec, max_dim=640):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out = []
    for t in timestamps_sec:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        scale = min(1.0, max_dim / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            out.append(base64.b64encode(buf.tobytes()).decode("ascii"))
    cap.release()
    return out


def _call_vision_model(model, images_b64, base_url=None, api_key=None, timeout=60):
    base_url = base_url or os.environ["LITELLM_BASE_URL"]
    api_key = api_key or os.environ["LITELLM_API_KEY"]
    content = [{"type": "text", "text": PROMPT}]
    for b64 in images_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    resp = requests.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 50,
            "temperature": 0,
        },
        timeout=timeout,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        # sam3.1 turns out to be a real Segment-Anything model, not a
        # general vision-QA endpoint: it 400s on this chat-style request
        # every time ("Invalid SAM options" / "at most one image"),
        # regardless of image count - confirmed empirically against the
        # real endpoint. The original design assumed a parseable low-
        # confidence JSON reply to trigger the gpt6-luna fallback below;
        # in practice it's always a hard HTTP error, which this converts
        # into the same (None, None, None) "couldn't resolve" signal so
        # verify_segment's existing fallback still fires instead of the
        # whole call crashing.
        return None, None, None
    text = resp.json()["choices"][0]["message"]["content"]
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        parsed = json.loads(text[start:end])
        return (
            bool(parsed.get("racing")),
            float(parsed.get("confidence", 0.5)),
            parsed.get("vehicle_type", "unknown"),
        )
    except (ValueError, KeyError):
        return None, None, None


def verify_segment(video_path, start_sec, end_sec, sam_model=None, luna_model=None):
    """Resolve one 'uncertain' segment with the minimum possible paid calls:
    1 SAM3.1 call on a handful of representative frames. Only if SAM3.1
    itself comes back low-confidence, escalate the SAME frames to gpt6-luna.
    Returns (is_racing: bool, resolved_by: str, vehicle_type: str).
    """
    mid = (start_sec + end_sec) / 2
    span = max(0.5, (end_sec - start_sec) / 4)
    timestamps = sorted(set(
        max(start_sec, min(end_sec, t))
        for t in (start_sec, mid - span, mid, mid + span, end_sec)
    ))[:4]

    frames_b64 = extract_frames_b64(video_path, timestamps)
    if not frames_b64:
        return False, "no_frames", "none"

    sam_model = sam_model or os.environ.get("LITELLM_SAM_MODEL", "sam3.1")
    is_racing, confidence, vehicle_type = _call_vision_model(sam_model, frames_b64)
    if is_racing is not None and confidence is not None and confidence >= 0.6:
        return is_racing, "sam3.1", vehicle_type

    luna_model = luna_model or os.environ.get("LITELLM_LUNA_MODEL", "gpt6-luna")
    is_racing, _confidence, vehicle_type = _call_vision_model(luna_model, frames_b64)
    if is_racing is not None:
        return is_racing, "gpt6-luna", vehicle_type

    return False, "unresolved", "none"
