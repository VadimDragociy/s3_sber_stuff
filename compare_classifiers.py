import argparse
import json
import os

from env_config import load_env_file
import verification
from segment_detector import classify_span


def main():
    ap = argparse.ArgumentParser(
        description="For each already-known 'uncertain' span of a set of local "
                     "videos, run both the classical heuristic (classify_span) "
                     "and the real SAM3.1/gpt6-luna verification on the exact "
                     "same span, so the two can be compared directly."
    )
    ap.add_argument("--cases", required=True,
                     help="JSON file: [{\"video\": \"local/path.mp4\", "
                          "\"spans\": [[s, e], ...]}, ...]")
    ap.add_argument("--out", default="compare_results.json")
    args = ap.parse_args()

    load_env_file()
    with open(args.cases, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    for case in cases:
        video = case["video"]
        for s, e in case["spans"]:
            print(f"=== {os.path.basename(video)} [{s}, {e}] ===")

            classical_racing, classical_metrics = classify_span(video, s, e)
            print(f"  classical: racing={classical_racing} {classical_metrics}")

            try:
                vision_racing, resolved_by, vehicle_type = verification.verify_segment(video, s, e)
                print(f"  vision:    racing={vision_racing} resolved_by={resolved_by} vehicle_type={vehicle_type}")
                vision_error = None
            except Exception as exc:
                vision_racing, resolved_by, vehicle_type = None, None, None
                vision_error = str(exc)
                print(f"  vision:    ERROR {vision_error}")

            agree = (classical_racing == vision_racing) if vision_racing is not None else None
            print(f"  agree={agree}")

            results.append({
                "video": video, "span": [s, e],
                "classical_racing": classical_racing, "classical_metrics": classical_metrics,
                "vision_racing": vision_racing, "vision_resolved_by": resolved_by,
                "vision_vehicle_type": vehicle_type, "vision_error": vision_error,
                "agree": agree,
            })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    n = len(results)
    n_agree = sum(1 for r in results if r["agree"] is True)
    n_disagree = sum(1 for r in results if r["agree"] is False)
    n_errors = sum(1 for r in results if r["vision_error"] is not None)
    print(f"\n{n} spans: {n_agree} agree, {n_disagree} disagree, {n_errors} vision errors. -> {args.out}")


if __name__ == "__main__":
    main()
