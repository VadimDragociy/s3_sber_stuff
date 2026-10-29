# SAM3.1/gpt6-luna vs. classical heuristic - comparison test

Tested against `https://litellmru.ds-hub-cf.ru` with a real API key, on 14
"uncertain" spans from 9 videos adjacent to (but distinct from) the
IgoraDrive C-04 series used to validate the classical heuristic on the
`no-vision-verification-local-clips` branch. Each span was resolved three
ways: the classical heuristic (`classify_span`), the real vision call
(`verification.verify_segment`), and manual visual inspection of multiple
frames across the span (not just the midpoint - several cars turned out to
be visible only briefly near a span's edge, which a single midpoint check
missed on the first pass).

## sam3.1 does not work as integrated

`sam3.1` is a real Segment-Anything model, not a general vision-QA
endpoint. Every call to it returned HTTP 400:
- `Invalid SAM options; see documented fields and limits` (1 image)
- `Use at most one text, one image and one mask` (4 images)

The original code's `resp.raise_for_status()` let this propagate as an
uncaught `HTTPError`, so `verify_segment` would have crashed on the first
uncertain segment it ever tried to resolve in production - the "only
escalate to gpt6-luna on low SAM confidence" design never worked, because
SAM never returns a parseable confidence; it always hard-errors. This test
branch patches that to treat the error as "unresolved" so the existing
fallback fires, but the practical effect is that **every single
verification call costs one full gpt6-luna request** (plus a wasted SAM
request) - the two-tier cost-saving design doesn't save anything with this
endpoint's actual sam3.1 behavior.

Also: the working Luna model's real id is `chatgpt/gpt-6-luna`, not the
`gpt6-luna` default hardcoded in `verification.py`.

## Accuracy: 14/14 for gpt6-luna vs. 10/14 for the classical heuristic

| span | ground truth (manual) | classical | gpt6-luna |
|---|---|---|---|
| 19_04_14_000 [7.91,10.78] | empty | False ✓ | False ✓ |
| 19_04_14_000 [10.94,12.01] | empty | False ✓ | False ✓ |
| 19_04_14_001 [10.71,12.01] | car (brief, near start) | True ✓ | True ✓ |
| 19_04_14_002 [0.0,4.47] | car | True ✓ | True ✓ |
| 19_04_14_002 [6.27,7.74] | car | True ✓ | True ✓ |
| 19_04_14_002 [9.54,12.41] | empty | False ✓ | False ✓ |
| 19_04_14_002 [13.75,15.21] | car (enters mid-span) | **False ✗** | True ✓ |
| 19_05_27_000 [2.3,4.7] | car | True ✓ | True ✓ |
| 19_05_27_001 [0.0,2.6] | empty | **True ✗** | False ✓ |
| 19_05_27_001 [8.38,9.84] | empty | False ✓ | False ✓ |
| 19_06_14_000 [7.44,12.01] | car (brief, near start) | **False ✗** | True ✓ |
| 19_06_14_001 [11.88,13.35] | car | True ✓ | True ✓ |
| 19_06_56_000 [0.0,2.84] | car | True ✓ | True ✓ |
| 19_06_56_001 [7.91,9.38] | car | **False ✗** | True ✓ |

**gpt6-luna: 14/14 (100%). classical heuristic: 10/14 (71.4%)** - 3 false
negatives (missed a real car), 1 false positive (a near-static glint).

## Why the classical heuristic misses cars

All 3 classical false negatives are the same failure mode: a car is
visible for only part of the span (often right at its start or end, the
rest of the window genuinely empty), and the heuristic's signals
(texture/shape/motion, computed as a MEDIAN or AVERAGE over the whole
track) get diluted by the empty portion. A single midpoint frame check
missed this too on the first pass of manual verification here - it took
sampling multiple points across each span to catch it.

## Conclusion

On this test set, real vision verification is meaningfully more accurate
than the classical heuristic (100% vs. 71.4%), and the cost is not "maybe
one cheap SAM call" as originally designed - it's one gpt6-luna call per
uncertain segment, always. Whether that's worth it depends on how much the
~5 false-positive/negative percentage points matter at the target scale
(44k+ videos) versus the token cost estimated earlier (~150-300M tokens
for the full bucket) - this file reports the accuracy side of that
tradeoff, not a recommendation.
