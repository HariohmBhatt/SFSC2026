# SFSC control error-stage audit

Generated: **2026-09-09T06:24:35.345362+00:00**.

This is a truth-assisted historical validation diagnostic. It is not deployment logic and cannot promote a model.

## Overall attribution

Across 169 annotation records and 1237 truths, the control matched 723 and missed 514 under one-to-one strict IoU > 0.5 matching.

| Stage | Count | Share of misses |
|---|---:|---:|
| score_rejection | 256 | 49.8% |
| insufficient_mask_quality | 103 | 20.0% |
| ambiguous_or_merged_proposal | 60 | 11.7% |
| no_detector_proposal | 40 | 7.8% |
| refinement_damage | 28 | 5.4% |
| assembly_loss | 27 | 5.3% |
| matched | 723 | — |

## Misses by truth size

| Size | Truths | Matched | Misses | ambiguous_or_merged_proposal | assembly_loss | insufficient_mask_quality | no_detector_proposal | refinement_damage | score_rejection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| large:>=4096 | 148 | 85 | 63 | 0 | 10 | 25 | 0 | 4 | 24 |
| medium:1024-4095 | 581 | 388 | 193 | 12 | 11 | 58 | 11 | 19 | 82 |
| small:256-1023 | 483 | 246 | 237 | 45 | 6 | 20 | 24 | 5 | 137 |
| tiny:<256 | 25 | 4 | 21 | 3 | 0 | 0 | 5 | 0 | 13 |

## Decision rule

The largest actionable bucket is **quality/threshold selection** (256 attributed misses). This selects the next isolated experiment; it does not estimate leaderboard gain.

## Limitations

- The cache begins after detector box NMS and seed-mask deduplication, so losses inside those operations are included in no_detector_proposal.
- Every annotation record is evaluated independently, matching the competition aggregation.
- Truth-assisted attribution and oracle quantities cannot be used during test inference.
