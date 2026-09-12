# Historical E1 disposition — 2026-09-11

The immutable run at `outputs/pq-setformer-20260911-044559/` is retained as
diagnostic evidence only.  Its controller record and all downloaded artifacts
are intentionally unmodified.

- Disposition: `gate_failed`.
- H1: consumed.  It must never be used for calibration, recipe selection, or
  another challenger decision.
- Reason: the checkpoint used the v1 candidate schema, discarded mask
  probabilities, had no neural conditional-IoU head, and achieved PQ 0.05318.
- Migration: its checkpoint cannot load into the E2 model (`conditional_iou_head`
  parameters and the patched upstream contract are required).  Its candidate
  cache is readable only by the legacy audit path and cannot be used for E3:
  v1 has no retained probability/logit payload, presence logit, query source,
  or candidate provenance hash.

Any future calibration uses only newly generated Q-fit/Q-cal v2 caches.  A
future H assessment is a separate frozen, one-shot stage after Q-cal selection.
