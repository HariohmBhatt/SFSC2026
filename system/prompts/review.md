# Model fleet 2 — Approach Validation

Read every candidate and its cited academic evidence. Compare all candidates
using the same training envelope. Return one review per candidate, exactly once.
Estimate training hours, peak VRAM in GB, serialized model size in MB, and
implementation complexity from 1 (small) to 5 (large). Give PQ potential from
0 to 1 as an evidence-based judgment, explicitly not measured PQ. State estimate
assumptions (crop size, batch size, steps, parameter count), uncertainty, research
weaknesses, and the expected return on training time. Set `compatible` only when
the implementation fits the TensorFlow Kaggle P100 runtime in GPU.md.

The system applies the saved weights and hard resource limits, records the full
ranking and chooses the highest scoring feasible candidate. You supply the
judgments and rationale; measured validation PQ remains the experimental goal.
Complete only when every approach has all five comparison dimensions and a
reasoned verdict. Return exactly the JSON object in `response_contract`.
