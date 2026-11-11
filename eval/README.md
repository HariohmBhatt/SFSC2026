# Final V4 OOF morphology evaluation

Open `SFSC2026_Morphology_Eval.ipynb` locally or in a **CPU Colab runtime**.
The notebook imports the evaluation, figure, summary and validation helpers from
this directory. Install `requirements-eval.txt` if the evaluation dependencies
are absent. No GPU, model fitting, external labels, or test images are used.

Set the notebook's dataset, Drive mirror, OOF map, scratch, and results paths.
Required inputs are the competition training JSON/images, V3 OOF probability
maps, and the frozen fusion decision artifacts. The compact decision artifacts
are bundled under `results/inputs/`; alternatively point at the original Drive
mirror's `outputs/sfsc_v5/fusion_sfsc_unet_v3_aug` and `outputs/sfsc_unet_v3`.
Original V3 OOF `.npz` maps can be used directly if staged `.npy` maps are absent.
The notebook checks the lawful training JSON hash and reproduces **every image's**
saved PQ statistics before morphology. It never calls an all-data scorer on OOF.

Full instance label maps are compressed under the chosen **scratch** directory,
`masks/<image_stem>.npz`, with `v4_evidence` and `v3_rescored` uint16 arrays.
`results/mask_manifest.csv` records their SHA256 hashes. They stay outside this
repository to keep the delivered evaluation small. New Colab sessions recreate
them on CPU. Existing scratch maps are reused only after manifest/input checks.

`PROTOCOL.md` defines matching, missed-object penalties, skeleton lengths,
widths, split/merge relations, strata, human comparisons, and date bootstrap.
The results directory contains all per-record/per-object data (gzip CSV), compact
summary CSV/JSON, and report figures (PNG/PDF). CSV files use native pixels.
`STATUS.md` gives measured report-ready findings and limitations.

Analytic checks: `python -m unittest discover -s eval -p 'test_*.py' -v` from the
repository root. This is an evaluation module, with no training code.
