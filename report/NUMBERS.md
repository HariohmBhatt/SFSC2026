# Number -> source table for main.tex

Abbreviations:
- V4NB = `v4/SFSC2026_V4_Fusion_UNet_YOLO11_L4_Colab (1).ipynb` (retained cell outputs; cell index = JSON cell number)
- V3NB = `v3/SFSC2026_V3_UNet_EffV2_4fold_L4_Colab.ipynb`
- MIR = `~/sfsc_work/v5fusion/drive_mirror/outputs/`
- MEM = `~/.claude/projects/-home-hariohm-SFSC2026/memory/`
- EVAL = `~/SFSC2026/eval/STATUS.md` and `eval/results/metrics_ci.csv`, `paired_deltas_ci.csv`, `selection_changes.json`
- EFF = `~/SFSC2026/docs/EFFICIENCY.md`
- C4 = `~/sfsc_research/r4/C/C_external.md`

Status: M = measured in a log or output; R = read from a research report or memory (not re-measured by me); I = inferred/estimate; U = unverified.

| Number in report | Source | Status |
|---|---|---|
| 707 images, 1154 records, 8199 instances; 411/145/151 images with 1/2/3 records | V4NB cell 10; V3NB cell 10 | M |
| Folds 177/177/177/176 images | V3NB cell 14 | M |
| 0 of 656 dates span two folds | `~/sfsc_research/r2/B_gap_forensics.md` §1.0 | R |
| PQ engine matches the official scorer to within 0.0002 | MEM sfsc-leaderboard-truth.md | R |
| Single-fold V3 PQ 0.424 to 0.452 (0.4521/0.4243/0.4275/0.4502) | V3NB cell 36 (quick-calibrated, per-fold recipe) | M |
| Acceptance rule +0.003 and 3/4 folds | `v5/V5_PLAN_R2.md` Rules; V4NB config min_gain/min_fold_wins | M |
| Disk radius 0.88*1024 = 901.1 px; 512 radial bins; median kernel 41 at 1/4 res (about 164 px) | MIR sfsc_unet_v3/fold0/config.json; V3NB cell 12; V3 code comment | M |
| Crop 768, tile 1024/768, 40 epochs x 250 steps, eff. batch 8, LR 3e-4/1e-3, wd 0.01, warm-up 2, EMA 0.999, p_pos 0.5, p_hardneg 0.25, rotation p 0.7, scale 0.85-1.15, contact_radius 2 | fold0/config.json | M |
| AdamW (fused), bf16 | V3NB code (`torch.optim.AdamW(param_groups(model), fused=True)`), cell 6 output "AMP: torch.bfloat16" | M |
| 22.1 M parameters | V3NB cell 24 | M |
| Loss weights 0.5 BCE + 1.0 Lovasz + 0.2 edge + 0.5 contact | V3NB cell 26; config | M |
| Candidate geometry t_low 0.4, contact 0.5, min core 16, assign radius 6, offset -1, fill holes | MIR sfsc_unet_v3/frozen_recipe_v3.json | M |
| 12 candidate features | V4NB cell 16 | M |
| YOLO11m-seg 1280 px, 40 epochs, cls gain 1.5, conf floor 0.05, min area 32, det-only floor 0.10, max claimed 0.30, grow 4, claim p 0.2 | V4NB cell 6 Config | M |
| One detector sample per annotator record (1154) | V4NB cell 18 | M |
| HGB 400 iter, LR 0.05, 31 leaves, min leaf 40, L2 1.0; R-weighting | V4NB cell 28 `make_hgb`, VARIANTS | M |
| Keep threshold tau = 0.225 (v4_full) | V4NB cell 28 output | M |
| Selection optimism 0.0005 | MEM sfsc-v5-plan.md (Step 0); r2 B table row 4e | R |
| 1077 test filaments, 5.98/image, 3 empty images | V4NB cell 32 | M |
| V4 OOF 0.4550 / SQ .6816 / RQ .6676 / P .7146 / R .6264 / TP 5136 / FP 2051 / FN 3063 | V4NB cell 28; also reproduced in r2 B_gap_forensics.md | M |
| V3 re-scored .4501/.6811/.6609, P .6880 R .6358, TP 5213 FP 2364 FN 2986 | V4NB cell 28 | M |
| V3 + scorer (cross-fitted) .4484/.6823/.6572, TP 5062 FP 2143 FN 3137 | V4NB cell 28 (v3_replica row); frozen_recipe_v3.json oof_pq_scorer | M |
| V3 grid filter .4461/.6847/.6514, TP 4920 FP 1986 FN 3279 | V3NB cell 40 | M |
| V3 in-sample .486/.6804/.7142, TP 5386 FP 1498 FN 2813 | V3NB cell 50 | M (in-sample) |
| V4 gain +0.0049, CI +0.0022..+0.0075, 4/4 folds | V4NB cell 28/34 | M |
| V1 .3175/.6596/.4814 public 0.28; V2 .4014/.6777/.5923 public 0.35 (dev subsets) | `~/SFSC2026/README.md` (both versions) | R |
| Public V3 0.39, V4 0.40 | README.md; MEM sfsc-leaderboard-truth.md | R |
| Public = OOF - 0.056 +/- 0.005 over 5 submissions | MEM sfsc-leaderboard-truth.md | R |
| Channels each <= 0.006; OOF further in time than test; public subset sd 0.012; gap 4.45 sigma | `~/sfsc_research/r2/B_gap_forensics.md` §0; `v5/V5_PLAN_R2.md` | R |
| Peer 0.4747 OOF / public 0.41 (Salty Taro) | `~/sfsc_research/r4/SUMMARY.md`; r4/B/B_notebooks.md | R (self-reported) |
| Inter-annotator PQ 0.3397, SQ 0.6336, RQ 0.5361, TP 4288 FP 3711 FN 3711 | frozen_recipe_v3.json oracles; V3NB cell 18 | M |
| Organizer stats: 6516 pairs, mean IoU 0.605, Dice 0.736, GT 1882/6135/182, pred 854/6159/174 | V4NB cell 30 | M |
| Table 2 (outcome by size, radius, support) | V4NB cell 30 | M |
| 56% of FPs on multi-record images match another annotator | V4NB cell 30 | M |
| FP split: 1141 near-miss, 910 spurious | V4NB cell 30 ("Predictions by source" table) | M |
| Classical baseline 0.1796 (139 date-grouped holdout images), public 0.15 | C4 §4 (Regmi README snapshot) | R (self-reported) |
| pragyaangaur 0.4568 OOF (folds hashed by name), public 0.39 | C4 §4 | R (self-reported) |
| theainerd 0.4519 date-grouped, public 0.40 | C4 §2/§4 | R (self-reported) |
| biconcavelens 0.461 (70-image repeated calibration splits), public 0.40 | C4 §4 | R (self-reported) |
| Step 0: t_low 0.3 -0.017, Otsu -0.04, offset 0 -0.006 | MEM sfsc-v5-plan.md | R |
| Offsets -1.5/-2: -0.015/-0.058; tree+antichain DP -0.014 | `v5/V5_PLAN_R2.md` Dead list | R |
| YOLO11l 1536 seeds: recall@0.3 0.973; fusion 0.4531 | MEM sfsc-v5-plan.md | R |
| 4-fold refiner 0.4545, SQ 0.688 vs 0.682 | MEM sfsc-v5-plan.md | R |
| Rotated refiner -0.018 (90% CI -0.026..-0.011, 1/4) | V5_PLAN_R2.md step 4 | R |
| BCE+Dice+spine -0.035 (90% CI -0.044..-0.027, 0/4) | V5_PLAN_R2.md step 2 (memory says "CI", plan says "90% CI") | R |
| UNet/R34 average -0.0016 (CI -0.0049..+0.0016, 2/4) | V5_PLAN_R2.md step 3 | R |
| Level-augmented scorer +0.0011 (CI -0.0013..+0.0036, 3/4) | V5_PLAN_R2.md step 1; MIR sfsc_v5/fusion_sfsc_unet_v3_aug/selection_v5.json | M |
| Agreement features -0.0006 (CI -0.0027..+0.0014, 2/4) | MEM sfsc-v5-plan.md (r3 C action 1) | R |
| External detector +0.0001 (CI -0.0026..+0.0028) / +0.0010 (CI -0.0015..+0.0036), 2/4 | r4/SUMMARY.md "Result of action 1" | R |
| Public probes: ~4 kept 0.37, ~12 kept 0.31, offset 0 0.39 | MEM sfsc-v5-plan.md | R |
| Morphology: PQ 0.4545 CI 0.4447-0.4642 (replica), matched IoU 0.680 (0.677-0.683), Dice 0.807 | EVAL | M (eval agent) |
| Spine coverage 60.4/61.7/62.3% at 1/3/5 px; longest uncovered 38.1% at 3 px | EVAL | M (eval agent) |
| Coverage centre 64.0% vs limb 49.4%; support all 87.8% vs none 33.5% | EVAL | M (eval agent) |
| Human reference PQ 0.340, coverage 51.3%; V4 on same images 0.452 | EVAL | M (eval agent) |
| Length ratio 1.24, width ratio 0.84; splits 2.6%, merges 2.3% | EVAL | M (eval agent) |
| V4 vs V3: +382/-245 objects, recall +1.5 pts (CI 1.0-2.0), coverage +1.4 pts, matched IoU -0.001 | EVAL selection_changes.json, paired_deltas_ci.csv | M (eval agent) |
| Replica tau 0.200 vs shipped 0.225; shipped kept 3 detector-only candidates OOF | EVAL (provenance); V4NB cell 28 and cell 30 ("detector+unet pixels: 3") | M |
| 656 date clusters, 2000 draws | EVAL | M |
| YOLO training 0.951-0.957 h per fold, 3.82 L4-h total | EFF (from V4NB cell 20 outputs) | M |
| UNet about 130 s/epoch, about 5.8 L4-h training; about 1.4 L4-h dense maps; about 11 L4-h total | EFF | I |
| 288 UNet tile passes + 4 detector passes per test image | EFF; V3NB cell 28 (9 tiles, 8 TTA) | M (count) |
| Detector GPU memory about 9.6 GB; L4 22,563 MiB | V4NB cell 20 output | M |
| Example image 20110306082634Lh, fold 1, records with 6/7/6 filaments | `figures/make_fig_data.py` output | M |
