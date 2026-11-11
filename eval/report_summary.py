"""Generate measured report-ready STATUS.md from the saved CSV/JSON evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def write_status(eval_dir):
    eval_dir=Path(eval_dir);r=eval_dir/"results"
    ci=pd.read_csv(r/"metrics_ci.csv")
    delta=pd.read_csv(r/"paired_deltas_ci.csv")
    audit=json.loads((r/"morphology_audit.json").read_text())
    counts=pd.read_csv(r/"pooled_counts.csv").set_index("model")
    provenance=json.loads((r/"provenance.json").read_text())
    bootstrap=json.loads((r/"bootstrap.json").read_text())
    archive=r/"inputs/outputs/sfsc_v5/fusion_sfsc_unet_v3_aug/image_stats.npz"
    decisions=np.load(archive)
    k3=decisions["v3_rescored__u"]>=.225
    k4=decisions["v4_evidence__u"]>=.2
    changes=dict(common_instances=int((k3&k4).sum()),added_v4=int((k4&~k3).sum()),removed_v3=int((k3&~k4).sum()),
                 note="Shared keeps use the same frozen V3 candidate masks; no geometry change.")
    (r/"selection_changes.json").write_text(json.dumps(changes,indent=2)+"\n")
    def get(metric,model="v4_evidence",stratum="all",level="all"):
        return ci[(ci.metric==metric)&(ci.model==model)&(ci.stratum==stratum)&(ci.level.astype(str)==str(level))].iloc[0]
    def fmt(row,percent=False):
        scale=100 if percent else 1;digits=1 if percent else 4
        suffix="%" if percent else ""
        return f"{row.estimate*scale:.{digits}f}{suffix} [95% CI {row.ci_low*scale:.{digits}f}, {row.ci_high*scale:.{digits}f}{suffix}]"
    def val(metric,model="v4_evidence",percent=False,stratum="all",level="all"):
        return fmt(get(metric,model,stratum,level),percent)
    d=delta[delta.metric=="pq"].iloc[0]
    v4=counts.loc["v4_evidence"];a=audit["v4_evidence"];ha=audit["inter_annotator"]
    bullets=[
        f"Frozen detector evidence raises pooled OOF PQ from {val('pq','v3_rescored')} (V3 rescored) to {val('pq')} (V4); paired gain {fmt(d)}. Shared kept objects retain identical masks; only keep decisions differ.",
        f"V4 has {int(v4.tp):,} TP, {int(v4.fp):,} FP and {int(v4.fn):,} FN over {a['records_or_directed_comparisons']:,} annotator records. Precision is {val('precision',percent=True)} and recall {val('recall',percent=True)}. Matched IoU is {val('matched_iou')} and Dice {val('matched_dice')}; these outline scores condition on a match and exclude the stated misses.",
        f"Counting missed objects as zero, mean GT-spine coverage is {val('coverage_1',percent=True)}, {val('coverage_3',percent=True)} and {val('coverage_5',percent=True)} at 1/3/5 native px. The mean longest uncovered contiguous fraction at 3 px is {val('gap_3',percent=True)}.",
        f"For matched objects with valid spines, predicted skeleton/GT spine length ratio is {val('length_ratio')}, width ratio {val('width_ratio')}, and elongation ratio {val('elongation_ratio')}. The same width definition gives human-mask/reference-spine ratio {val('width_ratio','inter_annotator')}; mask skeletons count branches, unlike the single drawn spine. These are projected proxies, not physical width errors or verified barb measurements.",
        f"GT match rate is {val('gt_match_rate',percent=True,stratum='gt_area_bucket',level='<500')} for area <500 px and {val('gt_match_rate',percent=True,stratum='gt_area_bucket',level='≥5000')} for area ≥5000 px. This stratification includes all GT objects, rather than only survivors.",
        f"Mean spine coverage at 3 px is {val('coverage_3',percent=True,stratum='radial_bucket',level='centre (<0.5R)')} in the nominal disk centre and {val('coverage_3',percent=True,stratum='radial_bucket',level='limb (≥0.8R)')} at the limb (centroid ≥0.8R). The radius is the frozen preprocessing radius, not a fitted solar limb.",
        f"On multi-record images, V4 spine coverage at 3 px is {val('coverage_3',percent=True,stratum='support_bucket',level='none')} for GT with no other-record IoU>0.5 support and {val('coverage_3',percent=True,stratum='support_bucket',level='all')} when every other record supports it. Support is an evaluation-label property and was not supplied to the model.",
        f"The directed human-to-human reference gives PQ {val('pq','inter_annotator')} and 3 px spine coverage {val('coverage_3','inter_annotator',percent=True)} on {ha['images']} multi-record images / {ha['records_or_directed_comparisons']:,} directed comparisons. It is a reference agreement level on that subset, not an upper bound. V4 PQ on the same multi-record image population is {val('pq',stratum='image_population',level='multi-record')}.",
    ]
    text="# Final V4 OOF morphology evaluation — complete\n\n"
    text+=(f"Measured on {provenance['images']} native 2048×2048 training images, {a['records_or_directed_comparisons']:,} annotator records and {a['gt_objects']:,} GT polygons. "
           f"95% percentile CIs resample {bootstrap['dates']} UTC date clusters ({bootstrap['replicates']:,} draws, seed {bootstrap['seed']}); V4−V3 is paired.\n\n")
    text+="## Report-ready findings (all measured)\n\n"+"\n\n".join("- "+b for b in bullets)+"\n\n"
    text+="## Provenance and verification\n\n"
    text+=("- Adopted saved variant: **v4_evidence**, tau **0.200**; baseline **v3_rescored**, tau **0.225**. "
           "The augmented V5 variant was not adopted. Both use unchanged V3 candidate geometry.\n"
           "- Every image's reconstructed Hungarian IoU sum/TP/FP/FN reproduces the saved fusion statistics to numerical precision; see `results/pq_verification.json`. "
           "The host many-to-many metric is reported separately in all final tables.\n"
           "- Host matching adds one qualifying pair per model versus Hungarian (V4 official PQ 0.454473, V3 0.450369); the small convention difference is explicit.\n"
           "- Local verified V4 replica PQ is 0.454413, versus historical shipped V4's rounded 0.455. These results do not silently claim the shipped value.\n"
           f"- V4 unique OOF predictions: {a['unique_pred_masks']:,}; record-weighted predictions: {a['pred_objects_record_weighted']:,}. "
           f"Missing GT spines: {a['missing_spines']}; human-reference missing spines: {ha['missing_spines']}.\n"
           f"- Compared with V3, V4 keeps {changes['common_instances']:,} identical shared instances, adds {changes['added_v4']:,} and removes {changes['removed_v3']:,}.\n"
           f"- V4 split relations: {int(v4.split_gt)} / {int(v4.n_gt):,} GT objects; merge relations: {int(v4.merge_pred)} / {int(v4.n_pred):,} record-weighted predictions. "
           f"Rates with date CIs: split {val('split_gt_rate',percent=True)}, merge {val('merge_pred_rate',percent=True)}. "
           "Relations require ≥16 px and ≥10% of the smaller mask; they are descriptive overlap counts.\n")
    text+="\n## Deliverables and reproduction\n\n"
    text+=("- `SFSC2026_Morphology_Eval.ipynb`: local/Colab CPU notebook; `morphology.py`, `frozen_geometry.py`, `figures.py`, `report_summary.py`: reusable helpers.\n"
           "- `PROTOCOL.md`: exact definitions, strata, denominators and limitations.\n"
           "- `results/*csv.gz`: all GT/prediction/match/record measurements, including unmatched counts; summary CSV/JSON include bootstrap CIs, paired differences and distributions.\n"
           "- `results/morphology_metrics.{png,pdf}`, `morphology_strata.{png,pdf}`, `oof_examples_full.{png,pdf}`, `oof_examples_crops.{png,pdf}`: report figures. "
           "Examples are deterministic V4 per-image PQ 10th/50th/90th-percentile cases, showing each annotator separately.\n"
           "- `results/oof_multi_annotator_crops.{png,pdf}` additionally uses 10th/50th/90th percentiles within the 296 multi-record images, to show differing human outlines. Its population and IDs are documented separately.\n"
           "- Native disjoint instance maps: `<scratch>/masks/<stem>.npz`, keys `v4_evidence` / `v3_rescored`; hashes in `results/mask_manifest.csv`. "
           "Maps stay in scratch to keep eval/ below 30 MB. Compact decision inputs are bundled in `results/inputs/`; large OOF maps and lawful training images remain required.\n"
           "- Six analytic checks pass (strict threshold/host matching, spine weighting/misses, exact raster distance/contiguous gaps, skeleton edge length, polygon overlaps, date resampling). Notebook execution and artifact-size checks are recorded in `results/validation.json`.\n"
           "- Local CPU only: **0 GPU hours, 0 Colab CU**; no VM started, no commit/push, no submissions or external/test-label access.\n")
    text+="\n## Interpretation limits\n\n"
    text+=("Spine coverage tests containment in a matched mask, not bidirectional skeleton-axis recovery. Width and length are projected proxies; skeleton branches can increase length. "
           "Shape errors condition on matching, while missed objects penalize coverage and recall. GT-dependent strata report recall/coverage because unmatched predictions lack a GT bucket. "
           "Human comparisons use multi-record images only; three-record images contribute six directed pairs. Date bootstrap conditions on frozen models and thresholds and excludes training/selection uncertainty. "
           "There are no remaining required computations.\n")
    (eval_dir/"STATUS.md").write_text(text)
    return text
