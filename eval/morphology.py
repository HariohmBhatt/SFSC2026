"""CPU-only, native-resolution evaluation of frozen SFSC OOF instance masks.

No training, threshold selection, or external annotations. See PROTOCOL.md for
denominators and the distinction between host matching and one-to-one morphology.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import pandas as pd
from pycocotools import mask as coco
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

try:
    from . import frozen_geometry as frozen
except ImportError:
    import frozen_geometry as frozen

MODELS = ("v3_rescored", "v4_evidence")
SEED = 20261024


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def annotation_index(dataset):
    path = Path(dataset) / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    raw = json.loads(path.read_text())
    anns = defaultdict(list)
    for a in raw["annotations"]:
        anns[str(a["image_id"])].append(a)
    records = defaultdict(list)
    for im in raw["images"]:
        records[Path(im["file_name"]).name].append({
            "id": str(im["id"]), "annotations": anns[str(im["id"])],
            "shape": (int(im["height"]), int(im["width"])),
        })
    for r in records.values():
        r.sort(key=lambda x: x["id"])
    return path, records


def rle_of(ann, shape):
    h, w = shape
    seg = ann["segmentation"]
    if isinstance(seg, list):
        seg = seg if isinstance(seg[0], list) else [seg]
        return coco.merge(coco.frPyObjects(seg, h, w))
    if isinstance(seg["counts"], list):
        return coco.frPyObjects(seg, h, w)
    return {"size": [h, w], "counts": seg["counts"].encode() if isinstance(seg["counts"], str) else seg["counts"]}


def record_masks(rec):
    return [coco.decode(rle_of(a, rec["shape"])).astype(bool) for a in rec["annotations"]]


def overlap(masks, pred_masks):
    """Exact polygon-raster overlap; handles overlapping human instances."""
    gr = [coco.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks]
    pr = [coco.encode(np.asfortranarray(m.astype(np.uint8))) for m in pred_masks]
    if not gr or not pr:
        z = np.zeros((len(gr), len(pr)))
        return z, z.copy()
    iou = coco.iou(gr, pr, [0] * len(pr))
    ga, pa = np.asarray(coco.area(gr)), np.asarray(coco.area(pr))
    inter = np.rint(iou * (ga[:, None] + pa[None, :]) / (1 + iou))
    return inter, iou


def label_overlap(masks, labels):
    n = int(labels.max())
    area = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    inter = np.array([np.bincount(labels[m], minlength=n + 1)[1:] for m in masks], dtype=float)
    inter = inter.reshape(len(masks), n)
    ga = np.array([m.sum() for m in masks])
    union = ga[:, None] + area[None, :] - inter
    return inter, np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def pairs(iou):
    if not iou.size:
        return []
    sub = np.where(iou > .5, iou, 0.)
    r, c = linear_sum_assignment(sub, maximize=True)
    return [(int(g), int(p), float(sub[g, p])) for g, p in zip(r, c) if sub[g, p] > 0]


def counts(iou, official=True):
    if official:
        hit = iou > .5
        return np.array([iou[hit].sum(), hit.sum(), (~hit.any(0)).sum(), (~hit.any(1)).sum()], float)
    p = pairs(iou)
    return np.array([sum(v for _, _, v in p), len(p), iou.shape[1] - len(p), iou.shape[0] - len(p)], float)


def summary(st):
    u, tp, fp, fn = np.asarray(st, float)[:4]
    den = tp + .5 * (fp + fn)
    return dict(pq=float(u / den) if den else 1., sq=float(u / tp) if tp else 0.,
                rq=float(tp / den) if den else 1., tp=int(tp), fp=int(fp), fn=int(fn),
                precision=float(tp / (tp + fp)) if tp + fp else 0.,
                recall=float(tp / (tp + fn)) if tp + fn else 0.)


def map_pair(maps, fusion_root, fold, stem):
    folder = Path(maps) / f"oof_fold{fold}"
    paths = [folder / f"{stem}_{k}.npy" for k in ("fg", "ct")]
    if all(p.is_file() for p in paths):
        return [np.load(p, mmap_mode="r") for p in paths]
    matches = list((Path(fusion_root) / "outputs/sfsc_unet_v3/oof").glob(f"fold{fold}_*/{stem}.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected staged maps or one original OOF npz for {stem}: {matches}")
    with np.load(matches[0]) as z:
        return [z[k] for k in ("fg", "ct")]


def _reconstruct_one(task):
    name, fold, maps, root, config, geometry, us, src, taus, recs, dest = task
    frozen.configure(config)
    fg, ct = map_pair(maps, root, fold, Path(name).stem)
    lab, n = frozen.geometry_labels(fg, ct, geometry)
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    maxima = np.zeros(n + 1)
    np.maximum.at(maxima, lab.ravel(), np.asarray(fg).ravel())
    # Only max_p and area are used by the frozen prefilter. Preserve its float32 rounding.
    feats = np.zeros((n, 3), np.float32)
    feats[:, 2] = maxima[1:] / 255.
    keep = frozen.prefilter_keep(feats, area)
    lut = np.zeros(n + 1, np.uint16)
    lut[np.flatnonzero(keep) + 1] = np.arange(1, keep.sum() + 1)
    lab = lut[lab]
    nv3 = int(keep.sum())
    assert nv3 == int((src == 0).sum()), (name, nv3, (src == 0).sum())
    assert np.all(src[:nv3] == 0) and np.all(src[nv3:] != 0), name
    out, rows = {}, []
    for model in MODELS:
        assert not np.any(us[model][nv3:] >= taus[model]), "Unexpected detector-only keep in evidence/baseline"
        ids = np.flatnonzero(us[model][:nv3] >= taus[model]) + 1
        lut = np.zeros(nv3 + 1, np.uint16)
        lut[ids] = np.arange(1, len(ids) + 1)
        out[model] = lut[lab]
    for rec in recs:
        if not rec["annotations"]:
            continue
        masks = record_masks(rec)
        for model in MODELS:
            _, iou = label_overlap(masks, out[model])
            for engine in ("official", "fusion_hungarian"):
                st = counts(iou, engine == "official")
                rows.append(dict(file_name=name, record=rec["id"], date=name[:8], site=Path(name).stem[14:],
                                 fold=int(fold), n_annotators=len(recs), model=model, engine=engine,
                                 sum_iou=st[0], tp=int(st[1]), fp=int(st[2]), fn=int(st[3])))
    np.savez_compressed(Path(dest) / f"{Path(name).stem}.npz", **out)
    return rows


def reconstruct(dataset, fusion_root, fusion_run, maps, scratch, results, workers=4, limit=None):
    """Reconstruct with saved OOF u, never with all-data final scorer.pkl."""
    results, scratch = Path(results), Path(scratch)
    results.mkdir(parents=True, exist_ok=True)
    dest = scratch / "masks"
    dest.mkdir(parents=True, exist_ok=True)
    ann_path, records = annotation_index(dataset)
    run = Path(fusion_run)
    sel = json.loads((run / "selection_v5.json").read_text())
    assert sel["adopted"] == "v4_evidence", "This evaluator is frozen to the adopted evidence variant."
    assert sel["maps"] == ["sfsc_unet_v3"]
    v3 = Path(fusion_root) / "outputs/sfsc_unet_v3"
    cfg = json.loads((v3 / "fold0/config.json").read_text())["config"]
    frozen_rec = json.loads((v3 / "frozen_recipe_v3.json").read_text())["recipe"]
    assert sha256(ann_path) == frozen_rec["annotation_sha256"]
    geometry = frozen_rec["geometry"]
    geom = (geometry["t_low"], geometry["contact"], geometry["offset"])
    taus = {r["variant"]: r["tau"] for r in sel["summary"] if r["variant"] in MODELS}
    z = np.load(run / "image_stats.npz")
    names = z["names"].tolist()
    assert set(names) == set(records), "OOF evaluation must cover every training image."
    folds = pd.read_csv(v3 / "folds_v3.csv")
    assert folds.groupby(folds.file_name.str[:8]).fold.nunique().max() == 1
    rows, tasks = [], []
    for i, name in enumerate(names[:limit] if limit else names):
        a, b = z["off"][i:i+2]
        us = {m: z[m + "__u"][a:b] for m in MODELS}
        tasks.append((name, int(z["folds"][i]), maps, fusion_root, cfg, geom,
                      us, z["src"][a:b], taus, records[name], str(dest)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, r in enumerate(pool.map(_reconstruct_one, tasks)):
            rows.extend(r)
            if (i + 1) % 50 == 0 or i + 1 == len(tasks):
                print(f"Masks + PQ verified: {i+1}/{len(tasks)}", flush=True)
    df = pd.DataFrame(rows)
    checks = {}
    for model in MODELS:
        sub = df[(df.model == model) & (df.engine == "fusion_hungarian")]
        got = sub.groupby("file_name")[["sum_iou", "tp", "fp", "fn"]].sum().reindex(names[:len(tasks)]).to_numpy()
        ref = z[f"{model}__tau{taus[model]:g}"][:len(tasks), :4]
        np.testing.assert_allclose(got, ref, atol=1e-9, rtol=1e-12)
        checks[model] = {"fusion_hungarian": summary(got.sum(0)),
                         "official": summary(df[(df.model == model) & (df.engine == "official")][["sum_iou", "tp", "fp", "fn"]].sum()),
                         "max_per_image_error": float(abs(got-ref).max()), "tau": taus[model]}
    df.to_csv(results / "pq_records.csv", index=False, float_format="%.12g")
    write_json(results / "pq_verification.json", checks)
    write_json(results / "provenance.json", {
        "annotation_sha256": sha256(ann_path), "image_stats_sha256": sha256(run / "image_stats.npz"),
        "selection_sha256": sha256(run / "selection_v5.json"), "geometry_module_sha256": sha256(Path(__file__).with_name("frozen_geometry.py")),
        "fusion_run": str(run), "maps": str(maps), "images": len(tasks), "full_images": len(names),
        "records": sum(bool(r["annotations"]) for n in names[:len(tasks)] for r in records[n]),
        "geometry": geometry, "scorer_refitted": False, "gpu_hours": 0, "colab_cu": 0,
        "mask_directory": str(dest), "native_shape": cfg["expected_shape"], "date_folds_disjoint": True,
        "map_source": "staged uint8 V3 OOF maps; original NPZ fallback", "seed": SEED,
    })
    print(json.dumps(checks, indent=2), flush=True)
    return df


def spine_samples(ann, spacing=.5):
    p = np.asarray(ann.get("spine", []), float)
    if p.size < 4 or p.size % 2 or not np.isfinite(p).all():
        return None
    p = p.reshape(-1, 2)
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    p = p[np.r_[True, d > 1e-10]]
    if len(p) < 2:
        return None
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    s = np.linspace(0., arc[-1], int(np.ceil(arc[-1] / spacing)) + 1)
    xy = np.stack([np.interp(s, arc, p[:, k]) for k in (0, 1)], 1)
    edges = np.r_[0., (s[:-1] + s[1:]) / 2, arc[-1]]
    return xy, np.diff(edges), float(arc[-1])


def shape_features(mask, radius_fraction=.88):
    y, x = np.nonzero(mask)
    area = len(x)
    if not area:
        raise ValueError("Empty instance")
    h, w = mask.shape
    radial = float(np.hypot(y.mean() - (h-1)/2, x.mean() - (w-1)/2) / (radius_fraction * min(h,w)/2))
    cov = np.cov(np.stack([y, x]), bias=True) if area > 1 else np.zeros((2,2))
    ev = np.maximum(np.linalg.eigvalsh(cov), 1e-3)
    elongation = float(np.sqrt(ev[-1] / ev[0]))
    y0, y1, x0, x1 = y.min(), y.max()+1, x.min(), x.max()+1
    crop = np.pad(mask[y0:y1, x0:x1], 1)
    sk = skeletonize(crop)
    length = float((sk[:, :-1] & sk[:, 1:]).sum() + (sk[:-1] & sk[1:]).sum())
    # Suppress diagonal shortcuts across an existing orthogonal skeleton corner.
    d1 = sk[:-1,:-1] & sk[1:,1:] & ~sk[:-1,1:] & ~sk[1:,:-1]
    d2 = sk[:-1,1:] & sk[1:,:-1] & ~sk[:-1,:-1] & ~sk[1:,1:]
    length += float((d1.sum() + d2.sum()) * np.sqrt(2))
    distance = ndi.distance_transform_edt(crop)
    return dict(area=float(area), radial=radial, elongation=elongation,
                skeleton_length=length, skeleton_pixels=int(sk.sum()),
                width_area_length=float(area/length) if length else np.nan,
                width_edt=float(2*np.median(distance[sk])) if sk.any() else np.nan,
                skeleton_components=int(ndi.label(sk, np.ones((3,3)))[1]))


def coverage(samples, mask):
    if samples is None:
        return {f"{kind}_{t}": np.nan for t in (1,3,5) for kind in ("coverage", "gap")}
    xy, weights, length = samples
    if mask is None:
        return {f"{kind}_{t}": float(kind == "gap") for t in (1,3,5) for kind in ("coverage", "gap")}
    # Coordinates in polygons/spines are image coordinates; raster cell centres are x+.5,y+.5.
    y, x = np.nonzero(mask)
    dist = cKDTree(np.stack([x+.5, y+.5],1)).query(xy, k=1)[0]
    out = {}
    for t in (1,3,5):
        hit = dist <= t
        starts = np.flatnonzero(np.r_[True, hit[1:] != hit[:-1]])
        ends = np.r_[starts[1:], len(hit)]
        longest = max([float(weights[a:b].sum()) for a,b in zip(starts,ends) if not hit[a]] or [0.])
        out[f"coverage_{t}"] = float(weights[hit].sum()/length)
        out[f"gap_{t}"] = longest/length
    return out


def bucket(value, edges, labels):
    return labels[int(np.searchsorted(edges, value, side="right"))] if np.isfinite(value) else "missing"


def strata(feat, spine_length=np.nan, support=np.nan, n_annotators=1):
    width = feat["area"] / spine_length if spine_length > 0 else np.nan
    return dict(gt_area_bucket=bucket(feat["area"], [500,1500,5000], ["<500","500–1499","1500–4999","≥5000"]),
                gt_width_bucket=bucket(width, [5,10,20], ["<5","5–<10","10–<20","≥20"]),
                radial_bucket=bucket(feat["radial"], [.5,.8], ["centre (<0.5R)","middle (0.5–<0.8R)","limb (≥0.8R)"]),
                support_bucket=("single record" if n_annotators == 1 else
                                "none" if support == 0 else "all" if support == 1 else "partial"),
                gt_width=width)


def _comparison(base, model, rec, gt_masks, pred_masks, gt_feat, pred_feat, samples, supports, inter, iou, pred_ids):
    assignment = {g:(p,v) for g,p,v in pairs(iou)}
    matched_pred = {p for p,_ in assignment.values()}
    # A relation requires >=16 native pixels AND >=10% of the smaller instance.
    ga = np.array([f["area"] for f in gt_feat])
    pa = np.array([f["area"] for f in pred_feat])
    edges = (inter >= 16) & (inter >= .1 * np.minimum(ga[:,None],pa[None,:]))
    splits = edges.sum(1) >= 2
    merges = edges.sum(0) >= 2
    gt_rows, pred_rows, match_rows = [], [], []
    st = counts(iou)
    recrow = dict(**base, model=model, record=rec["id"], sum_iou=st[0], tp=int(st[1]), fp=int(st[2]), fn=int(st[3]),
                  split_gt=int(splits.sum()), merge_pred=int(merges.sum()), n_gt=len(gt_masks), n_pred=len(pred_masks),
                  one_to_one_matches=len(assignment), ambiguous_threshold_pairs=int((iou>.5).sum()-len(assignment)))
    for g, ann in enumerate(rec["annotations"]):
        p, val = assignment.get(g, (None, 0.))
        sf = samples[g]
        glen = sf[2] if sf else np.nan
        row = dict(**base, model=model, record=rec["id"], annotation=ann["id"],
                   matched=int(p is not None), official_matched=int((iou[g]>.5).any()), iou=val,
                   dice=2*val/(1+val) if val else 0., best_iou=float(iou[g].max()) if len(pred_masks) else 0.,
                   spine_length=glen, spine_samples=len(sf[0]) if sf else 0, support=supports[g],
                   split=int(splits[g]), overlap_degree=int(edges[g].sum()),
                   **{"gt_"+k:v for k,v in gt_feat[g].items()},
                   **strata(gt_feat[g], glen, supports[g], base["n_annotators"]),
                   **coverage(sf, pred_masks[p] if p is not None else None))
        for key in ("skeleton_length", "width_area_length", "width_edt", "elongation"):
            row["pred_"+key] = pred_feat[p][key] if p is not None else np.nan
        row["length_ratio"] = pred_feat[p]["skeleton_length"] / glen if p is not None and glen > 0 else np.nan
        row["length_abs_relative_error"] = abs(row["length_ratio"]-1) if np.isfinite(row["length_ratio"]) else np.nan
        row["width_ratio"] = pred_feat[p]["width_area_length"] / row["gt_width"] if p is not None and row["gt_width"] > 0 else np.nan
        row["elongation_ratio"] = pred_feat[p]["elongation"] / gt_feat[g]["elongation"] if p is not None else np.nan
        gt_rows.append(row)
    for p, feat in enumerate(pred_feat):
        pred_rows.append(dict(**base, model=model, record=rec["id"], prediction=str(pred_ids[p]),
                              matched=int(p in matched_pred), official_matched=int((iou[:,p]>.5).any()),
                              merge=int(merges[p]), overlap_degree=int(edges[:,p].sum()), **feat))
    for g,p in zip(*np.where(iou > .5)):
        val = float(iou[g,p])
        match_rows.append(dict(**base, model=model, record=rec["id"], annotation=rec["annotations"][g]["id"],
                               prediction=str(pred_ids[p]), iou=val, dice=2*val/(1+val),
                               one_to_one=int(g in assignment and assignment[g][0]==p)))
    return gt_rows, pred_rows, match_rows, [recrow]


def _evaluate_image(task):
    name, recs, scratch, fold = task
    all_masks = [record_masks(r) for r in recs]
    all_features = [[shape_features(m) for m in masks] for masks in all_masks]
    all_samples = [[spine_samples(a) for a in r["annotations"]] for r in recs]
    supports = [np.zeros(len(masks)) for masks in all_masks]
    aa = {}
    for a,b in permutations(range(len(recs)),2):
        aa[a,b] = overlap(all_masks[a],all_masks[b])
        supports[a] += (aa[a,b][1] > .5).any(1)
    supports = [s/(len(recs)-1) if len(recs)>1 else np.full(len(s),np.nan) for s in supports]
    base = dict(file_name=name,date=name[:8],site=Path(name).stem[14:],n_annotators=len(recs),fold=int(fold))
    output = [[],[],[],[]]
    z = np.load(Path(scratch)/"masks"/f"{Path(name).stem}.npz")
    for model in MODELS:
        lab = z[model]
        pm = [lab==p for p in range(1,int(lab.max())+1)]
        pf = [shape_features(m) for m in pm]
        for a,rec in enumerate(recs):
            if not rec["annotations"]:
                continue
            inter,iou = label_overlap(all_masks[a],lab)
            rows = _comparison(base,model,rec,all_masks[a],pm,all_features[a],pf,all_samples[a],supports[a],inter,iou,list(range(1,len(pm)+1)))
            for out,row in zip(output,rows):out.extend(row)
    for a,b in permutations(range(len(recs)),2):
        if not recs[a]["annotations"]:
            continue
        inter,iou = aa[a,b]
        iabase = dict(base,reference_prediction_record=recs[b]["id"])
        rows = _comparison(iabase,"inter_annotator",recs[a],all_masks[a],all_masks[b],all_features[a],all_features[b],all_samples[a],supports[a],inter,iou,[x["id"] for x in recs[b]["annotations"]])
        for out,row in zip(output,rows):out.extend(row)
    return output


def evaluate(dataset, fusion_root, scratch, results, workers=4, limit=None):
    start=time.monotonic()
    results = Path(results)
    # Enforce the PQ gate before any morphology is calculated.
    provenance = json.loads((results/"provenance.json").read_text())
    verification = json.loads((results/"pq_verification.json").read_text())
    assert all(v["max_per_image_error"] < 1e-8 for v in verification.values())
    if not limit:
        assert provenance["images"] == provenance["full_images"]
    _,records = annotation_index(dataset)
    foldmap = pd.read_csv(Path(fusion_root)/"outputs/sfsc_unet_v3/folds_v3.csv").set_index("file_name").fold.to_dict()
    tasks = [(name,records[name],str(scratch),foldmap[name]) for name in sorted(records)[:limit]] if limit else [(name,records[name],str(scratch),foldmap[name]) for name in sorted(records)]
    all_rows = [[],[],[],[]]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,rows in enumerate(pool.map(_evaluate_image,tasks)):
            for out,row in zip(all_rows,rows):out.extend(row)
            if (i+1)%50 == 0 or i+1 == len(tasks):
                print(f"Morphology evaluated: {i+1}/{len(tasks)}",flush=True)
    tables = {}
    for key,rows in zip(("gt_instances","pred_instances","matches","records"),all_rows):
        df = pd.DataFrame(rows)
        df.to_csv(results/f"{key}.csv.gz",index=False,float_format="%.8g",compression={"method":"gzip","mtime":0})
        tables[key]=df
    write_json(results/"evaluation_run.json",dict(code_sha256=sha256(__file__),
        mask_manifest_sha256=sha256(results/"mask_manifest.csv") if (results/"mask_manifest.csv").exists() else None,
        elapsed_seconds=time.monotonic()-start,workers=workers,images=len(tasks),gpu_hours=0,colab_cu=0))
    return tables


def load_tables(results):
    return {k:pd.read_csv(Path(results)/f"{k}.csv.gz",dtype={"date":str}) for k in ("gt_instances","pred_instances","matches","records")}


def record_mask_manifest(scratch, results):
    """Fingerprint the small compressed label maps for safe notebook cache reuse."""
    provenance=json.loads((Path(results)/"provenance.json").read_text())
    expected=pd.read_csv(Path(results)/"pq_records.csv").file_name.unique()
    rows=[]
    for name in sorted(expected):
        path=Path(scratch)/"masks"/f"{Path(name).stem}.npz"
        rows.append(dict(file_name=name,sha256=sha256(path),bytes=path.stat().st_size))
    assert len(rows)==provenance["images"]
    pd.DataFrame(rows).to_csv(Path(results)/"mask_manifest.csv",index=False)


def cached_masks_valid(dataset,fusion_run,scratch,results):
    results=Path(results)
    try:
        provenance=json.loads((results/"provenance.json").read_text())
        ann,_=annotation_index(dataset)
        assert provenance["images"]==provenance["full_images"]
        assert sha256(ann)==provenance["annotation_sha256"]
        assert sha256(Path(fusion_run)/"image_stats.npz")==provenance["image_stats_sha256"]
        assert sha256(Path(fusion_run)/"selection_v5.json")==provenance["selection_sha256"]
        assert sha256(Path(__file__).with_name("frozen_geometry.py"))==provenance["geometry_module_sha256"]
        manifest=pd.read_csv(results/"mask_manifest.csv")
        assert len(manifest)==provenance["images"]
        for row in manifest.itertuples():
            assert sha256(Path(scratch)/"masks"/f"{Path(row.file_name).stem}.npz")==row.sha256
        return True
    except (OSError,ValueError,KeyError,AssertionError):
        return False


def bootstrap_weights(dates, n=2000):
    return np.random.default_rng(SEED).multinomial(len(dates),np.ones(len(dates))/len(dates),size=n).astype(float)


def _ratio_rows(df, numerator, denominator, dates, weights):
    d = pd.DataFrame({"date":df.date.astype(str),"num":np.asarray(numerator,float),"den":np.asarray(denominator,float)})
    a = d.groupby("date")[["num","den"]].sum().reindex(dates,fill_value=0).to_numpy()
    total = a.sum(0)
    if not total[1]:return None
    samples = weights@a
    vals = np.divide(samples[:,0],samples[:,1],out=np.full(len(weights),np.nan),where=samples[:,1]>0)
    ci = np.nanpercentile(vals,[2.5,97.5])
    return dict(estimate=float(total[0]/total[1]),ci_low=float(ci[0]),ci_high=float(ci[1]),
                numerator=float(total[0]),denominator=float(total[1]),n_dates=int((a[:,1]>0).sum())),vals


def aggregate(tables,results,n_bootstrap=2000):
    gt,pr,ma,recs=[tables[k] for k in ("gt_instances","pred_instances","matches","records")]
    dates=sorted(recs.date.astype(str).unique())
    weights=bootstrap_weights(dates,n_bootstrap)
    rows=[];boot={}
    def add(df,num,den,metric,model,stratum="all",level="all",unit="record"):
        got=_ratio_rows(df,num,den,dates,weights)
        if got is None:return
        stats,draws=got
        rows.append(dict(model=model,stratum=stratum,level=str(level),metric=metric,unit=unit,**stats))
        if stratum=="all":boot[model,metric,unit]=draws
    # Official pooled metrics and genuine image-level stratification (FP included).
    for model,r in recs.groupby("model"):
        groups=[("all","all",r)]
        groups.extend(("image_population",label,sub) for label,sub in
                      (("multi-record",r[r.n_annotators>1]),("single-record",r[r.n_annotators==1])) if len(sub))
        for col in ("site","n_annotators","fold"):
            groups.extend((col,level,sub) for level,sub in r.groupby(col))
        for col,level,sub in groups:
            den=sub.tp+.5*(sub.fp+sub.fn)
            for metric,num,d in (("pq",sub.sum_iou,den),("sq",sub.sum_iou,sub.tp),("rq",sub.tp,den),
                                 ("precision",sub.tp,sub.tp+sub.fp),("recall",sub.tp,sub.tp+sub.fn),
                                 ("split_gt_rate",sub.split_gt,sub.n_gt),("merge_pred_rate",sub.merge_pred,sub.n_pred)):
                add(sub,num,d,metric,model,col,level)
        # Each image equally weighted after pooling its annotation records.
        image=r.groupby(["file_name","date"],as_index=False)[["sum_iou","tp","fp","fn"]].sum()
        den=image.tp+.5*(image.fp+image.fn)
        add(image,image.sum_iou/den,np.ones(len(image)),"pq_image_macro",model,unit="image")
    for model,g in gt.groupby("model"):
        groups=[("all","all",g)]
        for col in ("gt_area_bucket","gt_width_bucket","radial_bucket","site","n_annotators","support_bucket"):
            groups.extend((col,level,sub) for level,sub in g.groupby(col))
        for col,level,sub in groups:
            add(sub,sub.official_matched,np.ones(len(sub)),"gt_match_rate",model,col,level,"GT instance")
            for metric in ("coverage_1","coverage_3","coverage_5","gap_1","gap_3","gap_5", "length_ratio","length_abs_relative_error","width_ratio","elongation_ratio"):
                valid=sub[sub[metric].notna()]
                if len(valid):add(valid,valid[metric],np.ones(len(valid)),metric,model,col,level,"GT instance")
            valid=sub[sub.matched.astype(bool)]
            for metric in ("iou","dice"):
                if len(valid):add(valid,valid[metric],np.ones(len(valid)),"matched_"+metric,model,col,level,"GT instance")
        # Annotation-object macro -> annotator-record macro -> image macro.
        for metric in ("coverage_1","coverage_3","coverage_5","gap_3"):
            r=g.groupby(["file_name","date","record"],as_index=False)[metric].mean()
            image=r.groupby(["file_name","date"],as_index=False)[metric].mean().dropna()
            add(image,image[metric],np.ones(len(image)),metric+"_image_macro",model,unit="image")
        for metric in ("gt_width","gt_elongation","spine_length","pred_skeleton_length","pred_width_area_length","pred_width_edt","pred_elongation"):
            valid=g[g[metric].notna()]
            if len(valid):add(valid,valid[metric],np.ones(len(valid)),metric,model,unit="GT instance")
    for model,p in pr.groupby("model"):
        for metric in ("skeleton_length","width_area_length","width_edt","elongation"):
            valid=p[p[metric].notna()]
            if len(valid):add(valid,valid[metric],np.ones(len(valid)),"all_pred_"+metric,model,unit="prediction per record")
    deltas=[]
    for (model,metric,unit),a in boot.items():
        key=("v3_rescored",metric,unit)
        if model=="v4_evidence" and key in boot:
            v3=next(r for r in rows if r["model"]=="v3_rescored" and r["metric"]==metric and r["unit"]==unit and r["stratum"]=="all")
            v4=next(r for r in rows if r["model"]==model and r["metric"]==metric and r["unit"]==unit and r["stratum"]=="all")
            lo,hi=np.nanpercentile(a-boot[key],[2.5,97.5])
            deltas.append(dict(metric=metric,unit=unit,estimate=v4["estimate"]-v3["estimate"],ci_low=float(lo),ci_high=float(hi)))
    pd.DataFrame(rows).to_csv(Path(results)/"metrics_ci.csv",index=False,float_format="%.9g")
    pd.DataFrame(deltas).to_csv(Path(results)/"paired_deltas_ci.csv",index=False,float_format="%.9g")
    distribution=[]
    for model,g in ma.groupby("model"):
        for metric in ("iou","dice"):
            qs=g[metric].quantile([0,.1,.25,.5,.75,.9,1])
            distribution.append(dict(model=model,metric=metric,n=len(g),mean=float(g[metric].mean()),
                                     **{f"q{int(q*100):02d}":float(v) for q,v in qs.items()}))
    pd.DataFrame(distribution).to_csv(Path(results)/"matched_distributions.csv",index=False)
    totals=recs.groupby("model")[["sum_iou","tp","fp","fn","split_gt","merge_pred","n_gt","n_pred","ambiguous_threshold_pairs"]].sum()
    totals.to_csv(Path(results)/"pooled_counts.csv",float_format="%.10g")
    audit={}
    for model,g in gt.groupby("model"):
        p=pr[pr.model==model]
        r=recs[recs.model==model]
        audit[model]=dict(images=int(g.file_name.nunique()),records_or_directed_comparisons=len(r),
            gt_objects=len(g),pred_objects_record_weighted=len(p),missing_spines=int(g.spine_length.isna().sum()),
            unique_pred_masks=int(p[["file_name","prediction"]].drop_duplicates().shape[0]),
            gt_unmatched_one_to_one=int((g.matched==0).sum()),pred_unmatched_one_to_one=int((p.matched==0).sum()),
            gt_unmatched_official=int((g.official_matched==0).sum()),pred_unmatched_official=int((p.official_matched==0).sum()),
            valid_matched_length_ratios=int(g.length_ratio.notna().sum()),
            zero_length_pred_skeletons=int((p.skeleton_length==0).sum()),
            gt_shape_mean={k:float(g[k].mean()) for k in ("gt_area","spine_length","gt_width","gt_elongation")},
            pred_shape_mean={k:float(p[k].mean()) for k in ("area","skeleton_length","width_area_length","width_edt","elongation")})
    write_json(Path(results)/"morphology_audit.json",audit)
    write_json(Path(results)/"bootstrap.json",dict(seed=SEED,replicates=n_bootstrap,dates=len(dates),
               cluster="UTC calendar date from filename; all sites, records, objects and directed human comparisons travel together",
               interval="percentile 95%; frozen models, thresholds and examples; no refitting"))
    return pd.DataFrame(rows),pd.DataFrame(deltas)
