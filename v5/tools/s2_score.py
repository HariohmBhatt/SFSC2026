"""V5 step 2 scoring (CPU): new fold-0 OOF maps vs V3 fold-0 OOF maps, same images, same pipeline.

Candidates: V3/V4 geometry o1 (t_low 0.40, contact split 0.5, -1 px), prefilter max_p >= 0.40 & area >= 16
(the red team's d_build worker, which reproduces the shipped V3 candidates). Keep-scorer: V3/V4's HGB on the
12 component features, label = mean over records of (best IoU if > 0.5 else 0). Pooled PQ, many-to-many rule.

Two scorer protocols, both paired (same images, same splits):
  inner : scorer cross-fitted inside fold 0 (4 date-grouped splits); each arm learns from its own candidates,
          which is what a full 4-fold retrain would do.
  outer : scorer trained on folds 1-3 V3 candidates, applied to fold 0 (penalises a calibration shift).
Headline = inner at V4's tau 0.225; paired bootstrap over capture dates gives the interval.

Usage: python3 s2_score.py NEW_OOF_DIR [BASE_OOF_DIR]
"""
import gzip, json, os, pickle, sys, time
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

R2 = Path("/home/hariohm/sfsc_research/r2")
sys.path.insert(0, str(R2 / "D_scripts"))
sys.path.insert(0, str(R2 / "B_scripts"))
import d_build as DB          # noqa: E402
import s4_scorer as S4        # noqa: E402

DB.CONFIGS = [("o1", 0.40, 1.0)]
TAU = 0.225
TAUS = np.round(np.arange(0.10, 0.401, 0.025), 3)
SEED = 20261011


def build(oof_dir, names, gt, workers):
    npz = {p.stem: str(p) for p in Path(oof_dir).glob("*.npz")}
    missing = [n for n in names if Path(n).stem not in npz]
    assert not missing, f"{len(missing)} fold-0 maps missing in {oof_dir}, e.g. {missing[:3]}"
    items = [(n, npz[Path(n).stem]) for n in names]
    with mp.get_context("fork").Pool(workers, initializer=DB._init, initargs=(gt,)) as pool:
        res = dict(pool.imap_unordered(DB.work, items, chunksize=1))
    bad = [k for k, v in res.items() if "error" in v]
    assert not bad, res[bad[0]]["error"]
    return {k: v["o1"] for k, v in res.items()}


def labels(c):
    if not c["iou"] or not len(c["area"]):
        return np.zeros(len(c["area"]))
    B = np.stack([m.max(0) if m.shape[0] else np.zeros(len(c["area"])) for m in c["iou"]])
    return np.where(B > 0.5, B, 0.0).mean(0)


def table(cands, names):
    X, y, own = [], [], []
    for i, n in enumerate(names):
        c = cands[n]
        X.append(c["feats"]); y.append(labels(c)); own.append(np.full(len(c["area"]), i))
    return np.concatenate(X), np.concatenate(y), np.concatenate(own)


def per_image_stats(cands, names, u, own, tau):
    st = np.zeros((len(names), 4))
    for i, n in enumerate(names):
        k = u[own == i] > tau
        for m in cands[n]["iou"]:
            st[i] += S4.m2m_stats(m[:, k])
    return st


def pq(st):
    den = st[1] + 0.5 * (st[2] + st[3])
    return float(st[0] / den) if den else 0.0


def reach(cands, names):
    hit = tot = 0
    for n in names:
        for m in cands[n]["iou"]:
            tot += m.shape[0]
            hit += int((m > 0.5).any(1).sum()) if m.size else 0
    return hit / max(tot, 1)


def main():
    new_dir = Path(sys.argv[1])
    base_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else next((R2 / "B_data" / "oof").glob("fold0_*"))
    workers = int(os.environ.get("S2_WORKERS", str(max(2, os.cpu_count() - 1))))
    folds = pd.read_csv(R2 / "B_data" / "folds_v3.csv")
    names = sorted(folds.loc[folds.fold == 0, "file_name"])
    raw = json.loads(DB.ANN.read_text())
    img_by_id = {im["id"]: im["file_name"] for im in raw["images"]}
    anns = defaultdict(list)
    for a in raw["annotations"]:
        anns[a["image_id"]].append(a)
    gt = defaultdict(list)
    for rid, al in anns.items():
        gt[img_by_id[rid]].append({"id": rid, "ann": al, "area": np.asarray([a["area"] for a in al], np.float64)})
    gt = dict(gt)

    t0 = time.time()
    arms = {"v3": build(base_dir, names, gt, workers), "new": build(new_dir, names, gt, workers)}
    print(f"candidates built in {time.time() - t0:.0f}s | base {base_dir.name} | new {new_dir}", flush=True)

    dates = np.array([Path(n).stem[:8] for n in names])
    splits = list(GroupKFold(4).split(names, groups=dates))
    with gzip.open(R2 / "D_results" / "d_candidates.pkl.gz", "rb") as f:
        dc = pickle.load(f)
    other = sorted(folds.loc[folds.fold != 0, "file_name"])
    Xo, yo, _ = table({n: dc[n]["o1"] for n in other}, other)
    outer = S4.make_hgb().fit(Xo, yo)

    res, img_stats = {}, {}
    for arm, cands in arms.items():
        X, y, own = table(cands, names)
        u_in = np.zeros(len(X))
        for tr_i, te_i in splits:
            tr, te = np.isin(own, tr_i), np.isin(own, te_i)
            u_in[te] = S4.make_hgb().fit(X[tr], y[tr]).predict(X[te])
        u_out = outer.predict(X)
        sweep = {t: pq(per_image_stats(cands, names, u_in, own, t).sum(0)) for t in TAUS}
        st = per_image_stats(cands, names, u_in, own, TAU)
        img_stats[arm] = st
        tot = st.sum(0)
        res[arm] = {"pq_inner": pq(tot), "sq": tot[0] / max(tot[1], 1), "rq": tot[1] / (tot[1] + 0.5 * (tot[2] + tot[3])),
                    "tp_fp_fn": tot[1:].astype(int).tolist(),
                    "pq_inner_best_tau": max(sweep.values()), "best_tau": max(sweep, key=sweep.get),
                    "pq_outer": pq(per_image_stats(cands, names, u_out, own, TAU).sum(0)),
                    "candidates": int(len(X)), "reach_iou50": reach(cands, names),
                    "split_pq": [pq(st[te].sum(0)) for _, te in splits]}

    d = img_stats["new"] - img_stats["v3"]
    rng = np.random.default_rng(SEED)
    ud = np.unique(dates)
    idx_of = {k: np.flatnonzero(dates == k) for k in ud}
    boots = []
    for _ in range(2000):
        pick = np.concatenate([idx_of[k] for k in rng.choice(ud, len(ud))])
        boots.append(pq(img_stats["new"][pick].sum(0)) - pq(img_stats["v3"][pick].sum(0)))
    delta = res["new"]["pq_inner"] - res["v3"]["pq_inner"]
    wins = int(sum(a > b for a, b in zip(res["new"]["split_pq"], res["v3"]["split_pq"])))
    out = {"arms": res, "delta_inner": delta, "delta_outer": res["new"]["pq_outer"] - res["v3"]["pq_outer"],
           "delta_ci90": [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))],
           "split_wins": f"{wins}/4", "gate": bool(delta >= 0.003 and wins >= 3)}
    print(json.dumps(out, indent=1, default=float))
    (new_dir.parent / "s2_score.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
