"""Step 3 verdict: paired per-image comparison of two fusion runs' image_stats.npz.

usage: python s3_compare.py NEW_RUN_DIR OLD_RUN_DIR
Each variant in NEW is compared with the same variant in OLD, plus NEW's best variant against OLD v4_evidence
(the variant V4 shipped). Gate: gain >= +0.003 and wins >= 3/4 folds.
"""
import sys
from pathlib import Path

import numpy as np


def load(run):
    z = np.load(Path(run) / "image_stats.npz", allow_pickle=True)
    stats = {k.split("__tau")[0]: (float(k.split("__tau")[1]), z[k]) for k in z.files if "__tau" in k}
    return z["names"], z["folds"], stats


def pq(st):
    den = st[1] + 0.5 * (st[2] + st[3])
    return float(st[0] / den) if den else 0.0


def boot(a, b, n=2000, seed=20261011):
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(a), np.full(len(a), 1.0 / len(a)), size=n).astype(np.float64)
    sa, sb = w @ a[:, :4], w @ b[:, :4]
    f = lambda s: s[:, 0] / np.maximum(s[:, 1] + 0.5 * (s[:, 2] + s[:, 3]), 1e-9)
    return np.percentile(f(sa) - f(sb), [2.5, 97.5])


def compare(label, a, b, folds):
    gs = sorted(np.unique(folds))
    fa = [pq(a[folds == g].sum(0)) for g in gs]
    fb = [pq(b[folds == g].sum(0)) for g in gs]
    gain = pq(a.sum(0)) - pq(b.sum(0))
    lo, hi = boot(a, b)
    wins = sum(x > y for x, y in zip(fa, fb))
    verdict = "PASS" if gain >= 0.003 and wins >= 3 else "fail"
    print(f"{label:34s} {pq(a.sum(0)):.4f} vs {pq(b.sum(0)):.4f}  gain {gain:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
          f"wins {wins}/{len(gs)}  per-fold {' '.join(f'{x - y:+.4f}' for x, y in zip(fa, fb))}  {verdict}")


if __name__ == "__main__":
    n_new, f_new, new = load(sys.argv[1])
    n_old, f_old, old = load(sys.argv[2])
    assert (n_new == n_old).all() and (f_new == f_old).all(), "runs scored different images or folds"
    for v in sorted(set(new) & set(old)):
        compare(f"{v} (tau {new[v][0]:g} vs {old[v][0]:g})", new[v][1], old[v][1], f_new)
    best = max(("v4_evidence", "v4_full"), key=lambda v: pq(new[v][1].sum(0)))
    compare(f"NEW {best} vs OLD v4_evidence", new[best][1], old["v4_evidence"][1], f_new)
