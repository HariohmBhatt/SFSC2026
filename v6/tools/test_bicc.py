"""CPU unit tests for the V6 BiCC term (run: python test_bicc.py [path/to/reference_losses.py]).

Loads tools/bicc_src.py exactly as it is inlined into SFSC2026_V6_BiCC_Pilot_Colab.ipynb (the builder asserts the
notebook cells contain this text verbatim), then checks values, gradient signs, soft-target handling, batch
independence, agreement with the independent reference implementation on binary targets, and CPU timing.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
NS = {"np": np, "torch": torch, "ndi": ndi}
exec(compile((HERE / "bicc_src.py").read_text(), "bicc_src.py", "exec"), NS)
voronoi_ids_u8, bicc_term = NS["voronoi_ids_u8"], NS["bicc_term"]
torch.manual_seed(0)
RESULTS = {}


def batch(soft_np, logit_np):
    soft_u8 = np.clip(np.rint(np.asarray(soft_np) * 255), 0, 255).astype(np.uint8)
    vor = np.stack([voronoi_ids_u8(s) for s in soft_u8])[:, None]
    soft = torch.from_numpy(soft_u8[:, None].astype(np.float32) / 255.0)
    logit = torch.tensor(np.asarray(logit_np, np.float32)[:, None], requires_grad=True)
    return logit, soft, torch.from_numpy(vor)


def run(soft_np, logit_np, alpha=0.5):
    logit, soft, vor = batch(soft_np, logit_np)
    loss, st = bicc_term(logit, soft, vor, alpha)
    loss.backward()
    g = logit.grad[:, 0].numpy()
    assert torch.isfinite(loss) and np.isfinite(g).all(), "non-finite loss / gradient"
    return float(loss), g, {k: float(v) for k, v in st.items()}


def check(name, cond, info=""):
    RESULTS[name] = bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  {info}" if info else ""))
    assert cond, name


S = 64
# 1. Voronoi partition
m = np.zeros((S, S)); m[10:14, 5:20] = 1.0; m[50:54, 40:60] = 1.0; m[30:34, 30:34] = 0.34   # 2 consensus + 1 weak
v = voronoi_ids_u8(np.rint(m * 255).astype(np.uint8))
check("voronoi: 2 cells from consensus components only (weak one excluded)", set(np.unique(v)) == {1, 2})
check("voronoi: component pixels carry their own id", v[12, 10] != v[52, 50] and (v > 0).all())
check("voronoi: empty crop -> all zero", voronoi_ids_u8(np.zeros((S, S), np.uint8)).max() == 0)
check("voronoi: nearest-component assignment", v[0, 0] == v[12, 10] and v[63, 63] == v[52, 50])

# 2. missed GT component -> gradient pushes logits UP inside it (negative dL/dlogit), loss near 1
soft = np.zeros((1, S, S)); soft[0, 20:26, 10:50] = 1.0
loss, g, st = run(soft, np.full((1, S, S), -3.0))
check("missed GT: loss ~ 1 (1 - Dice of an empty prediction)", 0.9 < loss <= 1.0, f"loss={loss:.4f}")
check("missed GT: dL/dlogit < 0 on every GT pixel", (g[0][soft[0] > 0] < 0).all())
check("missed GT: no FP components", st["bicc_fp"] == 0 and st["bicc_k"] == 1, str(st))

# 3. perfect prediction -> loss ~ 0
loss_p, _, _ = run(soft, np.where(soft > 0, 8.0, -8.0))
check("perfect prediction: loss ~ 0", loss_p < 0.01, f"loss={loss_p:.5f}")

# 4. false-positive blob far from any annotation -> counted, pushed DOWN
lg = np.where(soft > 0, 8.0, -8.0); lg[0, 45:50, 45:55] = 2.0
loss_fp, g, st = run(soft, lg)
p_blob = 1 / (1 + np.exp(-2.0))
check("FP blob: counted once", st["bicc_fp"] == 1, str(st))
pg, pn = 1 / (1 + np.exp(-8.0)), 1 / (1 + np.exp(8.0))      # the blob also sits in the GT's Voronoi cell
I, Y = 240 * pg, 240.0
P = 240 * pg + 50 * p_blob + (S * S - 290) * pn
e_k = 1 - (2 * I + 1) / (P + Y + 1)
exp = 2 * (0.5 * e_k + 0.5 * p_blob) / 2                    # one GT cell + one FP with mean p = sigmoid(2)
check("FP blob: loss = 2(a*sum e + (1-a)*sum f)/(K+J)", abs(loss_fp - exp) < 0.01, f"{loss_fp:.4f} vs {exp:.4f}")
check("FP blob: dL/dlogit > 0 on blob pixels", (g[0, 45:50, 45:55] > 0).all())

# 5. soft-target handling: a blob on a single-annotator filament (soft = 1/3) is NOT a false positive
soft2 = soft.copy(); soft2[0, 45:50, 45:55] = 1 / 3
_, g2, st2 = run(soft2, lg)
check("soft target: blob over sub-consensus support is not an FP", st2["bicc_fp"] == 0, str(st2))
check("soft target: sub-consensus filament is not a GT cell either", st2["bicc_k"] == 1, str(st2))
# blob touching support with a single pixel is also exempt
soft3 = soft.copy(); soft3[0, 47, 44] = 0.2
lg3 = lg.copy(); lg3[0, 47, 44] = 2.0
_, _, st3 = run(soft3, lg3)
check("soft target: blob touching one supported pixel is not an FP", st3["bicc_fp"] == 0, str(st3))
# the Voronoi Dice uses the soft target: inside the GT cell a weak pixel prefers p ~ its soft value
soft4 = np.zeros((1, S, S)); soft4[0, 20:26, 10:50] = 2 / 3
l_hi, _, _ = run(soft4, np.where(soft4 > 0, 8.0, -8.0))
l_mid, _, _ = run(soft4, np.where(soft4 > 0, np.log(2.0), -8.0))      # p = 2/3 exactly
check("soft target: annotation side is finite and in [0,1] for 2/3 consensus", 0 <= l_mid <= 1 and 0 <= l_hi <= 1,
      f"p=1: {l_hi:.4f}, p=2/3: {l_mid:.4f}")

# 6. two GT components, one detected one missed -> loss ~ 0.5, gradient only on the missed one is strong
soft5 = np.zeros((1, S, S)); soft5[0, 5:9, 5:40] = 1.0; soft5[0, 50:54, 20:60] = 1.0
lg5 = np.full((1, S, S), -8.0); lg5[0, 5:9, 5:40] = 8.0
l5, g5, st5 = run(soft5, lg5)
check("size-independent: one of two components missed -> loss ~ 0.5", abs(l5 - 0.5) < 0.02 and st5["bicc_k"] == 2,
      f"loss={l5:.4f}")

# 7. batch independence: sample 1 has the FP blob, sample 0 is perfect
sb = np.concatenate([soft, soft]); lb = np.concatenate([np.where(soft > 0, 8.0, -8.0), lg])
lb_, gb, stb = run(sb, lb)
check("batch: ids do not mix across samples (K=2, J=1)", stb["bicc_k"] == 2 and stb["bicc_fp"] == 1, str(stb))
check("batch: blob gradient only in its own sample", np.abs(gb[0, 45:50, 45:55]).max() < 1e-6 and (gb[1, 45:50, 45:55] > 0).all())

# 8. empty batch and random noise: finite, no NaN
le, ge, ste = run(np.zeros((2, S, S)), np.full((2, S, S), -5.0))
check("empty batch: loss 0, finite gradient", le == 0.0 and np.isfinite(ge).all())
rs = (np.random.default_rng(1).random((4, 128, 128)) > 0.7) * np.random.default_rng(2).choice([1 / 3, 2 / 3, 1.0], (4, 128, 128))
lr_, gr, str_ = run(rs, np.random.default_rng(3).normal(0, 3, (4, 128, 128)))
check("random soft targets + random logits: finite", np.isfinite(lr_), f"loss={lr_:.4f} {str_}")
lx, gx, _ = run(np.ones((1, S, S)), np.full((1, S, S), 30.0))
check("saturated logits (+30) on a full target: finite, ~0", np.isfinite(lx) and lx < 1e-3)

# 9. agreement with the independent reference implementation on BINARY targets
ref_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if ref_path and ref_path.is_file():
    RNS = {}
    exec(compile(ref_path.read_text(), "ref_losses.py", "exec"), RNS)
    rng = np.random.default_rng(7)
    for trial in range(5):
        y = np.zeros((3, 96, 96))
        for b in range(3):
            for _ in range(rng.integers(0, 4)):
                r, c = rng.integers(0, 90, 2); y[b, r:r + rng.integers(2, 6), c:c + rng.integers(3, 30)] = 1.0
        lg = rng.normal(-1, 2.5, (3, 96, 96))
        logit, soft_t, vor = batch(y, lg)
        ours, _ = bicc_term(logit, soft_t, vor, 0.5)
        vref = torch.from_numpy(np.stack([RNS["voronoi_cells"](yy > 0.5) for yy in y])[:, None])
        theirs = RNS["bicc_loss"](torch.tensor(lg[:, None], dtype=torch.float32), torch.from_numpy(y[:, None].astype(np.float32)), vref, 0.5)
        check(f"reference agreement (binary target, trial {trial})", abs(float(ours) - float(theirs)) < 1e-5,
              f"ours {float(ours):.6f} ref {float(theirs):.6f}")
else:
    print("SKIP reference agreement (no reference file given)")

# 10. CPU timing at training size (8 x 768^2): labelling + region sums; Voronoi per crop
B, H = 8, 768
yy = np.zeros((B, H, H))
for b in range(B):
    for _ in range(12):
        r, c = np.random.randint(0, H - 40, 2); yy[b, r:r + 6, c:c + 120] = np.random.choice([1 / 3, 2 / 3, 1.0])
t0 = time.perf_counter()
vv = [voronoi_ids_u8(np.rint(s * 255).astype(np.uint8)) for s in yy]
t_vor = (time.perf_counter() - t0) / B
logit, soft_t, vor = batch(yy, np.random.normal(-2, 2, (B, H, H)))
t0 = time.perf_counter()
for _ in range(3):
    l, _ = bicc_term(logit, soft_t, vor)
    l.backward()
t_loss = (time.perf_counter() - t0) / 3
print(f"timing: voronoi {t_vor * 1e3:.1f} ms/crop (DataLoader worker) | bicc fwd+bwd {t_loss * 1e3:.0f} ms / 8x768^2 on CPU")
RESULTS["timing_ms"] = {"voronoi_per_crop": round(t_vor * 1e3, 1), "bicc_fwd_bwd_8x768_cpu": round(t_loss * 1e3, 1)}
(HERE / "test_bicc_results.json").write_text(json.dumps(RESULTS, indent=1))
print("ALL TESTS PASSED")
