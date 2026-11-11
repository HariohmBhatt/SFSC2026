"""Builds v6/SFSC2026_V6_BiCC_Pilot_Colab.ipynb: fold-0 pilot of V3 + a 2-D BiCC component term (weight 0.3).

Source = the V3 notebook (v3/SFSC2026_V3_UNet_EffV2_4fold_L4_Colab.ipynb), patched as one controlled change:
  * loss: V3's 0.5 BCE(soft) + 1.0 Lovasz + 0.2 edge + 0.5 contact  PLUS  0.3 * BiCC (tools/bicc_src.py, verbatim)
  * target tensor gets a third uint8 channel: Voronoi cell ids of the consensus (soft >= 0.5) components, computed in
    the DataLoader workers on the final augmented crop (no RNG is consumed, so the crop stream equals V3's)
  * folds_v3.csv read only, only fold 0 trained, fold-0 OOF maps written (V3 format), no test export
  * a timing cell (debug run) measures the per-step overhead of the term on real crops; the full run aborts if the
    first epoch is > 1.5 x V3's 129.8 s on L4 (the agreed kill rule)
Everything else -- encoder, crops, sampling, augmentation, EMA, schedule, seed, calibrated-PQ checkpoint pick, D4 OOF
inference -- is V3, unchanged. Pattern: /tmp/v5build/build_s2.py (V5 step-2 pilot).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3_NB = Path("/home/hariohm/SFSC2026/v3/SFSC2026_V3_UNet_EffV2_4fold_L4_Colab.ipynb")
OUT = HERE.parent / "SFSC2026_V6_BiCC_Pilot_Colab.ipynb"
src = ["".join(c["source"]) for c in json.loads(V3_NB.read_text())["cells"]]
BICC = (HERE / "bicc_src.py").read_text()
BICC_DATA = BICC.split("# ==== SECTION: loss ====\n")[0].split("# ==== SECTION: dataset ====\n")[1]
BICC_LOSS = BICC.split("# ==== SECTION: loss ====\n")[1]
V3_EPOCH_S = 129.8          # V3-recipe epoch time on L4 (V5 step-2 log, 250 steps x micro 8)


def rep(text, old, new, count=1):
    n = text.count(old)
    assert n == count, f"expected {count} x {old[:70]!r}, found {n}"
    return text.replace(old, new)


cells = []
def md(s): cells.append(("markdown", s.strip() + "\n"))
def code(s): cells.append(("code", s.strip("\n") + "\n"))

md(r"""
# SFSC 2026 — V6: BiCC component-loss pilot, fold 0 (Colab / NVIDIA L4)

One controlled change to the V3 UNet (EfficientNetV2-S) recipe, trained on folds 1-3 and judged on fold 0:

| | V3 | this pilot |
|---|---|---|
| mask loss | 0.5 BCE(soft) + 1.0 Lovász hinge + 0.2 edge BCE + 0.5 contact BCE | same **+ 0.3 · BiCC** |
| targets | soft filament (mean of annotators), soft contact | same + Voronoi cell ids of consensus components |
| everything else | encoder, crops, augmentation, EMA 0.999, 40 x 250 steps, seed, calibrated-PQ checkpoint pick, D4 OOF | unchanged |

**BiCC term** (idea of arXiv 2609.30223; follows the independent SFSC implementation in
`theainerd/filament_segmentation`, adapted to soft multi-annotator targets):
* annotation side: the crop is split into Voronoi cells around the *consensus* components (soft target ≥ 0.5,
  the same mask V3's Lovász term uses); each cell contributes `1 − softDice(p, soft)` inside the cell;
* prediction side: each 8-connected component of `p > 0.5` that touches **no pixel with any annotator support**
  (soft target > 0) contributes its mean probability. Sub-consensus filaments are never hard negatives here;
* `BiCC = 2·(α Σ e_k + (1−α) Σ f_j)/(K + J)`, α = 0.5; weight 0.3 fixed before any validation.

Folds are V3's `folds_v3.csv` (read only): fold 0 is the same 177 JPEGs V3's fold-0 model never saw.
Output: `MyDrive/SFSC2026/outputs/sfsc_v6/bicc_f0/` (`fold0/`, `oof/fold0_*/*.npz` with `fg`, `ct` uint8 maps like V3).
Scoring against V3 fold 0 (same images, same candidate pipeline, cross-fitted scorer) runs off-GPU (`tools/v6_score.py`).

**Run:** Runtime -> L4 GPU -> Run all. Resumable (fold 0 resumes from `last.pt`). Environment switches (all optional):
`SFSC_DEBUG=1` (2 x 20 steps, 12 eval images, + overhead timing), `SFSC_DRIVE_MODE=mount|rclone|local`.
L4 estimate: V3 ~130 s/epoch -> ~90 min training + ~10 min evaluation + ~10 min OOF; the run aborts if epoch 1
takes > 1.5 x 130 s (BiCC overhead > 50 %).
""")

# ---- Cell 01: Drive access (mount | rclone | local), inputs read only
md("## Cell 01 — Drive access, inputs (read only), output folder")
code(r'''
from pathlib import Path
import os, shutil, subprocess, threading, time

IN_COLAB = bool(os.environ.get("COLAB_RELEASE_TAG"))
DRIVE_MODE = os.environ.get("SFSC_DRIVE_MODE", "mount" if IN_COLAB else "local")   # mount | rclone | local
RCLONE_REMOTE = os.environ.get("SFSC_RCLONE_REMOTE", "gdrive")
DEBUG = os.environ.get("SFSC_DEBUG", "0") == "1"
TIMING = os.environ.get("SFSC_TIMING", "1" if DEBUG else "0") == "1"
OUT_NAME = "bicc_f0_debug" if DEBUG else "bicc_f0"
LOCAL_ROOT = Path(os.environ.get("SFSC_LOCAL_ROOT", "/content/sfsc_v6" if IN_COLAB else "/tmp/sfsc_v6"))
LOCAL_DATA = LOCAL_ROOT / "dataset"
LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

def rclone(*args, check=True):
    return subprocess.run(["rclone", *args, "--transfers", "4", "--checkers", "8", "--tpslimit", "5",
                           "--retries", "5"], check=check)

if DRIVE_MODE == "mount":
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_ROOT = Path("/content/drive/MyDrive/SFSC2026")
elif DRIVE_MODE == "rclone":
    if shutil.which("rclone") is None:
        subprocess.run("curl -fsSL https://rclone.org/install.sh | bash", shell=True, check=True)
    DRIVE_ROOT = LOCAL_ROOT / "drive_mirror"
    remote = f"{RCLONE_REMOTE}:SFSC2026"
    rclone("copy", f"{remote}/data", str(DRIVE_ROOT / "data"), "--include", "/sfsc2026-data.zip")
    rclone("copy", f"{remote}/outputs/sfsc_unet_v3", str(DRIVE_ROOT / "outputs" / "sfsc_unet_v3"),
           "--include", "/{folds_v3.csv,resolved_encoder.json}")
    rclone("copy", f"{remote}/outputs/sfsc_v6/{OUT_NAME}", str(DRIVE_ROOT / "outputs" / "sfsc_v6" / OUT_NAME),
           "--exclude", "*.part", "--exclude", "oof/**", check=False)       # resume state, if any
else:
    DRIVE_ROOT = Path(os.environ["SFSC_DRIVE_ROOT"])

DATA_ZIP = DRIVE_ROOT / "data" / "sfsc2026-data.zip"
DRIVE_EXTRACTED = DRIVE_ROOT / "data" / "SFSCDataset"
V1_SPLIT_CSV = DRIVE_ROOT / "outputs" / "sfsc_deeplabv3_clahe_v1" / "split.csv"   # unused: folds are reused
V3_FOLDS_CSV = DRIVE_ROOT / "outputs" / "sfsc_unet_v3" / "folds_v3.csv"
RUN_BASE = DRIVE_ROOT / "outputs" / "sfsc_v6" / "bicc_f0"
assert V3_FOLDS_CSV.is_file(), f"missing {V3_FOLDS_CSV}"

_SYNC_STOP = threading.Event()
def push():
    """rclone mode: mirror checkpoints/history to Drive for resume (OOF maps are fetched from the VM directly)."""
    if DRIVE_MODE == "rclone" and RUN_BASE.parent.joinpath(OUT_NAME).is_dir():
        rclone("copy", str(RUN_BASE.parent / OUT_NAME), f"{RCLONE_REMOTE}:SFSC2026/outputs/sfsc_v6/{OUT_NAME}",
               "--exclude", "*.part", "--exclude", "*.tmp", "--exclude", "oof/**", check=False)

def _sync_loop():
    while not _SYNC_STOP.wait(600):
        push()

_SYNC_THREAD = threading.Thread(target=_sync_loop, daemon=True)
if DRIVE_MODE == "rclone":
    _SYNC_THREAD.start()
print("Drive mode:", DRIVE_MODE, "| output:", RUN_BASE.parent / OUT_NAME, "| debug:", DEBUG, "| timing:", TIMING)
''')

# ---- Cell 02: dependencies (V3 as is)
md("## Cell 02 — Install missing dependencies (keeps Colab's CUDA PyTorch)")
code(src[4])

# ---- Cell 03: config
c6 = src[6]
c6 = rep(c6, "    debug_run: bool = False                  # True = quick pipeline check (separate *_debug folder)\n"
             "    local_dataset_override: str | None = None\n",
         "    debug_run: bool = DEBUG\n"
         "    local_dataset_override: str | None = os.environ.get(\"SFSC_DATASET\") or None\n")
c6 = rep(c6, "    folds_to_train: tuple = (0, 1, 2, 3)\n",
         "    folds_to_train: tuple = tuple(int(f) for f in os.environ.get(\"SFSC_FOLDS\", \"0\").split(\",\"))\n")
c6 = rep(c6, "    export_test: bool = True\n", "    export_test: bool = False\n")
c6 = rep(c6, "    cldice_iters: int = 6\n",
         "    cldice_iters: int = 6\n"
         "    bicc_weight: float = 0.3                       # BiCC component term (V3: none); fixed in advance\n"
         "    bicc_alpha: float = 0.5                        # annotation-side vs prediction-side balance (paper default)\n")
c6 = rep(c6, 'RUN = V3_RUN_BASE.with_name(V3_RUN_BASE.name + ("_debug" if CFG.debug_run else ""))',
         'RUN = RUN_BASE.with_name(OUT_NAME)')
c6 = rep(c6, 'raise RuntimeError("V3 needs a CUDA GPU. Runtime -> Change runtime type -> L4.")',
         'raise RuntimeError("The pilot needs a CUDA GPU. Runtime -> Change runtime type -> L4.")')
md("## Cell 03 — Imports, configuration (edit here), reproducibility and helpers")
code(c6)

# ---- Cell 04, 05: dataset staging, COCO parsing (V3 as is)
md("## Cell 04 — Stage the dataset on the local Colab SSD")
code(src[8])
md("## Cell 05 — COCO parsing, record index, integrity inventory")
code(src[10])

# ---- Cell 06: caches (V3, minus the test-image cache: no test export in the pilot)
c12 = rep(src[12], '''    list(tqdm(pool.map(lambda n: cache_image("test", n, TEST_PATHS[n]), sorted(TEST_PATHS)),
              total=len(TEST_PATHS), desc="Cache test images + flattened"))
''', "")
md("## Cell 06 — Disk geometry, limb-darkening flattening, local caches (V3; test images not cached)")
code(c12)

# ---- Cell 07: folds = V3's frozen folds, read only
c14 = rep(src[14], 'FOLDS_PATH = RUN / "folds_v3.csv"', 'FOLDS_PATH = V3_FOLDS_CSV                              # V3 folds, read only')
md("## Cell 07 — Folds: V3's frozen `folds_v3.csv` (date + hash grouped), read only")
code(c14)

# ---- Cell 08: PQ engine (V3 as is). V3 cell 09 (label oracles) is skipped: CPU-only diagnostics, already in V3.
md("## Cell 08 — Exact PQ engine and instance extraction (V3, unchanged)")
code(src[16])

md("## Cell 09 — Hard-negative centres (V3, unchanged)")
code(src[20])

# ---- Cell 10: crops + Voronoi channel
c22 = src[22]
c22 = rep(c22, "def normalize_np(img01, flat_ratio, radial):", BICC_DATA + "\ndef normalize_np(img01, flat_ratio, radial):")
c22 = rep(c22, "        y = np.stack([np.clip(np.rint(crop[3]), 0, 255), np.clip(np.rint(crop[4]), 0, 255)]).astype(np.uint8)\n",
          "        y = np.stack([np.clip(np.rint(crop[3]), 0, 255), np.clip(np.rint(crop[4]), 0, 255)]).astype(np.uint8)\n"
          "        y = np.concatenate([y, voronoi_ids_u8(y[0])[None]])           # BiCC: Voronoi ids (no RNG used)\n")
c22 = c22.rstrip() + '''
print("Voronoi cells in the preview crops:", [int(_preview[i][1][2].max()) for i in range(4)])
'''
md("## Cell 10 — Training crops (V3) + third target channel: Voronoi cells of the consensus components")
code(c22)

# ---- Cell 11: model (V3), encoder resolved by V3
c24 = rep(src[24], "ENCODER_PATH = RUN / \"resolved_encoder.json\"",
          "ENCODER_PATH = DRIVE_ROOT / \"outputs\" / \"sfsc_unet_v3\" / \"resolved_encoder.json\"   # same encoder as V3")
md("## Cell 11 — Model: V3's UNet / EfficientNetV2-S (same resolved encoder)")
code(c24)

# ---- Cell 12: loss
c26 = src[26]
c26 = rep(c26, "class LossV3(nn.Module):", BICC_LOSS + "\nclass LossV3(nn.Module):")
c26 = rep(c26, '''        else:
            cldice = x.new_zeros(())
''', '''        else:
            cldice = x.new_zeros(())
        if c.bicc_weight > 0:
            bicc, bst = bicc_term(fg_logit, soft, target_u8[:, 2:3], c.bicc_alpha)
        else:
            bicc, bst = x.new_zeros(()), {"bicc_k": x.new_zeros(()), "bicc_fp": x.new_zeros(())}
''')
c26 = rep(c26, '''                 + c.contact_weight * contact + c.cldice_weight * cldice)''',
          '''                 + c.contact_weight * contact + c.cldice_weight * cldice + c.bicc_weight * bicc)''')
c26 = rep(c26, '''"contact": contact.detach(), "cldice": cldice.detach()}''',
          '''"contact": contact.detach(), "cldice": cldice.detach(),
                 "bicc": bicc.detach(), "bicc_k": bst["bicc_k"], "bicc_fp": bst["bicc_fp"]}''')
c26 = rep(c26, 'LOSS_KEYS = ("loss", "bce", "lovasz", "edge", "contact", "cldice")',
          'LOSS_KEYS = ("loss", "bce", "lovasz", "edge", "contact", "cldice", "bicc", "bicc_k", "bicc_fp")')
c26 = rep(c26, '''print("Loss:", CFG.bce_weight, "BCE +", CFG.lovasz_weight, "Lovasz +", CFG.edge_weight, "edge +",
      CFG.contact_weight, "contact +", CFG.cldice_weight, "clDice")''',
          '''print("Loss:", CFG.bce_weight, "BCE +", CFG.lovasz_weight, "Lovasz +", CFG.edge_weight, "edge +",
      CFG.contact_weight, "contact +", CFG.cldice_weight, "clDice +", CFG.bicc_weight, f"BiCC (alpha {CFG.bicc_alpha})")''')
md("## Cell 12 — Loss: V3 (0.5·BCE + 1.0·Lovász + 0.2·edge + 0.5·contact) + 0.3·BiCC component term")
code(c26)

md("## Cell 13 — Full-resolution tiled inference with D4 TTA (V3, unchanged)")
code(src[28])

# ---- Cell 14: self-tests with the 3-channel target + BiCC checks on the device
c30 = src[30]
c30 = rep(c30, "tgt = torch.zeros((2, 2, 64, 64), dtype=torch.uint8, device=DEVICE); tgt[:, 0, 20:30, 10:50] = 255",
          "tgt = torch.zeros((2, 3, 64, 64), dtype=torch.uint8, device=DEVICE); tgt[:, 0, 20:30, 10:50] = 255\n"
          "    tgt[:, 2] = torch.from_numpy(voronoi_ids_u8(tgt[0, 0].cpu().numpy())).to(DEVICE)")
c30 = rep(c30, '''    assert torch.isfinite(val) and torch.isfinite(logits.grad).all()
''', '''    assert torch.isfinite(val) and torch.isfinite(logits.grad).all()
    assert float(parts["bicc_k"]) == 2 and float(parts["bicc"]) > 0.5, parts      # 2 missed GT cells
    assert (logits.grad[:, 0, 20:30, 10:50] < 0).all(), "BiCC/V3 loss must push logits up on missed GT"
    # BiCC false-positive side on the device: confident blob far from annotation -> pushed down; on support -> exempt
    lg = torch.full((1, 1, 64, 64), -8.0, device=DEVICE); lg[..., 20:30, 10:50] = 8.0; lg[..., 50:55, 50:60] = 2.0
    lg.requires_grad_(True)
    soft_t = tgt[:1, 0:1].float() / 255.0
    b, st = bicc_term(lg, soft_t, tgt[:1, 2:3]); b.backward()
    assert float(st["bicc_fp"]) == 1 and (lg.grad[..., 50:55, 50:60] > 0).all(), st
    soft_t[..., 52, 55] = 1 / 3                                     # one annotator touches the blob
    b2, st2 = bicc_term(lg.detach(), soft_t, tgt[:1, 2:3])
    assert float(st2["bicc_fp"]) == 0, "blob with annotator support must not be a false positive"
''')
c30 = rep(c30, 'print("PASS: PQ matching, D4 inversion, hole filling, contact split, offsets, features, loss gradients")',
          'print("PASS: PQ matching, D4 inversion, hole filling, contact split, offsets, features, loss gradients, BiCC")')
md("## Cell 14 — Self-tests (PQ, extraction, TTA inversion, loss gradients, BiCC on the device)")
code(c30)

# ---- Cell 15: training utilities (3-channel preflight target, own signature)
c32 = src[32]
c32 = rep(c32, "y = torch.randint(0, 256, (mb, 2, S, S), dtype=torch.uint8, device=DEVICE)",
          "y = torch.randint(0, 256, (mb, 3, S, S), dtype=torch.uint8, device=DEVICE)")
c32 = rep(c32, '"impl": "sfsc-v3-unet-20261010"', '"impl": "sfsc-v6-bicc-20261024"')
md("## Cell 15 — Training utilities: EMA, param groups, warmup-cosine schedule, checkpoints, micro-batch preflight")
code(c32)

# ---- Cell 16: training loop (BiCC in the log, push, slowdown kill rule)
c34 = src[34]
c34 = rep(c34, '''f"edge {row['edge']:.3f} ct {row['contact']:.3f}) | gn''',
          '''f"edge {row['edge']:.3f} ct {row['contact']:.3f} bicc {row['bicc']:.3f} K {row['bicc_k']:.1f} "
                  f"FP {row['bicc_fp']:.2f}) | gn''')
c34 = rep(c34, '''            if not math.isfinite(row["loss"]):
                raise FloatingPointError(f"Non-finite loss in fold {fold}, epoch {epoch + 1}")
''', '''            if not math.isfinite(row["loss"]):
                raise FloatingPointError(f"Non-finite loss in fold {fold}, epoch {epoch + 1}")
            if not CFG.debug_run and epoch == start_epoch and train_s > MAX_EPOCH_S:
                pd.DataFrame(history + [row]).to_csv(fdir / "history_aborted.csv", index=False)
                raise RuntimeError(f"KILL RULE: epoch {epoch + 1} took {train_s:.0f}s > {MAX_EPOCH_S:.0f}s "
                                   f"(> 1.5 x V3's {V3_EPOCH_S}s): BiCC overhead above 50 %")
''')
c34 = rep(c34, "    write_json(done_path, info)\n", "    write_json(done_path, info)\n    push()\n")
c34 = f"V3_EPOCH_S = {V3_EPOCH_S}                                   # V3 recipe on L4 (V5 step-2 log)\n" \
      "MAX_EPOCH_S = float(os.environ.get(\"SFSC_MAX_EPOCH_S\", 1.5 * V3_EPOCH_S))\n\n" + c34
md("## Cell 16 — Fold training loop (V3: resumable, calibrated-PQ checkpoint pick; + BiCC log, >50 % slowdown kill)")
code(c34)

# ---- Timing cell: A/B per-step cost on identical real crops (debug run only)
md("## Cell 17 — Overhead timing (debug run): V3 loss vs V3 + BiCC on the same real crops, and Voronoi cost per crop")
code(r'''
import dataclasses

def time_overhead(n_warm=5, n_steps=30):
    train_names = FOLDS.loc[FOLDS.fold != 0, "file_name"].tolist()
    ds = CropDatasetV3(train_names, (n_warm + n_steps) * MICRO, 0, CFG)
    # DataLoader side: crop generation with vs without the Voronoi channel (single process, same crops)
    t_crop, t_vor = [], []
    for i in range(24):
        t0 = time.perf_counter(); _, yb = ds[i]; t_crop.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); voronoi_ids_u8(yb[0].numpy()); t_vor.append(time.perf_counter() - t0)
    loader = make_loader(ds, MICRO)
    batches = [b for b in loader]
    loader = None
    crit_off = LossV3(dataclasses.replace(CFG, bicc_weight=0.0)).to(DEVICE)
    model = to_device(build_model(RESOLVED_ENCODER, pretrained=False)).train()
    opt = torch.optim.AdamW(param_groups(model), fused=True)
    res = {}
    for label, crit in (("v3", crit_off), ("v3+bicc", CRITERION), ("v3_again", crit_off), ("v3+bicc_again", CRITERION)):
        times = []
        for i, (xb, yb) in enumerate(batches):
            xb = xb.to(DEVICE, non_blocking=True).contiguous(memory_format=torch.channels_last)
            yb = yb.to(DEVICE, non_blocking=True)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with autocast_context():
                out = model(xb)
            loss, _ = crit(out, yb)
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if i >= n_warm:
                times.append(time.perf_counter() - t0)
        res[label] = float(np.median(times))
    v3 = (res["v3"] + res["v3_again"]) / 2; vb = (res["v3+bicc"] + res["v3+bicc_again"]) / 2
    out = {"gpu_step_s": res, "gpu_overhead_pct": round(100 * (vb / v3 - 1), 1), "micro": MICRO,
           "crop_ms_median": round(1e3 * float(np.median(t_crop)), 1),
           "voronoi_ms_median": round(1e3 * float(np.median(t_vor)), 1),
           "loader_workers": WORKERS,
           "loader_bound_crops_per_s": round(WORKERS / float(np.median(t_crop)), 1),
           "needed_crops_per_s_at_v3_speed": round(MICRO / v3, 1)}
    del model, opt; gc.collect(); torch.cuda.empty_cache()
    return out

if TIMING:
    TIMING_RESULT = time_overhead()
    write_json(RUN / "timing.json", TIMING_RESULT)
    print(json.dumps(TIMING_RESULT, indent=1))
''')

md("## Cell 18 — Train fold 0")
code(src[36])

# ---- OOF maps (V3 as is) + summary
md("## Cell 19 — Out-of-fold probability maps for fold 0, full D4 TTA (same format as V3's `oof/`)")
code(src[38] + r'''
summary = {"folds": TRAINED_FOLDS, "oof_dirs": sorted(p.name for p in OOF_DRIVE.iterdir() if p.is_dir()),
           "oof_files": {p.name: len(list(p.glob("*.npz"))) for p in OOF_DRIVE.iterdir() if p.is_dir()},
           "fold_info": {f: json.loads((RUN / f"fold{f}" / "done.json").read_text()) for f in TRAINED_FOLDS},
           "bicc": {"weight": CFG.bicc_weight, "alpha": CFG.bicc_alpha},
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
write_json(RUN / "summary_v6.json", summary)
_SYNC_STOP.set()
if _SYNC_THREAD.is_alive():
    _SYNC_THREAD.join()                 # no concurrent rclone copies (they create duplicate files on Drive)
push()
print(json.dumps(summary, indent=1, default=str))
''')

nb = {"cells": [], "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
                                "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for kind, s in cells:
    c = {"cell_type": kind, "metadata": {}, "source": s.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    nb["cells"].append(c)
OUT.write_text(json.dumps(nb, indent=1))
allsrc = "".join("".join(c["source"]) for c in nb["cells"])
assert BICC_DATA in allsrc and BICC_LOSS in allsrc, "BiCC source not inlined verbatim"
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        compile("".join(c["source"]), f"cell{i}", "exec")
print("wrote", OUT, len(nb["cells"]), "cells")
