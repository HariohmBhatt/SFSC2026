"""Report figure: one 3-record training image -> inputs, soft target, V3 OOF probability with annotator outlines."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, cv2, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pycocotools import mask as coco

DS = Path.home() / "SFSC2026/SFSCDataset/train"
V3 = Path.home() / "sfsc_work/v5fusion/drive_mirror/outputs/sfsc_unet_v3"
OUT = Path(sys.argv[1])
raw = json.loads((DS / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text())
anns = defaultdict(list)
for a in raw["annotations"]:
    anns[a["image_id"]].append(a)
recs = defaultdict(list)
for im in raw["images"]:
    recs[Path(im["file_name"]).name].append(im)
folds = pd.read_csv(V3 / "folds_v3.csv").set_index("file_name").fold.to_dict()
# deterministic choice: 3-record images, first by name whose records have 6-12 filaments each
cands = sorted(n for n, r in recs.items() if len(r) == 3 and all(6 <= len(anns[i["id"]]) <= 12 for i in r))
name = cands[int(sys.argv[2]) if len(sys.argv) > 2 else 0]
fold = folds[name]
oofdir = next((V3 / "oof").glob(f"fold{fold}_*"))
z = np.load(oofdir / name.replace(".jpeg", ".npz"))
fg, ct = z["fg"].astype(np.float32) / 255, z["ct"].astype(np.float32) / 255
img = cv2.imread(str(DS / "train_images" / name), cv2.IMREAD_GRAYSCALE)
H, W = img.shape
masks = []
for im in sorted(recs[name], key=lambda x: x["id"]):
    rm = []
    for a in anns[im["id"]]:
        seg = a["segmentation"]
        rle = coco.merge(coco.frPyObjects(seg, H, W)) if isinstance(seg, list) else seg
        rm.append(coco.decode(rle).astype(bool))
    masks.append(rm)
soft = np.mean([np.any(r, 0) for r in masks], 0)
# flattened channel (same recipe as V3 flatten_disk, simplified radial profile)
yy, xx = np.mgrid[:H, :W]; cy, cx = (H - 1) / 2, (W - 1) / 2
R = 0.88 * H / 2; r = np.hypot(yy - cy, xx - cx) / R; disk = r <= 1
rb = np.minimum((r * 511).astype(int), 511)
prof = np.bincount(rb[disk], img[disk], 512) / np.maximum(np.bincount(rb[disk], minlength=512), 1)
ratio = np.where(disk, img / np.maximum(prof[rb], 1), 1.0).astype(np.float32)
small = cv2.resize(ratio, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
med = cv2.resize(cv2.medianBlur(np.clip(small * 128, 0, 255).astype(np.uint8), 41).astype(np.float32) / 128, (W, H))
flat = np.where(disk, ratio / np.maximum(med, .25), 1.0)
# crop around the densest annotated region
ys, xs = np.nonzero(soft > 0)
s = 900
c0y = int(np.clip(np.median(ys) - s // 2, 0, H - s)); c0x = int(np.clip(np.median(xs) - s // 2, 0, W - s))
sl = (slice(c0y, c0y + s), slice(c0x, c0x + s))
fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.95), dpi=200)
ax[0].imshow(img[sl], cmap="gray"); ax[0].set_title("(a) raw H$\\alpha$", fontsize=7)
ax[1].imshow(np.clip(flat[sl], .6, 1.3), cmap="gray"); ax[1].set_title("(b) limb-flattened", fontsize=7)
ax[2].imshow(soft[sl], cmap="magma", vmin=0, vmax=1); ax[2].set_title("(c) soft target (3 records)", fontsize=7)
ax[3].imshow(fg[sl], cmap="viridis", vmin=0, vmax=1); ax[3].set_title("(d) V3 OOF $p_{fil}$ + outlines", fontsize=7)
for k, (rm, col) in enumerate(zip(masks, ["#ff4d4d", "#ffffff", "#4dd2ff"])):
    for m in rm:
        ax[3].contour(m[sl].astype(float), levels=[.5], colors=[col], linewidths=.45)
for a in ax: a.set_xticks([]); a.set_yticks([])
plt.tight_layout(pad=.25)
fig.savefig(OUT, bbox_inches="tight")
print(name, "fold", fold, "crop", c0y, c0x, "records", [len(r) for r in masks])
