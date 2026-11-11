"""Frozen geometry extracted verbatim from fusion_flat.py; no scorer fitting or mask changes.
Source SHA256: ea00048bfc3c2ef58f0471ff3c6c1764766b4d4f7e00c3cbf073a9947beef945
"""
import numpy as np
from scipy import ndimage as ndi
from types import SimpleNamespace
CFG = None
STRUCT8 = ndi.generate_binary_structure(2, 2)
STRUCT4 = ndi.generate_binary_structure(2, 1)
CONN = STRUCT8
F_IDX = {"max_p": 2}

def configure(config):
    global CFG, CONN
    CFG = SimpleNamespace(**config)
    CONN = STRUCT8 if CFG.connectivity == 8 else STRUCT4

def qt(t):
    return int(round(float(t) * 255))

def relabel(lab):
    if lab.max() == 0:
        return np.zeros(lab.shape, np.int32), 0
    areas = np.bincount(lab.ravel())
    ids = np.flatnonzero(areas[1:]) + 1
    lut = np.zeros(len(areas), np.int32)
    lut[ids] = np.arange(1, len(ids) + 1, dtype=np.int32)
    return lut[lab], len(ids)

def fill_label_holes(lab):
    fg = lab > 0
    holes = ndi.binary_fill_holes(fg) & ~fg
    if not holes.any():
        return lab
    hl, _ = ndi.label(holes, structure=STRUCT4)
    out = lab.copy()
    for i, sl in enumerate(ndi.find_objects(hl), 1):
        if sl is None:
            continue
        esl = (slice(max(sl[0].start - 1, 0), sl[0].stop + 1), slice(max(sl[1].start - 1, 0), sl[1].stop + 1))
        hole = hl[esl] == i
        ring = ndi.binary_dilation(hole, STRUCT8) & ~hole
        vals = lab[esl][ring]
        vals = vals[vals > 0]
        if vals.size:
            out[esl][hole] = np.bincount(vals).argmax()
    return out

def geometry_labels(fg, ct, geom):
    t_low, contact_t, offset = geom
    low = fg >= qt(t_low)
    if contact_t is not None:
        cores = low & (ct < qt(contact_t))
        lab, n = ndi.label(cores, structure=CONN)
        if n:
            areas = np.bincount(lab.ravel(), minlength=n + 1)
            small = areas < CFG.min_core_area
            small[0] = False
            if small.any():
                lab[small[lab]] = 0
        if lab.any():
            dist, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_distances=True, return_indices=True)
            lab = np.where(low & (dist <= CFG.contact_assign_radius), lab[iy, ix], 0).astype(np.int32)
        else:
            lab = np.zeros((fg.shape), np.int32)
    else:
        lab, _ = ndi.label(low, structure=CONN)
        lab = lab.astype(np.int32)
    if CFG.fill_holes and lab.any():
        lab = fill_label_holes(lab)
    for _ in range(max(offset, 0)):
        lab = np.where(lab == 0, ndi.grey_dilation(lab, footprint=STRUCT4), lab)
    for _ in range(max(-offset, 0)):
        lab = np.where(ndi.grey_erosion(lab, footprint=STRUCT4) == lab, lab, 0)
    return relabel(lab)

def prefilter_keep(feats, area):
    if len(area) == 0:
        return np.zeros(0, bool)
    return ((np.rint(feats[:, F_IDX["max_p"]] * 255) >= qt(CFG.scorer_prefilter_max))
            & (area >= CFG.scorer_prefilter_area))

