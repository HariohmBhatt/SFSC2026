# ==== SECTION: dataset ====
BICC_K = 256          # Voronoi cell ids live in target channel 2 as uint8 (0 = no consensus component in the crop)
_BICC_STRUCT = np.ones((3, 3), bool)

def voronoi_ids_u8(soft_u8):
    """BiCC annotation side, computed in the DataLoader workers (CPU) on the final augmented crop.

    GT components = consensus pixels (soft target >= 0.5, i.e. uint8 >= 128 -- the same mask V3's Lovasz term uses),
    8-connected. Every pixel of the crop gets the id (1..255) of its nearest component (Euclidean Voronoi partition).
    0 everywhere if the crop has no consensus component. Ids above 255 (never seen in practice) fold into 255.
    """
    lab, n = ndi.label(np.asarray(soft_u8) >= 128, structure=_BICC_STRUCT)
    if n == 0:
        return np.zeros(lab.shape, np.uint8)
    _, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    return np.minimum(lab[iy, ix], 255).astype(np.uint8)
# ==== SECTION: loss ====
def bicc_term(fg_logit, soft, vor_u8, alpha=0.5):
    """2-D BiCC-style component loss (idea of arXiv 2609.30223; follows the independent SFSC implementation of
    theainerd/filament_segmentation, adapted to soft multi-annotator targets).

    fg_logit (B,1,H,W) float logits; soft (B,1,H,W) mean-annotator target in [0,1]; vor_u8 (B,1,H,W) Voronoi ids.
    Annotation side: every Voronoi cell k of a consensus component contributes e_k = 1 - softDice(p, soft | cell k).
    Prediction side: every 8-connected component of (p > 0.5) that touches NO pixel with annotator support
    (soft > 0) contributes f_j = mean p over the component. Components touching any support -- including
    sub-consensus, single-annotator filaments -- are never treated as false positives.
    loss = 2 * (alpha * sum_k e_k + (1 - alpha) * sum_j f_j) / (K + J); 0 when the micro-batch has no object.
    Region sums run on the device (index_add); only the predicted-component labelling runs on the CPU (scipy),
    on a uint8 code map, and only the false-positive pixels' indices travel back.
    """
    B = fg_logit.shape[0]
    dev = fg_logit.device
    p = torch.sigmoid(fg_logit.float())
    pf, yf = p.reshape(-1), soft.float().reshape(-1)
    # ---- annotation side (device only, no host sync)
    gid = (vor_u8.reshape(B, -1).long() + torch.arange(B, device=dev)[:, None] * BICC_K).reshape(-1)
    n = B * BICC_K
    z = torch.zeros(n, device=dev, dtype=pf.dtype)
    inter = z.index_add(0, gid, pf * yf)
    psum = z.index_add(0, gid, pf)
    ysum = z.index_add(0, gid, yf)
    hsum = z.index_add(0, gid, (yf >= 0.5).to(pf.dtype))
    valid = ((torch.arange(n, device=dev) % BICC_K) != 0) & (hsum > 0)
    e = 1.0 - (2.0 * inter + 1.0) / (psum + ysum + 1.0)
    e_sum = (e * valid).sum()
    n_e = valid.sum()
    # ---- prediction side (labelling on CPU, no gradient through it)
    with torch.no_grad():
        code = ((p > 0.5).to(torch.uint8) + 2 * (soft > 0).to(torch.uint8)).reshape(B, *p.shape[-2:]).cpu().numpy()
        HW = code.shape[1] * code.shape[2]
        idx_all, id_all, nfp = [], [], 0
        for b in range(B):
            lab, nc = ndi.label((code[b] & 1).astype(bool), structure=_BICC_STRUCT)
            if nc == 0:
                continue
            touched = np.unique(lab[(code[b] & 2) > 0])
            keep = np.ones(nc + 1, bool)
            keep[0] = False
            keep[touched] = False
            k = int(keep.sum())
            if k == 0:
                continue
            remap = np.zeros(nc + 1, np.int64)
            remap[keep] = np.arange(k) + nfp
            flat = lab.ravel()
            ix = np.flatnonzero(keep[flat])
            idx_all.append(ix + b * HW)
            id_all.append(remap[flat[ix]])
            nfp += k
    if nfp:
        ix = torch.from_numpy(np.concatenate(idx_all)).to(dev, non_blocking=True)
        fid = torch.from_numpy(np.concatenate(id_all)).to(dev, non_blocking=True)
        s = torch.zeros(nfp, device=dev, dtype=pf.dtype).index_add(0, fid, pf[ix])
        cnt = torch.bincount(fid, minlength=nfp).to(pf.dtype)
        f_sum = (s / cnt).sum()
    else:
        f_sum = pf.new_zeros(())
    n_obj = n_e.to(pf.dtype) + float(nfp)
    loss = 2.0 * (alpha * e_sum + (1.0 - alpha) * f_sum) / n_obj.clamp_min(1.0)
    stats = {"bicc_k": n_e.detach().to(pf.dtype), "bicc_fp": pf.new_tensor(float(nfp))}
    return loss, stats
