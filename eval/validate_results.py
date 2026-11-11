"""End-to-end consistency checks on the measured scientific output tables."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from . import morphology as m
except ImportError:
    import morphology as m


def validate(results):
    results=Path(results)
    tables=m.load_tables(results)
    gt,pred,matches,recs=[tables[k] for k in ("gt_instances","pred_instances","matches","records")]
    ref=pd.read_csv(results/"pq_records.csv")
    for model in m.MODELS:
        r=recs[recs.model==model]
        g=gt[gt.model==model]
        p=pred[pred.model==model]
        assert len(g)==8199 and len(r)==1154 and r.file_name.nunique()==707
        got=r.groupby("file_name")[["sum_iou","tp","fp","fn"]].sum().sort_index()
        expected=ref[(ref.model==model)&(ref.engine=="official")].groupby("file_name")[["sum_iou","tp","fp","fn"]].sum().sort_index()
        np.testing.assert_allclose(got,expected,atol=2e-6,rtol=2e-8) # gzip tables retain 8 significant digits
        h=ref[(ref.model==model)&(ref.engine=="fusion_hungarian")][["tp","fp","fn"]].sum()
        assert int(g.matched.sum())==int(h.tp)
        assert int((g.matched==0).sum())==int(h.fn)
        assert int((p.matched==0).sum())==int(h.fp)
        assert int((g.official_matched==0).sum())==int(r.fn.sum())
        assert int((p.official_matched==0).sum())==int(r.fp.sum())
        assert len(matches[matches.model==model])==int(r.tp.sum())
    human=recs[recs.model=="inter_annotator"]
    assert human.file_name.nunique()==296 and len(human)==1196
    for t in (1,3,5):
        valid=gt[gt[f"coverage_{t}"].notna()]
        assert valid[f"coverage_{t}"].between(-1e-8,1+1e-8).all()
        assert valid[f"gap_{t}"].between(-1e-8,1+1e-8).all()
        assert (valid[f"gap_{t}"]<=1-valid[f"coverage_{t}"]+1e-7).all()
        missed=valid[valid.matched==0]
        assert (missed[f"coverage_{t}"]==0).all() and (missed[f"gap_{t}"]==1).all()
    valid=gt[gt.coverage_1.notna()]
    assert (valid.coverage_1<=valid.coverage_3+1e-8).all()
    assert (valid.coverage_3<=valid.coverage_5+1e-8).all()
    assert (valid.gap_1+1e-8>=valid.gap_3).all()
    assert (valid.gap_3+1e-8>=valid.gap_5).all()
    ci=pd.read_csv(results/"metrics_ci.csv")
    assert ci[["estimate","ci_low","ci_high"]].notna().all().all()
    assert (ci.denominator>0).all() and (ci.ci_low<=ci.ci_high).all()
    for stem in ("morphology_metrics","morphology_strata","oof_examples_full","oof_examples_crops","oof_multi_annotator_crops"):
        for suffix in ("png","pdf"):
            assert (results/f"{stem}.{suffix}").stat().st_size>1000
    size=sum(p.stat().st_size for p in results.parent.rglob("*") if p.is_file())
    assert size<30_000_000
    return dict(table_consistency="passed",coverage_and_gap_invariants="passed",human_reference_counts="passed",
                bootstrap_ci_tables="passed",report_figures="passed",eval_bytes=size,
                native_oof_images=707,annotator_records=1154,gt_objects=8199,human_images=296,human_directed_comparisons=1196)
