"""Small, legible PNG/PDF report figures with deterministic OOF selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from . import morphology as m
except ImportError:
    import morphology as m

COLORS={"v4_evidence":"#d65f2e","v3_rescored":"#386cb0","inter_annotator":"#27865b"}
LABELS={"v4_evidence":"V4","v3_rescored":"V3 rescored","inter_annotator":"Human ↔ human"}
ORDER=["v4_evidence","v3_rescored","inter_annotator"]


def style():
    plt.rcParams.update({"font.size":7,"axes.titlesize":8,"axes.labelsize":7,"xtick.labelsize":6.5,
                         "ytick.labelsize":6.5,"legend.fontsize":6.5,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42,"savefig.dpi":220})


def save(fig,path):
    for suffix in ("png","pdf"):
        fig.savefig(Path(path).with_suffix("."+suffix),bbox_inches="tight",pad_inches=.045)
    plt.close(fig)


def _point(ax,row,x,color,label=None):
    y=row.estimate
    ax.errorbar(x,y,yerr=np.array([[max(0,y-row.ci_low)],[max(0,row.ci_high-y)]]),
                fmt="o",color=color,markersize=3,capsize=2,label=label,linewidth=.9)


def plot_metrics(results):
    results=Path(results);style()
    ci=pd.read_csv(results/"metrics_ci.csv");allci=ci[ci.stratum=="all"]
    gt=pd.read_csv(results/"gt_instances.csv.gz")
    matches=pd.read_csv(results/"matches.csv.gz")
    def get(model,metric):return allci[(allci.model==model)&(allci.metric==metric)].iloc[0]
    fig,axs=plt.subplots(2,3,figsize=(7.2,4.6),layout="constrained")
    ax=axs[0,0]
    for j,model in enumerate(ORDER):
        for i,metric in enumerate(("pq","sq","rq")):
            _point(ax,get(model,metric),i+(j-1)*.16,COLORS[model],LABELS[model] if i==0 else None)
    ax.set(xticks=range(3),xticklabels=["PQ","SQ","RQ"],ylim=(.25,.8),title="A  Pooled quality (95% date CI)")
    ax.legend(loc="lower right",frameon=False)
    ax=axs[0,1]
    for model in ORDER:
        rs=[get(model,f"coverage_{t}") for t in (1,3,5)]
        ys=np.array([r.estimate for r in rs]);lo=[r.ci_low for r in rs];hi=[r.ci_high for r in rs]
        ax.plot([1,3,5],ys,"o-",ms=3,lw=1,color=COLORS[model])
        ax.fill_between([1,3,5],lo,hi,color=COLORS[model],alpha=.12)
    ax.set(xlabel="Native-pixel tolerance",ylabel="Mean GT spine coverage",xticks=[1,3,5],ylim=(0,1),title="B  Missed GT objects count as zero")
    ax=axs[0,2]
    for j,model in enumerate(ORDER):_point(ax,get(model,"gap_3"),j,COLORS[model])
    ax.set(xticks=range(3),xticklabels=["V4","V3","Human"],ylabel="Mean longest uncovered fraction",ylim=(0,1),title="C  Spine continuity at 3 px")
    ax=axs[1,0]
    for model in ORDER:
        vals=np.sort(matches.loc[matches.model==model,"iou"].to_numpy())
        ax.plot(vals,np.arange(1,len(vals)+1)/len(vals),color=COLORS[model],lw=1,label=LABELS[model])
    ax.set(xlabel="Matched IoU (>0.5)",ylabel="Cumulative fraction",title="D  Conditional outline agreement",xlim=(.5,1),ylim=(0,1))
    totals=pd.read_csv(results/"pooled_counts.csv").set_index("model")
    miss=["Unmatched GT / prediction"]+[f"{LABELS[model].replace(' rescored','').replace('Human ↔ human','Human')}: {int(totals.loc[model,'fn']):,} / {int(totals.loc[model,'fp']):,}" for model in ORDER]
    ax.text(.98,.035,"\n".join(miss),transform=ax.transAxes,ha="right",va="bottom",fontsize=5.8,
            bbox=dict(facecolor="white",edgecolor="none",alpha=.85,pad=1))
    ax=axs[1,1]
    levels=["<500","500–1499","1500–4999","≥5000"]
    for j,model in enumerate(ORDER):
        for i,level in enumerate(levels):
            row=ci[(ci.model==model)&(ci.metric=="gt_match_rate")&(ci.stratum=="gt_area_bucket")&(ci.level==level)]
            if len(row):_point(ax,row.iloc[0],i+(j-1)*.13,COLORS[model])
    ax.set(xticks=range(4),xticklabels=["<500","500–\n1499","1500–\n4999","≥5000"],xlabel="GT area (native pixels)",ylabel="GT match rate",ylim=(0,1),title="E  Detection versus annotated size")
    ax=axs[1,2]
    for j,model in enumerate(ORDER):_point(ax,get(model,"length_ratio"),j,COLORS[model])
    ax.axhline(1,color=".5",lw=.7,ls="--")
    ax.set(xticks=range(3),xticklabels=["V4","V3","Human"],ylabel="Mean predicted skeleton / GT spine",title="F  Conditional length (branches count)")
    save(fig,results/"morphology_metrics")

    fig,axs=plt.subplots(2,2,figsize=(7.2,4.2),layout="constrained")
    settings=[("gt_width_bucket",["<5","5–<10","10–<20","≥20"],"GT width, area / spine (px)"),
              ("radial_bucket",["centre (<0.5R)","middle (0.5–<0.8R)","limb (≥0.8R)"],"GT radial position"),
              ("support_bucket",["single record","none","partial","all"],"Agreement with other annotator records"),
              ("n_annotators",["1","2","3"],"Annotator records per image")]
    for ax,(col,levels,xlabel) in zip(axs.ravel(),settings):
        for j,model in enumerate(ORDER):
            for i,level in enumerate(levels):
                row=ci[(ci.model==model)&(ci.metric=="coverage_3")&(ci.stratum==col)&(ci.level.astype(str)==level)]
                if len(row):_point(ax,row.iloc[0],i+(j-1)*.13,COLORS[model])
        labels=[x.replace(" (","\n(").replace("single record","single\nrecord") for x in levels]
        ax.set(xticks=range(len(levels)),xticklabels=labels,xlabel=xlabel,ylabel="Mean spine coverage at 3 px",ylim=(0,1))
    axs[0,0].legend(handles=[Line2D([0],[0],marker="o",lw=0,color=COLORS[x],label=LABELS[x]) for x in ORDER],frameon=False,loc="lower right")
    save(fig,results/"morphology_strata")


def select_examples(results,multi_record=False):
    r=pd.read_csv(Path(results)/"records.csv.gz")
    r=r[(r.model=="v4_evidence") & ((r.n_annotators>1) if multi_record else True)].groupby("file_name",as_index=False)[["sum_iou","tp","fp","fn"]].sum()
    r["pq"]=r.sum_iou/(r.tp+.5*(r.fp+r.fn))
    r=r.sort_values(["pq","file_name"],kind="stable").reset_index(drop=True)
    rows=[]
    for label,q in (("Poor",.1),("Median",.5),("Good",.9)):
        i=int(np.floor(q*(len(r)-1)+.5));row=r.iloc[i]
        rows.append(dict(label=label,quantile=q,rank=i,file_name=row.file_name,pq=float(row.pq)))
    return rows


def plot_examples(dataset,scratch,results,multi_record=False):
    style();results=Path(results)
    _,index=m.annotation_index(dataset)
    selected=select_examples(results,multi_record)
    maxrec=max(len(index[r["file_name"]]) for r in selected)
    for zoom in ((True,) if multi_record else (False,True)):
        fig,axs=plt.subplots(3,maxrec+2,figsize=(min(7.2,(maxrec+2)*1.55),4.8),squeeze=False,
                             gridspec_kw={"wspace":.025,"hspace":.14})
        for row,example in enumerate(selected):
            name=example["file_name"];recs=index[name]
            example["annotator_record_ids"]=[rec["id"] for rec in recs]
            files=list((Path(dataset)/"train").rglob(name));assert len(files)==1
            image=np.asarray(Image.open(files[0]).convert("L"))
            h,w=image.shape
            gt=[m.record_masks(rec) for rec in recs]
            largest=max(gt[0],key=lambda mask:int(mask.sum()))
            ys,xs=np.nonzero(largest)
            if zoom:
                yc,xc=(ys.min()+ys.max())/2,(xs.min()+xs.max())/2
                half=max(96,(ys.max()-ys.min())/2+32,(xs.max()-xs.min())/2+32)
                box=[max(0,int(xc-half)),min(w,int(xc+half+1)),max(0,int(yc-half)),min(h,int(yc+half+1))]
                example["crop_box_xyxy"]=[box[0],box[2],box[1],box[3]]
            else:box=[0,w,0,h]
            x0,x1,y0,y1=box
            crop=image[y0:y1,x0:x1]
            if zoom:vmin,vmax=np.percentile(crop,[1,99])
            else:vmin,vmax=0,255
            z=np.load(Path(scratch)/"masks"/f"{Path(name).stem}.npz")
            for col in range(maxrec+2):
                ax=axs[row,col];ax.set_xticks([]);ax.set_yticks([])
                ax.imshow(crop,cmap="gray",vmin=vmin,vmax=vmax,origin="upper")
                masks=[];spines=[]
                if col<maxrec:
                    title=f"Annotator {col+1}"
                    if col<len(recs):
                        masks=gt[col];spines=[a.get("spine",[]) for a in recs[col]["annotations"]]
                        color="#22d3d3"
                    else:
                        ax.text(.5,.5,"No record",transform=ax.transAxes,ha="center",color="white",fontsize=7)
                else:
                    model="v4_evidence" if col==maxrec else "v3_rescored"
                    title="V4" if col==maxrec else "V3 rescored"
                    lab=z[model];masks=[lab==i for i in range(1,int(lab.max())+1)];color=COLORS[model]
                for mask in masks:
                    sub=mask[y0:y1,x0:x1]
                    if sub.any() and not sub.all():ax.contour(sub,levels=[.5],colors=[color],linewidths=.7 if zoom else .45)
                for spine in spines:
                    p=np.asarray(spine).reshape(-1,2)
                    if len(p):ax.plot(p[:,0]-x0-.5,p[:,1]-y0-.5,color="#ffe46b",lw=.65)
                ax.set(xlim=(-.5,x1-x0-.5),ylim=(y1-y0-.5,-.5))
                if col==0:
                    ax.text(0,1.02,f"{example['label']} · PQ {example['pq']:.3f}",transform=ax.transAxes,fontsize=7,va="bottom")
                    ax.set_ylabel(name.replace(".jpeg",""),fontsize=5.5,labelpad=3)
            # Native scale bar on prediction panel in both views.
            ax=axs[row,-1];length=50 if zoom else 500
            if length < (x1-x0)*.8:
                ax.plot([8,8+length],[y1-y0-12]*2,color="white",lw=1.5)
                ax.text(8,y1-y0-18,f"{length} px",color="white",fontsize=5.5)
        fig.subplots_adjust(left=.055,right=.998,top=.90,bottom=.035)
        fig.canvas.draw()
        titles=[f"Annotator {i+1}" for i in range(maxrec)]+["V4","V3 rescored"]
        for ax,title in zip(axs[0],titles):
            pos=ax.get_position()
            fig.text((pos.x0+pos.x1)/2,.975,title,ha="center",va="top",fontsize=8)
        stem="oof_multi_annotator_crops" if multi_record else "oof_examples_crops" if zoom else "oof_examples_full"
        save(fig,results/stem)
    manifest="multi_annotator_examples_manifest.json" if multi_record else "examples_manifest.json"
    m.write_json(results/manifest,dict(selection="Nearest rank to V4 image PQ 10th/50th/90th percentiles; ascending PQ, filename tie-break",
                 population="multi-record images only" if multi_record else "all 707 OOF images",
                 crop="Largest first-record GT polygon bounding square +32px, minimum 192px; fixed before viewing",examples=selected))
    return selected


def make_figures(dataset,scratch,results):
    plot_metrics(results)
    selected=plot_examples(dataset,scratch,results)
    plot_examples(dataset,scratch,results,multi_record=True)
    return selected
