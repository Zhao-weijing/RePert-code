#!/usr/bin/env python3
"""Prepare plate-level GE effects and compound-dose aggregates for cpg0004."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

DOSES=(0.04,0.12,0.37,1.11,3.33,10.0)
REF={"BRD-K50691590","BRD-K60230970"}
META=["cid","cell_id","det_plate","det_well","rna_plate","rna_well","pert_dose","pert_id","pert_idose","pert_iname_x","pert_iname_y","pert_time","pert_type_x","pert_type_y","pert_vehicle","x_smiles","pert_plate","batch","brew_prefix","group_id","pert_id_dose","pert_type","control_type"]

def dose(v):
    try: x=float(v)
    except (TypeError,ValueError): return ""
    if not np.isfinite(x): return ""
    y=min(DOSES,key=lambda z:abs(z-x))
    return f"{y:g}" if abs(y-x)<=0.08 else ""

def clean(v):
    if v is None or (isinstance(v,float) and np.isnan(v)): return ""
    s=str(v).strip(); return "" if s.lower() in {"nan","none","<na>"} else s

def enc(values):
    return np.asarray(values,dtype=f"S{max([1]+[len(str(x).encode()) for x in values])}")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,default=Path("/path/to/home/AIDD/baseline/data/MVC/LINCS-Pilot1/L1000/replicate_level_l1k.csv.gz")); p.add_argument("--outdir",type=Path,required=True); p.add_argument("--missing-threshold",type=float,default=.20); p.add_argument("--force",action="store_true"); a=p.parse_args()
    if a.outdir.exists() and any(a.outdir.iterdir()) and not a.force: raise FileExistsError(a.outdir)
    a.outdir.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.input,low_memory=False)
    missing=[c for c in META if c not in df.columns]
    if missing: raise ValueError(f"missing metadata: {missing}")
    features=[c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    tr=df["pert_type"].astype(str).str.strip().eq("trt"); ctl=df["pert_type"].astype(str).str.strip().eq("control")
    compounds=df["pert_id"].map(clean)
    dose_labels=df["pert_id_dose"].map(lambda x:dose(clean(x).rsplit("_",1)[-1]) if "_" in clean(x) else dose(x))
    keep=tr & dose_labels.ne("") & ~compounds.isin(REF)
    treatment=df.loc[keep,features].apply(pd.to_numeric,errors="coerce").reset_index(drop=True)
    control=df.loc[ctl,features].apply(pd.to_numeric,errors="coerce").reset_index(drop=True)
    combined=pd.concat([treatment,control],ignore_index=True)
    kept=[c for c in features if float(combined[c].isna().mean())<=a.missing_threshold]
    fallback=combined[kept].median(); med=control[kept].median().fillna(fallback); kept=[c for c in kept if pd.notna(med[c])]
    treatment=treatment[kept].fillna(med[kept]); control=control[kept].fillna(med[kept])
    ctl_meta=df.loc[ctl,["det_plate"]].copy(); ctl_meta["plate"]=ctl_meta["det_plate"].map(clean).to_numpy(dtype=str); ctl_meta=ctl_meta.reset_index(drop=True)
    ctl_values=control.to_numpy(dtype=np.float32); plate_ctl={}
    for plate,ix in ctl_meta.groupby("plate",sort=False).groups.items():
        if plate: plate_ctl[plate]=np.median(ctl_values[np.asarray(list(ix),dtype=np.int64)],axis=0).astype(np.float32)
    tmeta=pd.DataFrame({"compound_id":compounds.loc[keep].to_numpy(dtype=str),"dose":dose_labels.loc[keep].to_numpy(dtype=str),"det_plate":df.loc[keep,"det_plate"].map(clean).to_numpy(dtype=str),"det_well":df.loc[keep,"det_well"].map(clean).to_numpy(dtype=str),"batch":df.loc[keep,"batch"].map(clean).to_numpy(dtype=str),"rna_plate":df.loc[keep,"rna_plate"].map(clean).to_numpy(dtype=str),"rna_well":df.loc[keep,"rna_well"].map(clean).to_numpy(dtype=str)})
    tmeta["source_row"]=np.flatnonzero(keep.to_numpy()).astype(np.int64); vals=treatment.to_numpy(dtype=np.float32); deltas=[]; rows=[]
    for i,r in tmeta.iterrows():
        plate=r.det_plate
        if plate not in plate_ctl: raise ValueError(f"no GE control for {plate}")
        deltas.append(vals[i]-plate_ctl[plate]); rows.append(i)
    deltas=np.vstack(deltas).astype(np.float32)
    raw_meta=tmeta.copy(); raw_meta["technical_duplicate_count"]=1
    # Multiple wells for one compound-dose-det_plate are technical repeats;
    # collapse them to one independent detection-plate observation.
    group_cols=["compound_id","dose","det_plate"]
    out_rows=[]; out_vals=[]; out_base=[]
    for key,ix0 in tmeta.groupby(group_cols,sort=True).groups.items():
        ix=np.asarray(list(ix0),dtype=np.int64); g=tmeta.iloc[ix]; plate=key[2]
        out_rows.append({"compound_id":key[0],"dose":key[1],"det_plate":key[2],"det_well":";".join(sorted(set(g.det_well))),"batch":";".join(sorted(set(g.batch))),"rna_plate":";".join(sorted(set(g.rna_plate))),"rna_well":";".join(sorted(set(g.rna_well))),"source_rows":";".join(map(str,sorted(g.source_row))),"technical_duplicate_count":int(len(ix))})
        out_vals.append(np.median(deltas[ix],axis=0).astype(np.float32)); out_base.append(plate_ctl[plate])
    out_meta=pd.DataFrame(out_rows); out_delta=np.vstack(out_vals).astype(np.float32); out_base=np.vstack(out_base).astype(np.float32)
    if not np.isfinite(out_delta).all(): raise ValueError("non-finite GE delta")
    np.savez_compressed(a.outdir/"ge_plate_rows.npz",compound_id=enc(out_meta.compound_id.tolist()),dose=enc(out_meta.dose.tolist()),det_plate=enc(out_meta.det_plate.tolist()),det_well=enc(out_meta.det_well.tolist()),batch=enc(out_meta.batch.tolist()),delta=out_delta,baseline=out_base)
    # The observed GE evidence available to a CP prediction is the mean of all
    # independent detection plates for the same compound-dose.
    agg_rows=[]; agg_values=[]
    for (compound,label),ix0 in out_meta.groupby(["compound_id","dose"],sort=True).groups.items():
        ix=np.asarray(list(ix0),dtype=np.int64); agg_rows.append({"compound_id":compound,"dose":label,"ge_repeat_count":int(len(ix)),"det_plates":";".join(out_meta.iloc[ix].det_plate),"det_wells":";".join(out_meta.iloc[ix].det_well) }); agg_values.append(out_delta[ix].mean(axis=0,dtype=np.float64).astype(np.float32))
    agg=pd.DataFrame(agg_rows); np.savez_compressed(a.outdir/"ge_condition_aggregates.npz",compound_id=enc(agg.compound_id.tolist()),dose=enc(agg.dose.tolist()),ge_repeat_count=agg.ge_repeat_count.to_numpy(dtype=np.int16),ge_effect=np.vstack(agg_values).astype(np.float32)); out_meta.to_csv(a.outdir/"ge_plate_manifest.csv",index=False); agg.to_csv(a.outdir/"ge_condition_manifest.csv",index=False); pd.DataFrame({"feature":kept}).to_csv(a.outdir/"feature_names.csv",index=False)
    summary={"version":"cpg0004-LINCS-GE-plate-rows-2026-08-30","source":str(a.input),"source_sha256":hashlib.sha256(a.input.read_bytes()).hexdigest(),"raw_rows":int(len(df)),"raw_treatment_rows":int(tr.sum()),"raw_control_rows":int(ctl.sum()),"retained_feature_count":len(kept),"included_plate_rows":int(len(out_meta)),"included_condition_count":int(len(agg)),"included_compound_count":int(agg.compound_id.nunique()),"det_plate_count":int(out_meta.det_plate.nunique()),"repeat_distribution":{str(k):int(v) for k,v in agg.ge_repeat_count.value_counts().sort_index().items()},"dose_labels":sorted(agg.dose.unique().tolist(),key=float),"technical_duplicate_extra_rows":int(tmeta.groupby(group_cols).size().sub(1).clip(lower=0).sum()),"guardrails":["GE has no CP target or model input in this preparation","control median is per det_plate","technical wells are median-collapsed within det_plate","observed evidence is mean across independent det_plate rows","20-mM reference compounds excluded"]}
    (a.outdir/"PREPARATION_SUMMARY.json").write_text(json.dumps(summary,indent=2,sort_keys=True),encoding="utf-8"); print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=="__main__": main()
