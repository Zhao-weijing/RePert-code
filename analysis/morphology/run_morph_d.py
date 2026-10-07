#!/usr/bin/env python3
"""Frozen Morph-D data-driven Cell Painting response-program recovery.

K selection sees only high-quality training/validation reference profiles.
The selected PCA basis is then frozen before any test profile is projected.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SEEDS = (3407, 42, 2025); DOSES = ("0.04", "0.12", "0.37", "1.11", "3.33", "10")
KS = (8, 16, 24, 32); STABILITY_ROUNDS = 100; STABILITY_THRESHOLD = .90
METHODS = ("raw", "teacher", "posterior"); LABELS = {"raw":"M0_1R_raw", "teacher":"M1_frozen_teacher", "posterior":"M2_frozen_GE_posterior"}


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()


def seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(("MorphD|"+label).encode()).digest()[:8], "little") % (2**32-1)


def load_module(path: Path):
    spec=importlib.util.spec_from_file_location("frozen_bioc_morphd",path)
    if spec is None or spec.loader is None: raise ImportError(path)
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod); return mod


def parse() -> argparse.Namespace:
    here=Path(__file__).resolve().parent; base=here.parent/"external_validation"/"lincs_cpg0004"
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module",type=Path,default=(Path(__file__).resolve().parents[2] / "analysis/biological_applications/dose_response/run_dose_response.py"))
    p.add_argument("--data",type=Path,default=base/"data_preparation"/"artifact"/"cp_plate_rows.npz")
    p.add_argument("--p0-root",type=Path,default=base/"virtual_prior"/"results"/"1r_all")
    p.add_argument("--ge-root",type=Path,default=base/"single_repeat_expression_evidence"/"results"/"1r_all")
    p.add_argument("--pair-root",type=Path,default=base/"cell_painting_repeat_benchmark"/"results"/"1r_all")
    p.add_argument("--feature-mapping",type=Path,default=here/"feature_annotation_audit"/"FEATURE_MAPPING.csv")
    p.add_argument("--outdir",type=Path,default=here/"response_programs")
    p.add_argument("--bootstrap-rounds",type=int,default=10_000); p.add_argument("--stability-rounds",type=int,default=STABILITY_ROUNDS); p.add_argument("--force",action="store_true")
    return p.parse_args()


def biological_map(path: Path, dim: int) -> pd.DataFrame:
    x=pd.read_csv(path); need={"feature_index","feature_name","biological_endpoint","biological_module","object","channel","measurement_family"}
    if not need.issubset(x.columns) or len(x)!=dim: raise ValueError("Feature mapping not aligned to frozen input")
    x=x[x.biological_endpoint.astype(str).eq("Eligible")].sort_values("feature_index").reset_index(drop=True)
    if len(x)!=241 or int(x.feature_index.max())>=dim or "Batch_Number" in set(x.feature_name): raise ValueError("Expected 241 biological features excluding Batch_Number")
    return x


def high_quality_refs(artifact, features: np.ndarray) -> tuple[dict[str,np.ndarray],dict[str,list[tuple[str,str]]]]:
    data={"train":[],"valid":[]}; keys={"train":[],"valid":[]}
    for (compound,dose), rows in artifact.groups.items():
        split=str(artifact.split[rows[0]])
        if split not in data or len(rows)!=5 or any(str(artifact.split[r])!=split for r in rows): continue
        data[split].append(artifact.delta[np.asarray(rows),:][:,features].mean(axis=0,dtype=np.float64)); keys[split].append((str(compound),str(dose)))
    return {k:np.asarray(v,dtype=np.float64) for k,v in data.items()},keys


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    center=x.mean(axis=0); scale=x.std(axis=0,ddof=0); scale[~np.isfinite(scale)|(scale<1e-6)]=1.; return center,scale


def fit_pca(x: np.ndarray,k:int) -> tuple[np.ndarray,np.ndarray]:
    mean=x.mean(axis=0); _,_,vt=np.linalg.svd(x-mean,full_matrices=False); return mean,vt[:k].T


def reconstruction_rmse(x:np.ndarray,mean:np.ndarray,w:np.ndarray)->np.ndarray:
    recon=(x-mean)@w@w.T+mean; return np.sqrt(np.mean((x-recon)**2,axis=1))


def stability(full_w:np.ndarray,x:np.ndarray,ks:tuple[int,...],rounds:int)->pd.DataFrame:
    rng=np.random.default_rng(seed("basis-stability")); records=[]; n=len(x); draw_n=min(n,2000)
    for b in range(rounds):
        sample=x[rng.integers(0,n,size=draw_n,endpoint=False)]
        _,wb=fit_pca(sample,max(ks))
        for k in ks:
            singular=np.linalg.svd(full_w[:,:k].T@wb[:,:k],compute_uv=False)
            records.append({"bootstrap":b,"k":k,"subspace_stability":float(np.mean(np.clip(singular,0,1)**2))})
    return pd.DataFrame(records)


def select_basis(train:np.ndarray,valid:np.ndarray,rounds:int)->tuple[int,np.ndarray,np.ndarray,pd.DataFrame,pd.DataFrame]:
    c,s=standardize_fit(train); tr=(train-c)/s; va=(valid-c)/s; mean,wmax=fit_pca(tr,max(KS)); st=stability(wmax,tr,KS,rounds)
    rows=[]
    for k in KS:
        errors=reconstruction_rmse(va,mean,wmax[:,:k]); q=float(st[st.k.eq(k)].subspace_stability.quantile(.025))
        rows.append({"k":k,"validation_reconstruction_rmse":float(errors.mean()),"validation_reconstruction_se":float(errors.std(ddof=1)/math.sqrt(len(errors))),"stability_q025":q,"stability_mean":float(st[st.k.eq(k)].subspace_stability.mean()),"stable":bool(q>=STABILITY_THRESHOLD)})
    table=pd.DataFrame(rows); best=float(table.validation_reconstruction_rmse.min()); best_se=float(table.loc[table.validation_reconstruction_rmse.idxmin(),"validation_reconstruction_se"])
    allowed=table[table.stable & (table.validation_reconstruction_rmse<=best+best_se)]
    if allowed.empty:
        table["best_validation_rmse"]=best; table["best_validation_se"]=best_se; table["one_se_eligible"]=table.validation_reconstruction_rmse<=best+best_se; table["selected"]=False
        return None,c,s,None,table,st
    k=int(allowed.k.min()); final=np.vstack([train,valid]); fc,fs=standardize_fit(final); fm,fw=fit_pca((final-fc)/fs,k)
    table["best_validation_rmse"]=best; table["best_validation_se"]=best_se; table["one_se_eligible"]=table.validation_reconstruction_rmse<=best+best_se; table["selected"]=table.k.eq(k)
    return k,fc,fs,np.column_stack([fm,fw]),table,st


def pcc(a:np.ndarray,b:np.ndarray)->float:
    a=np.asarray(a,float);b=np.asarray(b,float); a=a-a.mean();b=b-b.mean();den=float(np.linalg.norm(a)*np.linalg.norm(b)); return float(a@b/den) if den>0 and np.isfinite(den) else math.nan


def comp_metrics(frame:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for key,g in frame.groupby(["seed","compound","method"],sort=True):
        valid=g[np.isfinite(g.coefficient_pcc)&np.isfinite(g.coefficient_rmse)]
        rows.append({"seed":key[0],"compound":key[1],"method":key[2],"profile_count":len(g),"valid_profile_count":len(valid),"coefficient_pcc":float(valid.coefficient_pcc.mean()) if len(g)==12 and len(valid)==12 else math.nan,"coefficient_rmse":float(valid.coefficient_rmse.mean()) if len(g)==12 and len(valid)==12 else math.nan})
    return pd.DataFrame(rows)


def boot(values:np.ndarray,label:str,rounds:int)->tuple[float,float,float,int]:
    values=np.asarray(values,float)
    if not len(values) or not np.isfinite(values).all(): return math.nan,math.nan,math.nan,-1
    s=seed(label); r=np.random.default_rng(s); ix=r.integers(0,len(values),size=(rounds,len(values)),endpoint=False); d=values[ix].mean(axis=1)
    return float(values.mean()),float(np.quantile(d,.025)),float(np.quantile(d,.975)),s


def contrast(metrics:pd.DataFrame,rounds:int)->pd.DataFrame:
    out=[]
    for comparison,left,right in (("teacher_minus_raw","teacher","raw"),("posterior_minus_teacher","posterior","teacher")):
        for endpoint in ("coefficient_pcc","coefficient_rmse"):
            by={}
            for s in SEEDS:
                l=metrics[(metrics.seed.eq(s))&(metrics.method.eq(left))].set_index("compound")[endpoint]; r=metrics[(metrics.seed.eq(s))&(metrics.method.eq(right))].set_index("compound")[endpoint]
                by[s]=pd.Series({c:float(l[c]-r[c]) for c in set(l.index)&set(r.index) if np.isfinite(l[c]) and np.isfinite(r[c])},dtype=float)
            common=sorted(set.intersection(*(set(v.index) for v in by.values()))); pooled=np.asarray([np.mean([by[s][c] for s in SEEDS]) for c in common])
            point,low,high,bs=boot(pooled,comparison+endpoint,rounds)
            out.append({"seed":"mean","comparison":comparison,"endpoint":endpoint,"seed3407":float(by[3407].mean()) if len(by[3407]) else math.nan,"seed42":float(by[42].mean()) if len(by[42]) else math.nan,"seed2025":float(by[2025].mean()) if len(by[2025]) else math.nan,"n_compounds":len(pooled),"point":point,"ci_low":low,"ci_high":high,"rounds":rounds,"bootstrap_unit":"compound","bootstrap_seed":bs})
    return pd.DataFrame(out)


def go(row:pd.Series)->str:
    return "GO" if all(float(row[x])>0 for x in ("seed3407","seed42","seed2025")) and float(row.ci_low)>0 else "NO-GO"


def main()->None:
    a=parse(); out=a.outdir.resolve()
    if a.bootstrap_rounds<=0 or a.stability_rounds<=0: raise ValueError("round counts must be positive")
    if out.exists() and any(out.iterdir()) and not a.force: raise FileExistsError("Use --force")
    (out/"figures").mkdir(parents=True,exist_ok=True); M=load_module(a.module); artifact=M.load_artifact(a.data); fmap=biological_map(a.feature_mapping,artifact.feature_dim); ix=fmap.feature_index.to_numpy(dtype=np.int64)
    refs,refkeys=high_quality_refs(artifact,ix)
    if len(refs["train"])==0 or len(refs["valid"])==0: raise RuntimeError("No high-quality train/valid references")
    k,center,scale,packed,selection,st=select_basis(refs["train"],refs["valid"],a.stability_rounds)
    if k is None:
        reason="NOT_RUN_VALIDATION_BASIS_NO_K_PASSES_STABILITY_AND_ONE_SE_RULE"
        config={"version":"cpg0004-LINCS-Morph-D-2026-08-30","trigger":"Morph-A and Morph-B fixed GO decisions","dataset":"cpg0004-LINCS","basis":{"type":"PCA","feature_count":241,"features_excluded":["Batch_Number"],"selection_train_rows":len(refs["train"]),"selection_valid_rows":len(refs["valid"]),"candidate_k":list(KS),"selected_k":None,"selection":"minimum stable K within one SE of best validation reconstruction","stability":{"rounds":a.stability_rounds,"metric":"mean squared canonical correlation","lower_quantile":.025,"threshold":STABILITY_THRESHOLD},"test_used_for_selection":False},"test_endpoint_status":reason,"bootstrap":{"rounds":a.bootstrap_rounds,"unit":"compound","paired":True}}
        shutil.copyfile(a.feature_mapping,out/"FEATURE_MAPPING.csv"); (out/"CONFIG.json").write_text(json.dumps(config,indent=2,sort_keys=True),encoding="utf-8"); (out/"PROTOCOL.md").write_text("# Morph-D protocol\n\nPCA basis/K must pass the frozen validation reconstruction plus bootstrap stability rule before any test projection. No K passed, so no test profile was selected, projected, or scored.\n",encoding="utf-8")
        selection.to_csv(out/"K_SELECTION.csv",index=False); st.to_csv(out/"PROGRAM_STABILITY.csv",index=False)
        pd.DataFrame([{"scope":"validation_basis","reason":reason,"detail":"No candidate K simultaneously passed stability q0.025 >= 0.90 and the one-standard-error validation reconstruction rule."}]).to_csv(out/"EXCLUSIONS.csv",index=False)
        pd.DataFrame(columns=["compound","split","reason"]).to_csv(out/"ELIGIBLE_COMPOUNDS.csv",index=False)
        for name,columns in {"METRICS_BY_COMPOUND.csv":["seed","compound","method","coefficient_pcc","coefficient_rmse"],"METRICS_BY_MODULE.csv":["seed","method","program","coefficient_squared_error"],"METRICS_BY_SEED.csv":["seed","method","coefficient_pcc","coefficient_rmse"],"PAIRED_CONTRASTS.csv":["comparison","endpoint","point","ci_low","ci_high"],"BOOTSTRAP_CI.csv":["comparison","endpoint","point","ci_low","ci_high"],"UNSCORABLE_COUNTS.csv":["seed","method","compound_units","unscorable_units"],"PROGRAM_INTERPRETATION.csv":["program","direction","rank","feature_name","loading"]}.items(): pd.DataFrame(columns=columns).to_csv(out/name,index=False)
        pd.DataFrame([{"control":"not_applicable","reason":"No test projection was performed because validation basis selection was blocked."}]).to_csv(out/"NEGATIVE_CONTROLS.csv",index=False)
        decisions=pd.DataFrame([{"comparison":"teacher_minus_raw","decision":"NOT_RUN"},{"comparison":"posterior_minus_teacher","decision":"NOT_RUN"}]); decisions.to_csv(out/"DECISION.csv",index=False); (out/"DECISION.md").write_text("# Morph-D decision\n\n**NOT RUN** — "+reason+". The prescribed basis gate stopped the analysis before selecting or scoring test slot profiles.\n",encoding="utf-8")
        sections=[("Objective","Assess data-driven morphology-program recovery only after a test-blind basis can be frozen."),("Frozen methods","M0/M1/M2 remained frozen; no method was retrained."),("Dataset","Only training/validation five-repeat CP references were selected for the basis gate."),("Feature mapping","241 biological features were eligible; Batch_Number was excluded."),("Eligibility","No test conditions were made eligible because K was not frozen."),("Support/reference isolation","No test support/reference profiles were selected, projected, or scored."),("Primary endpoint","Not evaluated."),("Secondary endpoints","Not evaluated."),("Negative controls","Not applicable without a test endpoint."),("Sample counts",json.dumps({"train_reference_conditions":len(refs["train"]),"valid_reference_conditions":len(refs["valid"])},sort_keys=True)),("Seed-level results","Not evaluated."),("Compound-paired bootstrap CI","Not evaluated."),("GO / SUPPORTIVE / NO-GO","NOT RUN: validation basis gate blocked."),("Biological interpretation","No data-driven-program recovery claim is made."),("What the result does NOT prove","It does not establish absence of recovery; only that the precommitted basis rule did not freeze a K."),("Limitations","The stability/reconstruction requirements were intentionally not relaxed after seeing validation results.")]
        report=["# Results — Morph-D response programs",""]
        for n,(title,body) in enumerate(sections,1): report += [f"## {n}. {title}",body,""]
        (out/"RESULTS.md").write_text("\n".join(report),encoding="utf-8"); (out/"figures"/"summary.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="160"><rect width="100%" height="100%" fill="white"/><text x="20" y="40" font-family="Arial" font-size="20">Morph-D: validation basis gate did not select K</text><text x="20" y="75" font-family="Arial" font-size="14">No test profiles were projected.</text></svg>',encoding="utf-8")
        print(json.dumps({"outdir":str(out),"selected_k":None,"reason":reason,"selection":selection.to_dict("records")},indent=2)); return
    mean=packed[:,0]; w=packed[:,1:]
    errors=[]; a.lambda_by_seed={3407:.10,42:.10,2025:.10};a.beta_by_seed={3407:.10,42:.15,2025:.15}; p0,ge,manifests,_=M.load_pair_and_prediction_bundles(a,a.lambda_by_seed,a.beta_by_seed,errors); slots,exclusions,_,audit=M.build_slots(artifact,p0,ge,manifests,errors); m1,m2=M.trim_to_common_sets(slots)
    if len(m1)!=260 or len(m2)!=258: raise RuntimeError(f"Unexpected frozen sets M1={len(m1)} M2={len(m2)}")
    sample=[]; program=[]; eligible=[]
    for c in sorted(m1):
        eligible.append({"compound":c,"split":"test","m1_eligible":1,"m2_eligible":int(c in m2),"doses":6,"frozen_support_rotations":2})
        for s in SEEDS:
            for r in range(2):
                for d in DOSES:
                    slot=slots[s][c][r][d]; ref=((slot.reference[ix].astype(float)-center)/scale-mean)@w
                    for method in METHODS:
                        if method=="posterior" and c not in m2: continue
                        coef=((slot.profiles[method][ix].astype(float)-center)/scale-mean)@w
                        value=pcc(coef,ref); rmse=float(np.sqrt(np.mean((coef-ref)**2))) if np.isfinite(coef).all() and np.isfinite(ref).all() else math.nan
                        sample.append({"seed":s,"compound":c,"dose":d,"rotation":r,"method":method,"method_label":LABELS[method],"coefficient_pcc":value,"coefficient_rmse":rmse,"scorable":int(np.isfinite(value) and np.isfinite(rmse)),"k":k})
                        for j,(pv,rv) in enumerate(zip(coef,ref),1): program.append({"seed":s,"compound":c,"dose":d,"rotation":r,"method":method,"program":j,"coefficient_error":float(pv-rv),"coefficient_squared_error":float((pv-rv)**2)})
    samples=pd.DataFrame(sample); metrics=comp_metrics(samples); unscore=metrics.assign(unscorable=~np.isfinite(metrics.coefficient_pcc)).groupby(["seed","method"],as_index=False).agg(compound_units=("compound","size"),unscorable_units=("unscorable","sum")); contrasts=contrast(metrics,a.bootstrap_rounds); primary=contrasts[contrasts.endpoint.eq("coefficient_pcc")].copy();primary["decision"]=primary.apply(go,axis=1);decisions=primary[["comparison","seed3407","seed42","seed2025","n_compounds","point","ci_low","ci_high","decision"]]
    interp=[]
    for j in range(k):
        load=fmap.assign(loading=w[:,j]);
        for direction,part in (("positive",load.nlargest(10,"loading")),("negative",load.nsmallest(10,"loading"))):
            for rank,row in enumerate(part.itertuples(index=False),1): interp.append({"program":j+1,"direction":direction,"rank":rank,"feature_index":row.feature_index,"feature_name":row.feature_name,"loading":row.loading,"biological_module":row.biological_module,"object":row.object,"channel":row.channel,"measurement_family":row.measurement_family})
    config={"version":"cpg0004-LINCS-Morph-D-2026-08-30","trigger":"Morph-A and Morph-B fixed GO decisions","dataset":"cpg0004-LINCS","basis":{"type":"PCA","feature_count":241,"features_excluded":["Batch_Number"],"training_reference":"exactly five physical repeat condition mean","selection_train_rows":len(refs["train"]),"selection_valid_rows":len(refs["valid"]),"candidate_k":list(KS),"selected_k":k,"selection":"minimum stable K within one SE of best validation reconstruction","stability":{"rounds":a.stability_rounds,"metric":"mean squared canonical correlation","lower_quantile":.025,"threshold":STABILITY_THRESHOLD},"test_used_for_selection":False,"final_refit":"train plus validation high-quality references"},"test":{"seeds":list(SEEDS),"doses":list(DOSES),"m1_compounds":len(m1),"m2_compounds":len(m2),"support_reference":"two frozen support rotations; reference is mean of other four physical rows"},"bootstrap":{"rounds":a.bootstrap_rounds,"unit":"compound","paired":True,"ci":"95% percentile"},"unscorable":"no epsilon or imputation","input_hashes":[{"path":str(p.resolve()),"sha256":sha256(p)} for p in [a.data,a.feature_mapping]],"slot_audit":audit,"mapping_error_count":len(errors)}
    shutil.copyfile(a.feature_mapping,out/"FEATURE_MAPPING.csv");(out/"CONFIG.json").write_text(json.dumps(config,indent=2,sort_keys=True,default=str),encoding="utf-8");(out/"PROTOCOL.md").write_text("# Morph-D protocol\n\nPCA basis and K use train/validation high-quality references only. Test profiles are projected only after the selected basis is frozen. The fixed K rule is minimum stable K within one validation reconstruction SE; stability threshold is lower 2.5% bootstrap subspace agreement >= 0.90.\n",encoding="utf-8")
    pd.DataFrame(eligible).to_csv(out/"ELIGIBLE_COMPOUNDS.csv",index=False);pd.DataFrame(exclusions+[{"scope":"audit","reason":x} for x in errors]).to_csv(out/"EXCLUSIONS.csv",index=False);metrics.to_csv(out/"METRICS_BY_COMPOUND.csv",index=False);metrics.groupby(["seed","method"],as_index=False)[["coefficient_pcc","coefficient_rmse"]].mean().to_csv(out/"METRICS_BY_SEED.csv",index=False);pd.DataFrame(program).groupby(["seed","method","program"],as_index=False).coefficient_squared_error.mean().to_csv(out/"METRICS_BY_MODULE.csv",index=False);contrasts.to_csv(out/"PAIRED_CONTRASTS.csv",index=False);contrasts.to_csv(out/"BOOTSTRAP_CI.csv",index=False);unscore.to_csv(out/"UNSCORABLE_COUNTS.csv",index=False);selection.to_csv(out/"K_SELECTION.csv",index=False);st.to_csv(out/"PROGRAM_STABILITY.csv",index=False);pd.DataFrame(interp).to_csv(out/"PROGRAM_INTERPRETATION.csv",index=False);pd.DataFrame([{"control":"not_applicable","reason":"No Morph-D negative-control analysis was preregistered."}]).to_csv(out/"NEGATIVE_CONTROLS.csv",index=False);decisions.to_csv(out/"DECISION.csv",index=False);(out/"DECISION.md").write_text("# Morph-D decision\n\n```csv\n"+decisions.to_csv(index=False)+"```\n",encoding="utf-8")
    report=["# Results — Morph-D response programs","","## 1. Objective","Evaluate whether frozen methods recover independent-reference coefficients in a data-driven morphology program basis.","","## 2. Frozen methods","M0 raw, M1 frozen teacher, M2 frozen GE posterior; no model update.","","## 3. Dataset","cpg0004-LINCS Cell Painting with PCA basis fit without test profiles.","","## 4. Feature mapping","241 biological features; Batch_Number excluded.","","## 5. Eligibility",f"M1={len(m1)} and M2={len(m2)} test compounds; each has six doses and two frozen supports.","","## 6. Support/reference isolation","Reference is the four-repeat mean excluding each support.","","## 7. Primary endpoint","PCC between K-dimensional predicted and independent-reference coefficient vectors, averaged per compound.","","## 8. Secondary endpoints","Coefficient RMSE and per-program squared coefficient error.","","## 9. Negative controls","No Morph-D negative control was preregistered; file records not applicable.","","## 10. Sample counts",json.dumps(config["test"],sort_keys=True),"","## 11. Seed-level results","METRICS_BY_SEED.csv.","","## 12. Compound-paired bootstrap CI",primary.to_csv(index=False),"","## 13. GO / SUPPORTIVE / NO-GO",decisions.to_csv(index=False),"","## 14. Biological interpretation","A GO supports recovery in a test-blind data-driven morphology basis, not a signaling-pathway label.","","## 15. What the result does NOT prove","It does not identify mechanism or replace physical repeats.","","## 16. Limitations","PCA programs are basis-dependent; only two recorded support rotations are available; unscorable PCC is not repaired. K selection and stability are in K_SELECTION.csv and PROGRAM_STABILITY.csv."]
    (out/"RESULTS.md").write_text("\n".join(report)+"\n",encoding="utf-8");(out/"figures"/"summary.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="240"><rect width="100%" height="100%" fill="white"/><text x="20" y="35" font-family="Arial" font-size="20">Morph-D frozen program recovery</text><text x="20" y="65" font-family="monospace" font-size="12">'+decisions.to_csv(index=False).replace("&","&amp;").replace("<","&lt;")+'</text></svg>',encoding="utf-8")
    print(json.dumps({"outdir":str(out),"selected_k":k,"selection":selection.to_dict("records"),"decisions":decisions.to_dict("records"),"errors":errors},indent=2,default=str))


if __name__=="__main__": main()
