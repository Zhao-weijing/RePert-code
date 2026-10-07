"""Derive mean-profile RMSE and paired 95% CIs from frozen outputs only.

BBBC047: RMSE per compound/seed -> mean across seeds -> compound mean.
sci-Plex3: sqrt(condition MSE of frozen prediction ensemble) -> dose mean
within drug -> drug mean within line -> equal-weight three-line macro.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"figures/.python_libs"))
import numpy as np
import pandas as pd

SC=ROOT/"analysis/predictor_supervision/sci_plex3/results/confirmation"
OUT=ROOT/"tables/predictor/inputs/rmse"
BB_METHODS=("M0_RawKAggregate","M2_CFRA1ViewAggregate_Residual")
SC_METHODS=("M0_RawAggregate","M1_CFRA1Aggregate")
LINES=("A549","K562","MCF7")
SURFACES=("cross_cell_unseen","cross_cell_seen")
DRAWS=10000


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def same(a,b,label,tol=2e-10):
    if not np.allclose(a,b,rtol=0,atol=tol):
        raise AssertionError(f"{label}: {a} != {b}")


def ci(x):
    return np.quantile(x,[.025,.975],method="linear")


def export(frame,name):
    frame.to_csv(OUT/name,index=False,float_format="%.17g")


def summary(dataset,setting,line,n,values,boots,seed):
    raw=values[:,0];candidate=values[:,1];diff=candidate-raw
    lo0,hi0=ci(boots[:,0]);lo1,hi1=ci(boots[:,1]);lod,hid=ci(boots[:,1]-boots[:,0])
    return dict(dataset=dataset,setting=setting,target_line=line,n_compounds=n,
        metric="raw_target_rmse",comparator_method="CFRA+Res" if dataset=="BBBC047" else "CFRA",
        raw_mean=raw.mean(),raw_ci_low=lo0,raw_ci_high=hi0,
        comparator_mean=candidate.mean(),comparator_ci_low=lo1,comparator_ci_high=hi1,
        difference=diff.mean(),difference_ci_low=lod,difference_ci_high=hid,
        bootstrap_draws=DRAWS,bootstrap_seed=int(seed),
        ci_unit="compound" if dataset=="BBBC047" else "drug_within_target_line",
        ci_status="pointwise_95_percent",metric_status="posthoc_frozen_prediction_supplement")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=[ROOT/"tables/predictor/inputs/rmse_inputs/BBBC_RMSE_PER_COMPOUND_SEED.csv",
            ROOT/"tables/predictor/inputs/rmse_inputs/BBBC_RMSE_EXTRACTION_AUDIT.json",
            SC/"CONFIRMATION_PER_CONDITION.csv",SC/"CONFIRMATION_PER_COMPOUND.csv",
            SC/"SURFACE_MACRO_FULL_COMPARISON.csv"]
    remote=json.loads(inputs[1].read_text())
    assert remote["status"]=="PASS" and remote["frozen_pcc_reproduced"]
    assert sha(inputs[0])==remote["output"]["sha256"]
    records=[];seeds={}
    bb=pd.read_csv(inputs[0])
    assert set(bb.method)==set(BB_METHODS)
    assert len(bb)==116460
    assert not bb.duplicated(["budget","method","seed","compound_id"]).any()
    count=bb.groupby(["budget","method","compound_id"]).seed.nunique()
    assert (count==5).all()
    assert np.isfinite(bb[["raw_target_mse","raw_target_rmse"]]).all().all()
    same(bb.raw_target_rmse**2,bb.raw_target_mse,"BB RMSE/MSE identity")
    grouped=bb.groupby(["budget","compound_id","method"],sort=True)[["raw_target_mse","raw_target_rmse"]].mean().reset_index()
    export(grouped,"BBBC_RMSE_PER_COMPOUND.csv")
    for budget,n in ((2,4040),(3,4000),(4,3606)):
        wide=grouped[grouped.budget==budget].pivot(index="compound_id",columns="method",values="raw_target_rmse").sort_index()
        assert len(wide)==n and not wide.isna().any().any()
        vals=wide[list(BB_METHODS)].to_numpy()
        seed=int.from_bytes(hashlib.sha256(f"BBBC047-rmse-compound-v1|b{budget}|draws10000".encode()).digest()[:4],"big")
        rng=np.random.default_rng(seed);boots=[]
        # Shared paired indices for both arms, bounded temporary memory.
        for start in range(0,DRAWS,250):
            ix=rng.integers(0,n,size=(min(250,DRAWS-start),n),dtype=np.int32)
            boots.append(vals[ix].mean(axis=1))
        boot=np.concatenate(boots)
        records.append(summary("BBBC047",f"{budget}R","",n,vals,boot,seed))
        seeds[f"BBBC047_{budget}R"]=seed

    cond=pd.read_csv(inputs[2]);compound=pd.read_csv(inputs[3])
    macro=pd.read_csv(inputs[4]).set_index(["surface","metric"])
    assert set(macro.input_sha256)=={sha(inputs[3])}
    cond=cond[cond.surface.isin(SURFACES)&cond.method.isin(SC_METHODS)].copy()
    compound=compound[compound.surface.isin(SURFACES)&compound.method.isin(SC_METHODS)].copy()
    keys=["surface","target_cell_line","method","drug"]
    assert not cond.duplicated(keys+["dose_nM"]).any()
    assert not compound.duplicated(keys).any()
    assert np.isfinite(cond.raw_target_mse).all() and (cond.raw_target_mse>=0).all()
    cond["raw_target_rmse"]=np.sqrt(cond.raw_target_mse)
    values=cond.groupby(keys,sort=True)[["raw_target_mse","raw_target_rmse","delta_pcc","full_target_pcc"]].mean()
    old=compound.set_index(keys).sort_index()
    assert values.index.equals(old.index)
    for metric in ("raw_target_mse","delta_pcc","full_target_pcc"):
        same(values[metric],old[metric],f"dose aggregation reproduces {metric}")
    same(cond.groupby(keys).size().sort_index(),old.n_doses,"dose counts")
    export(cond[keys+["dose_nM","raw_target_mse","raw_target_rmse"]],"SCIPLEX3_RMSE_PER_CONDITION.csv")
    export(values.reset_index(),"SCIPLEX3_RMSE_PER_DRUG.csv")
    for surface in SURFACES:
        setting="Unseen" if surface.endswith("unseen") else "Seen"
        n=36 if setting=="Unseen" else 183
        seed=int(macro.loc[(surface,"raw_target_mse"),"bootstrap_seed"])
        rng=np.random.default_rng(seed)
        indices={line:rng.integers(0,n,size=(DRAWS,n),dtype=np.int32) for line in LINES}
        line_vals=[];line_boot=[];mse_vals=[];mse_boot=[]
        for line in LINES:
            select=values.xs((surface,line),level=("surface","target_cell_line"))
            wide=select.raw_target_rmse.unstack("method").sort_index()
            mse=select.raw_target_mse.unstack("method").sort_index()
            assert len(wide)==n and wide.index.equals(mse.index) and not wide.isna().any().any()
            a=wide[list(SC_METHODS)].to_numpy();b=mse[list(SC_METHODS)].to_numpy()
            boot=a[indices[line]].mean(axis=1);bm=b[indices[line]].mean(axis=1)
            records.append(summary("sci-Plex3",setting,line,n,a,boot,seed))
            line_vals.append(a.mean(axis=0));line_boot.append(boot)
            mse_vals.append(b.mean(axis=0));mse_boot.append(bm)
        m=macro.loc[(surface,"raw_target_mse")]
        old_mean=np.mean(mse_vals,axis=0);old_boot=np.mean(mse_boot,axis=0)
        same(old_mean,[m[method+"_value"] for method in SC_METHODS],"frozen MSE macro means")
        same(ci(old_boot[:,1]-old_boot[:,0]),[m.M1_minus_M0_ci_low,m.M1_minus_M0_ci_high],"frozen MSE paired CI replay")
        records.append(summary("sci-Plex3",setting,"equal_line_macro",n,np.array(line_vals),np.mean(line_boot,axis=0),seed))
        seeds["sciPlex3_"+setting]=seed
    result=pd.DataFrame(records)
    assert len(result)==11 and np.isfinite(result.select_dtypes(include="number")).all().all()
    same(result.comparator_mean-result.raw_mean,result.difference,"summary paired identity")
    export(result,"RMSE_SUMMARY.csv")
    audit=dict(status="PASS",metric="mean_of_profile_RMSE",bootstrap_draws=DRAWS,
        bootstrap_ci="pointwise percentile 2.5/97.5; shared paired draws for arms and differences",
        bbbc_seed_aggregation="per-profile RMSE then five-seed mean then compound mean",
        sciplex_aggregation="sqrt frozen ensemble condition MSE then dose mean within drug then drug mean within line then equal-line macro",
        sciplex_original_mse_macro_means_and_ci_reproduced=True,
        no_sqrt_of_aggregate_mse=True,no_inference_or_training=True,no_primary_endpoint_change=True,
        source_scope="new supplementary metric, derived only from previously frozen outputs",
        formula="sqrt(mean_features((prediction-raw_target)**2)); no added normalization",
        baseline_cancellation="adding the same control baseline to prediction and target leaves RMSE unchanged",
        across_dataset_scale_comparison_allowed=False,bootstrap_seeds=seeds,
        numpy=np.__version__,pandas=pd.__version__,
        inputs=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in inputs],
        summary_sha256=sha(OUT/"RMSE_SUMMARY.csv"))
    (OUT/"RMSE_AUDIT.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(result[["dataset","setting","target_line","raw_mean","comparator_mean","difference","difference_ci_low","difference_ci_high"]].to_string(index=False))


if __name__=="__main__":
    main()
