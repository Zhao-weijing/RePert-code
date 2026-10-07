"""Read-only frozen-prediction RMSE extraction, executed on server 240.

Only reads the enumerated prediction NPZs and frozen reporting/audit files.
Does not import training code, load a model, open new repeats, or run inference.
Derived outputs must be outside the frozen run and must not already exist.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

METHODS = ("M0_RawKAggregate", "M2_CFRA1ViewAggregate_Residual")
SEEDS = (3407,42,2025,1337,7331)
COUNTS = {2:4040,3:4000,4:3606}
VERSION = "BBBC047-strict-selected-repeat-1R-CFRA-M2-v2-2026-09-16"
PER_HASH = "30386d9fab5a754269be32779d65c9572e2bea3fcb8d69f0287136569febbbe2"
MANIFEST_HASH = "30321438a735d437699054d942c0ded39866a9670f497568d5cdbcbcbf8227d0"


def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    h=hashlib.sha256()
    h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    root=args.run.resolve();out=args.out.resolve()
    assert root not in (out,*out.parents), "output must not modify frozen run"
    out.mkdir(parents=True,exist_ok=True)
    paths=[root/"TEST_PER_COMPOUND.csv",root/"STRICT_SELECTED_MANIFEST.csv",
           root/"SELECTION_FREEZE.json",root/"TEST_COMPLETE.json"]
    assert sha(paths[0])==PER_HASH
    assert sha(paths[1])==MANIFEST_HASH
    freeze=json.loads(paths[2].read_text())
    complete=json.loads(paths[3].read_text())
    assert freeze["selection_frozen"] and not freeze["test_loaded"]
    assert freeze["manifest_sha256"]==MANIFEST_HASH
    assert complete["status"]=="PASS" and complete["test_loaded_after_authorization"]
    assert not complete["test_used_for_selection"] and complete["version"]==VERSION
    frozen={}
    with paths[0].open(newline="") as f:
        for row in csv.DictReader(f):
            k=(int(row["budget"]),row["method"],int(row["seed"]),row["compound_id"])
            assert k not in frozen
            frozen[k]=float(row["delta_pcc"])
    provenance=[dict(path=str(p),sha256=sha(p)) for p in paths]
    records=[];npz_audits=[];max_error=0.0
    for budget,n in COUNTS.items():
        ids_ref=None;plates_ref=None;target_hash_ref=None
        for method in METHODS:
            for seed in SEEDS:
                p=root/f"budget{budget}"/f"seed{seed}_{method}_test_predictions.npz"
                before=sha(p)
                with np.load(p,allow_pickle=False) as z:
                    assert set(z.files)=={"compound","prediction","raw_k_aggregate","selected_plates"}
                    ids=z["compound"].astype(str)
                    plates=z["selected_plates"].astype(str)
                    target=z["raw_k_aggregate"]
                    prediction=z["prediction"]
                assert target.shape==prediction.shape==(n,775)
                assert ids.shape==(n,) and len(set(ids))==n
                assert plates.shape==(n,budget)
                assert all(len(set(row))==budget for row in plates.tolist())
                assert np.isfinite(target).all() and np.isfinite(prediction).all()
                th=array_sha(target)
                if ids_ref is None:
                    ids_ref=ids.copy();plates_ref=plates.copy();target_hash_ref=th
                else:
                    assert np.array_equal(ids,ids_ref)
                    assert np.array_equal(plates,plates_ref)
                    assert th==target_hash_ref, "raw targets differ between methods/seeds"
                a=prediction.astype(np.float64);b=target.astype(np.float64)
                mse=np.mean((a-b)**2,axis=1,dtype=np.float64)
                rmse=np.sqrt(mse)
                ac=a-a.mean(axis=1,keepdims=True);bc=b-b.mean(axis=1,keepdims=True)
                denom=np.sqrt((ac*ac).sum(axis=1)*(bc*bc).sum(axis=1))
                assert (denom>0).all()
                corr=(ac*bc).sum(axis=1)/denom
                old=np.array([frozen[(budget,method,seed,x)] for x in ids])
                error=float(np.max(np.abs(corr-old)))
                assert error<2e-7, f"frozen PCC mismatch: {error}"
                max_error=max(max_error,error)
                for i,compound in enumerate(ids):
                    records.append(dict(budget=budget,method=method,seed=seed,compound_id=compound,
                        selected_plates="|".join(plates[i]),n_features=775,
                        raw_target_mse=float(mse[i]),raw_target_rmse=float(rmse[i])))
                assert sha(p)==before, "prediction file changed during extraction"
                npz_audits.append(dict(path=str(p),sha256=before,n_compounds=n,
                    shape=list(target.shape),selected_plate_count=budget,
                    target_array_sha256=th,max_frozen_pcc_error=error))
        print(f"budget {budget}: verified {n} compounds, both methods and all 5 seeds",flush=True)
    assert len(records)==sum(COUNTS.values())*len(METHODS)*len(SEEDS)==len(frozen)
    for item in provenance:
        assert sha(Path(item["path"]))==item["sha256"]
    csv_path=out/"BBBC_RMSE_PER_COMPOUND_SEED.csv"
    with csv_path.open("x",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    audit=dict(status="PASS",version="frozen-prediction-rmse-report-v1-20260922",
        run=str(root),no_training_or_inference=True,no_checkpoint_reads=True,
        no_new_physical_repeat_reads=True,new_profile_dataset_reads=False,
        only_saved_predictions_and_their_matching_raw_targets_read=True,
        compound_ids_unique=True,selected_plate_counts_verified=True,
        raw_targets_identical_across_methods_and_seeds=True,
        frozen_pcc_reproduced=True,max_frozen_pcc_error=max_error,
        manifest_scope="original train/valid manifest; test selected plates verified from frozen prediction NPZs",
        row_count=len(records),npz_file_count=len(npz_audits),numpy=np.__version__,
        formula="sqrt(mean_features((float64(prediction)-float64(raw_k_aggregate))**2))",
        scope="supplementary post-hoc reconstruction metric on frozen predictions",
        input_files=provenance,prediction_files=npz_audits,
        output=dict(path=str(csv_path),sha256=sha(csv_path)))
    with (out/"BBBC_RMSE_EXTRACTION_AUDIT.json").open("x") as f:
        json.dump(audit,f,indent=2)
    print(json.dumps({k:v for k,v in audit.items() if k not in ("input_files","prediction_files")},indent=2),flush=True)


if __name__=="__main__":
    main()
