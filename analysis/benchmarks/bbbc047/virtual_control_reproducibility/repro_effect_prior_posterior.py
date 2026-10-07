#!/usr/bin/env python3
"""CP-only MVP: virtual prior plus repeat evidence yields a posterior held-plate estimate."""
from __future__ import annotations

import argparse, json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn

import repro_effect_virtual_cp_mvp as C

VERSION = "BBBC047-Prior-Evidence-Posterior-CP-MVP-2026-08-29"


@dataclass
class EvidencePairs:
    support: np.ndarray
    dispersion: np.ndarray
    target: np.ndarray
    smiles: list[str]
    held_rows: np.ndarray


@dataclass
class TeacherState:
    model: Any
    mean: np.ndarray
    scale: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cp-rows-root", type=Path, default=Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0"))
    p.add_argument("--aggregate-npz", type=Path, default=Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0/molecule_aggregates.npz"))
    p.add_argument("--model-h5", type=Path, default=Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/Paired_CP_GE_PlateMedian_v1_model_compat.h5"))
    p.add_argument("--split-lock", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/source/baseline/artifacts/split_locks/BBBC047_smiles_split_seed3407_official_v1.json"))
    p.add_argument("--outdir", type=Path); p.add_argument("--aggregate-root", type=Path); p.add_argument("--seed", type=int, choices=C.SEEDS)
    p.add_argument("--teacher-folds", type=int, default=5); p.add_argument("--teacher-epochs", type=int, default=80); p.add_argument("--student-epochs", type=int, default=40); p.add_argument("--gate-epochs", type=int, default=40); p.add_argument("--batch-size", type=int, default=256); p.add_argument("--learning-rate", type=float, default=3e-4); p.add_argument("--weight-decay", type=float, default=1e-5); p.add_argument("--null-rounds", type=int, default=32); p.add_argument("--bootstrap-rounds", type=int, default=10000); p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto"); p.add_argument("--dry-run", action="store_true"); p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.aggregate_root:
        if a.seed is not None or a.outdir is not None or a.dry_run or a.smoke: p.error("--aggregate-root cannot be combined with run options")
        return a
    if a.seed is None and not a.dry_run: p.error("--seed is required unless --dry-run or --aggregate-root is used")
    if not a.dry_run and a.outdir is None: p.error("--outdir is required for a run")
    if (a.teacher_folds, a.teacher_epochs, a.student_epochs, a.gate_epochs, a.batch_size) != (5, 80, 40, 40, 256): p.error("MVP locks folds=5, teacher_epochs=80, student_epochs=40, gate_epochs=40, batch_size=256")
    return a


def evidence_pairs(rows: Any) -> EvidencePairs:
    support=[]; dispersion=[]; target=[]; smiles=[]; held_rows=[]
    for value in sorted(rows.groups):
        group=rows.groups[value]
        if group.size < 3: continue
        values=rows.delta[group]
        for local, held in enumerate(group.tolist()):
            others=np.delete(values, local, axis=0)
            support.append(others.mean(axis=0, dtype=np.float64).astype(np.float32))
            dispersion.append(others.std(axis=0, dtype=np.float64).astype(np.float32))
            target.append(values[local]); smiles.append(value); held_rows.append(held)
    if not support: raise RuntimeError("No P>=3 evidence pairs")
    return EvidencePairs(np.vstack(support), np.vstack(dispersion), np.vstack(target), smiles, np.asarray(held_rows, dtype=np.int64))


def restrict_pairs(pairs: EvidencePairs, compounds: set[str]) -> EvidencePairs:
    ix=np.asarray([i for i,s in enumerate(pairs.smiles) if s in compounds], dtype=np.int64)
    if not ix.size: raise RuntimeError("Empty compound subset")
    return EvidencePairs(pairs.support[ix], pairs.dispersion[ix], pairs.target[ix], [pairs.smiles[i] for i in ix.tolist()], pairs.held_rows[ix])


def as_loo(pairs: EvidencePairs) -> Any:
    return C.BASE.LooPairs(pairs.support, pairs.target, pairs.smiles, pairs.held_rows)


def subset_virtual(data: C.VirtualSplit, compounds: set[str]) -> C.VirtualSplit:
    ix=np.asarray([i for i,s in enumerate(data.smiles) if s in compounds], dtype=np.int64)
    if not ix.size: raise RuntimeError("Empty virtual subset")
    return C.VirtualSplit([data.smiles[i] for i in ix.tolist()], data.control[ix], data.delta[ix], data.fingerprint[ix])


def inner_teacher_split(compounds: set[str], seed: int, label: str) -> tuple[set[str], set[str]]:
    valid={s for s in compounds if C.BASE.stable_seed(seed, f"{label}|{s}") % 10 == 0}
    if len(valid) < 100: raise RuntimeError("Teacher internal validation subset too small")
    return compounds-valid, valid


def fit_oof_teachers(train: EvidencePairs, virtual_train: C.VirtualSplit, args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, list[TeacherState], dict[str, Any]]:
    eligible=set(train.smiles); assignment=C.fold_map(virtual_train.smiles, args.seed, args.teacher_folds); output=np.empty_like(train.target); seen=np.zeros(len(train.smiles), dtype=bool); states=[]; meta=[]
    for fold in range(args.teacher_folds):
        held={s for s,v in assignment.items() if v==fold and s in eligible}; candidates=eligible-held; fitted, checkpoint=inner_teacher_split(candidates, args.seed, f"posterior-teacher-fold{fold}")
        seed=C.BASE.stable_seed(args.seed, f"posterior-teacher-model{fold}") % (2**32-1)
        ta=SimpleNamespace(seed=seed, hidden_dim=256, latent_dim=32, batch_size=args.batch_size, learning_rate=args.learning_rate, weight_decay=args.weight_decay, epochs=args.teacher_epochs, smoke=args.smoke)
        model, mean, scale, epoch, val_mse=C.BASE.fit_encoder(as_loo(restrict_pairs(train,fitted)), as_loo(restrict_pairs(train,checkpoint)), ta, device)
        held_pairs=restrict_pairs(train,held); pred=C.effect_array(model, as_loo(held_pairs), mean, scale, args.batch_size, device)
        locations=np.asarray([i for i,s in enumerate(train.smiles) if s in held],dtype=np.int64)
        output[locations]=pred; seen[locations]=True; states.append(TeacherState(model,mean,scale)); meta.append({"fold":fold,"fit_compounds":len(fitted),"checkpoint_compounds":len(checkpoint),"held_compounds":len(held),"best_epoch":epoch,"checkpoint_mse":val_mse})
    if not seen.all(): raise RuntimeError("Missing OOF teacher prediction")
    return output, states, {"definition":"Each OOF teacher fits and checkpoints only on other official-training compounds; it never sees the compound or held CP plate that it predicts.","folds":meta,"eligible_p_ge_3":len(eligible)}


def teacher_ensemble(states: list[TeacherState], pairs: EvidencePairs, batch_size: int, device: torch.device) -> np.ndarray:
    values=[C.effect_array(state.model, as_loo(pairs), state.mean, state.scale, batch_size, device) for state in states]
    return np.mean(np.stack(values),axis=0).astype(np.float32)


def fit_internal_prior(virtual_train: C.VirtualSplit, compounds: set[str], args: argparse.Namespace,
                       device: torch.device, label: str) -> tuple[Any, dict[str, np.ndarray], dict[str, Any]]:
    """Fit/checkpoint a C0 prior using only an internal official-train split."""
    fitted, checkpoint = inner_teacher_split(compounds, args.seed, label)
    fit_data, checkpoint_data = subset_virtual(virtual_train, fitted), subset_virtual(virtual_train, checkpoint)
    model, stats, epoch, value = C.fit_student(fit_data, checkpoint_data, fit_data.delta, args, device, label)
    return model, stats, {
        "fit_compounds": len(fitted), "checkpoint_compounds": len(checkpoint),
        "checkpoint_source": "official_train_internal",
        "best_epoch": epoch, "checkpoint_mse": value,
    }


def oof_prior(virtual_train: C.VirtualSplit, args: argparse.Namespace, device: torch.device) -> tuple[dict[str,np.ndarray], dict[str,Any]]:
    """OOF C0 prior with every checkpoint set strictly inside official training."""
    assignment=C.fold_map(virtual_train.smiles,args.seed,args.teacher_folds); output={}; meta=[]
    for fold in range(args.teacher_folds):
        held={s for s,v in assignment.items() if v==fold}; candidates=set(virtual_train.smiles)-held
        model, stats, details = fit_internal_prior(virtual_train, candidates, args, device, f"prior-oof-fold{fold}")
        pred=C.predict_student(model,subset_virtual(virtual_train,held),stats,args.batch_size,device)
        output.update(pred); meta.append({"fold":fold,"held_compounds":len(held),**details})
    if set(output)!=set(virtual_train.smiles): raise RuntimeError("Incomplete OOF virtual prior")
    return output,{"definition":"C0 prior is OOF for every posterior-training compound; all C0 checkpoints are selected within official training.","folds":meta}


class EvidenceGate(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.net=nn.Sequential(nn.Linear(C.CP_DIM*2,128),nn.GELU(),nn.Linear(128,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,x:torch.Tensor)->torch.Tensor: return torch.sigmoid(self.net(x))


def gate_features(pairs: EvidencePairs, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.hstack((pairs.support,pairs.dispersion))-mean)/scale


def fit_gate(train: EvidencePairs, teacher: np.ndarray, prior: dict[str,np.ndarray], valid: EvidencePairs, valid_teacher: np.ndarray, valid_prior: dict[str,np.ndarray], args: argparse.Namespace, device: torch.device) -> tuple[EvidenceGate,dict[str,np.ndarray],int,float]:
    C.set_seed(args.seed); feature=np.hstack((train.support,train.dispersion)); fmean=feature.mean(axis=0,dtype=np.float64).astype(np.float32); fscale=np.maximum(feature.std(axis=0,dtype=np.float64).astype(np.float32),1e-6); tmean=train.target.mean(axis=0,dtype=np.float64).astype(np.float32); tscale=np.maximum(train.target.std(axis=0,dtype=np.float64).astype(np.float32),1e-6)
    x=gate_features(train,fmean,fscale); prior_train=np.vstack([prior[s] for s in train.smiles]); y_t=(teacher-tmean)/tscale; y_p=(prior_train-tmean)/tscale; y=(train.target-tmean)/tscale
    vx=gate_features(valid,fmean,fscale); vp=np.vstack([valid_prior[s] for s in valid.smiles]); vt=(valid_teacher-tmean)/tscale; vy=(valid.target-tmean)/tscale
    model=EvidenceGate().to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay); gen=torch.Generator(); gen.manual_seed(args.seed); loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(x),torch.from_numpy(y_t),torch.from_numpy(y_p),torch.from_numpy(y)),batch_size=args.batch_size,shuffle=True,num_workers=0,generator=gen); best=None; best_epoch=-1; best_value=float("inf")
    for epoch in range(1 if args.smoke else args.gate_epochs):
        model.train()
        for evidence, teacher_value, prior_value, target in loader:
            opt.zero_grad(set_to_none=True); lam=model(evidence.to(device)); posterior=(1-lam)*teacher_value.to(device)+lam*prior_value.to(device); loss=torch.mean((posterior-target.to(device))**2)
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite posterior-gate loss")
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lam=model(torch.from_numpy(vx).to(device)); posterior=(1-lam)*torch.from_numpy(vt).to(device)+lam*torch.from_numpy(vp).to(device); value=float(torch.mean((posterior-torch.from_numpy(vy).to(device))**2).cpu())
        if value<best_value: best_value,best_epoch=value,epoch; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch%10==0: print(f"[posterior gate seed {args.seed} epoch {epoch}] validation_mse={value:.7f}",flush=True)
    assert best is not None; model.load_state_dict(best)
    return model,{"feature_mean":fmean,"feature_scale":fscale,"target_mean":tmean,"target_scale":tscale},best_epoch,best_value


def pair_value_subset(pairs: EvidencePairs, values: np.ndarray, compounds: set[str]) -> np.ndarray:
    indices=np.asarray([i for i,s in enumerate(pairs.smiles) if s in compounds],dtype=np.int64)
    if not indices.size: raise RuntimeError("Empty gate pair subset")
    return values[indices]


def fit_gate_internal(train: EvidencePairs, teacher: np.ndarray, prior: dict[str, np.ndarray],
                      args: argparse.Namespace, device: torch.device, label: str) -> tuple[EvidenceGate,dict[str,np.ndarray],dict[str,Any]]:
    """Fit/checkpoint the evidence gate on disjoint official-training compounds."""
    fitted, checkpoint=inner_teacher_split(set(train.smiles),args.seed,label)
    model, stats, epoch, value=fit_gate(
        restrict_pairs(train,fitted),pair_value_subset(train,teacher,fitted),prior,
        restrict_pairs(train,checkpoint),pair_value_subset(train,teacher,checkpoint),prior,
        args,device,
    )
    return model,stats,{"fit_compounds":len(fitted),"checkpoint_compounds":len(checkpoint),"checkpoint_source":"official_train_internal","best_epoch":epoch,"checkpoint_mse":value}


def gated_prediction(gate: EvidenceGate,pairs: EvidencePairs,teacher: np.ndarray,prior:dict[str,np.ndarray],stats:dict[str,np.ndarray],device:torch.device) -> tuple[dict[int,np.ndarray],np.ndarray]:
    x=gate_features(pairs,stats["feature_mean"],stats["feature_scale"])
    gate.eval()
    with torch.no_grad(): lam=gate(torch.from_numpy(x).to(device)).cpu().numpy().reshape(-1)
    p=np.vstack([prior[s] for s in pairs.smiles]); posterior=(1-lam[:,None])*teacher+lam[:,None]*p
    return {int(h):posterior[i] for i,h in enumerate(pairs.held_rows.tolist())},lam


def methods(pairs: EvidencePairs, teacher:np.ndarray,prior:dict[str,np.ndarray],gate:EvidenceGate,gate_stats:dict[str,np.ndarray],device:torch.device) -> tuple[dict[str,dict[int,np.ndarray]],np.ndarray]:
    posterior,lam=gated_prediction(gate,pairs,teacher,prior,gate_stats,device)
    return {"support_mean":{int(h):pairs.support[i] for i,h in enumerate(pairs.held_rows.tolist())},"a_teacher":{int(h):teacher[i] for i,h in enumerate(pairs.held_rows.tolist())},"c0_prior":{int(h):prior[s] for h,s in zip(pairs.held_rows.tolist(),pairs.smiles)},"posterior_gate":posterior},lam


def run(args:argparse.Namespace)->None:
    if not args.dry_run and args.outdir.exists(): raise FileExistsError(f"Refusing to overwrite {args.outdir}")
    cp=C.BASE.load_splits(args.cp_rows_root,args.split_lock); virtual=C.load_virtual(args.model_h5,args.aggregate_npz,args.split_lock)
    if args.dry_run:
        pairs={k:evidence_pairs(v) for k,v in cp.items()}; print(json.dumps({"version":VERSION,"split_counts":{k:len(v.smiles) for k,v in virtual.items()},"pair_counts":{k:len(v.smiles) for k,v in pairs.items()},"guards":["CP only","prior has no post-treatment input","gate sees support mean plus dispersion only","teacher and C0 prior are OOF for posterior training","test held plate is excluded from evidence"]},indent=2,sort_keys=True)); return
    args.outdir.mkdir(parents=True); device=C.select_device(args.device); train,valid,test=(evidence_pairs(cp[k]) for k in ("train","valid","test"))
    teacher_train,teacher_states,teacher_meta=fit_oof_teachers(train,virtual["train"],args,device); teacher_valid=teacher_ensemble(teacher_states,valid,args.batch_size,device); teacher_test=teacher_ensemble(teacher_states,test,args.batch_size,device)
    prior_train,prior_meta=oof_prior(virtual["train"],args,device)
    prior_model,prior_stats,prior_full_meta=fit_internal_prior(virtual["train"],set(virtual["train"].smiles),args,device,"c0-full-prior")
    prior_valid=C.predict_student(prior_model,virtual["valid"],prior_stats,args.batch_size,device); prior_test=C.predict_student(prior_model,virtual["test"],prior_stats,args.batch_size,device)
    gate,gate_stats,gate_meta=fit_gate_internal(train,teacher_train,prior_train,args,device,"posterior-gate")
    valid_methods,valid_lambda=methods(valid,teacher_valid,prior_valid,gate,gate_stats,device); test_methods,test_lambda=methods(test,teacher_test,prior_test,gate,gate_stats,device)
    valid_ref=C.BASE.reference_stats(cp["valid"],args.null_rounds,args.seed); valid_scores={k:C.BASE.method_stats(v,valid_ref,cp["valid"]) for k,v in valid_methods.items()}; test_ref=C.BASE.reference_stats(cp["test"],args.null_rounds,args.seed); test_scores={k:C.BASE.method_stats(v,test_ref,cp["test"]) for k,v in test_methods.items()}
    contrasts={f"posterior_minus_{base}":C.BASE.paired_bootstrap(test_scores["posterior_gate"],test_scores[base],args.bootstrap_rounds,C.BASE.stable_seed(args.seed,f"posterior|{base}")) for base in ("a_teacher","support_mean","c0_prior")}
    payload = {
        "version": VERSION, "seed": args.seed, "smoke": bool(args.smoke), "device": str(device),
        "locked_hyperparameters": {
            "teacher_folds": 5, "teacher_epochs": 80, "student_epochs": 40, "gate_epochs": 40,
            "gate_architecture": "(support_mean775+support_dispersion775)->128->32->sigmoid lambda",
            "prior_architecture": "(baselineCP775+ECFP4-2048)->512->128->CP775",
            "batch_size": 256, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "null_rounds": args.null_rounds, "bootstrap_rounds": args.bootstrap_rounds,
        },
        "teacher": teacher_meta,
        "prior": {"definition": "C0 is OOF during gate training and all C0 checkpoints use train-internal compounds only.", "oof": prior_meta, "full_train_internal_checkpoint": prior_full_meta},
        "gate": {
            "definition": "posterior=(1-lambda)A_teacher+lambda*C0_prior; lambda sees only support mean and within-support dispersion.",
            "train_internal_checkpoint": gate_meta,
            "lambda_summary": {"validation_mean": float(valid_lambda.mean()), "validation_median": float(np.median(valid_lambda)), "test_mean": float(test_lambda.mean()), "test_median": float(np.median(test_lambda))},
        },
        "validation_summaries": {key: C.BASE.summarize(value) for key, value in valid_scores.items()},
        "test_reference_molecule_count": len(test_ref),
        "summaries": {key: C.BASE.summarize(value) for key, value in test_scores.items()},
        "paired_bootstrap": contrasts,
        "definition": "Virtual prior plus actual CP-repeat evidence predicts an independent held-out CP plate. The final posterior is never used in a zero-repeat setting.",
        "guardrails": ["no GE data, feature, target, or loss", "C0 prior has molecule plus baseline input only", "gate sees evidence only and cannot see the held plate", "teacher, C0, and gate checkpoint splits are internal to official training", "official validation/test data are not used for training or checkpoint selection", "strict held-out RSF is primary", "smoke output is not evidence"],
    }
    C.write_scores(args.outdir/"per_molecule_scores.csv",test_scores); (args.outdir/"metrics.json").write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8"); print(json.dumps(payload,indent=2,sort_keys=True))


def aggregate(root:Path)->None:
    paths=[root/f"seed{s}"/"metrics.json" for s in C.SEEDS]; missing=[str(p) for p in paths if not p.is_file()]
    if missing: raise FileNotFoundError("Missing seed metrics: "+", ".join(missing))
    payloads=[json.loads(p.read_text(encoding="utf-8")) for p in paths]; names=sorted(payloads[0]["summaries"]); summary={n:{m:float(np.mean([p["summaries"][n][m] for p in payloads])) for m in ("model_excess_z_mean","replicate_excess_z_mean","rsf")} | {"molecule_count":int(payloads[0]["summaries"][n]["molecule_count"])} for n in names}; out={"version":VERSION,"seeds":list(C.SEEDS),"seed_metric_paths":[str(p) for p in paths],"seed_mean_summaries":summary,"test_lambda_means":{str(p["seed"]):p["gate"]["lambda_summary"]["test_mean"] for p in payloads},"note":"Inspect every seed-specific paired bootstrap posterior contrast before a decision."}; path=root/"aggregate_index.json"
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(out,indent=2,sort_keys=True),encoding="utf-8"); print(json.dumps(out,indent=2,sort_keys=True))


def main()->None:
    args=parse_args(); aggregate(args.aggregate_root) if args.aggregate_root else run(args)
if __name__=="__main__": main()
