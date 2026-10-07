#!/usr/bin/env python3
"""Frozen CP posterior + observed GE residual evidence for cpg0004."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from torch import nn

RDLogger.DisableLog("rdApp.warning")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cell_painting_repeat_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "virtual_prior"))
import run_cp_repeat_benchmark as B
import run_prior_posterior as P


VERSION = "cpg0004-LINCS-GE-residual-evidence-2026-08-30"


class GModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 512), nn.GELU(), nn.Linear(512, 128), nn.GELU(), nn.Linear(128, output_dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def args() -> argparse.Namespace:
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,required=True); p.add_argument("--ge",type=Path,required=True); p.add_argument("--smiles",type=Path,required=True); p.add_argument("--p0-valid",type=Path,required=True); p.add_argument("--p0-test",type=Path,required=True); p.add_argument("--outdir",type=Path,required=True); p.add_argument("--seed",type=int,required=True); p.add_argument("--budget",type=int,choices=(1,2,3),default=1); p.add_argument("--dose",default="0.04"); p.add_argument("--epochs",type=int,default=60); p.add_argument("--batch-size",type=int,default=256); p.add_argument("--learning-rate",type=float,default=3e-4); p.add_argument("--weight-decay",type=float,default=1e-5); p.add_argument("--beta-max",type=float,default=.50); p.add_argument("--null-rounds",type=int,default=32); p.add_argument("--bootstrap-rounds",type=int,default=2000); p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto"); return p.parse_args()


def decode(x: np.ndarray) -> np.ndarray:
    return np.char.decode(x,"utf-8",errors="replace").astype(str) if x.dtype.kind=="S" else x.astype(str)


def load_ge(path: Path) -> tuple[dict[tuple[str,str],np.ndarray],dict[tuple[str,str],int],int]:
    with np.load(path,allow_pickle=False) as d:
        c=decode(d["compound_id"]); dose=decode(d["dose"]); effect=d["ge_effect"].astype(np.float32); n=d["ge_repeat_count"].astype(int)
    mapping={(str(a),str(b)):effect[i] for i,(a,b) in enumerate(zip(c,dose))}; repeats={(str(a),str(b)):int(n[i]) for i,(a,b) in enumerate(zip(c,dose))}
    return mapping,repeats,int(effect.shape[1])


def load_fingerprints(path: Path) -> dict[str,np.ndarray]:
    out={}
    with path.open(encoding="utf-8",newline="") as h:
        for row in csv.DictReader(h):
            mol=Chem.MolFromSmiles(row["smiles"])
            if mol is None: raise ValueError(f"invalid SMILES {row['compound_id']}")
            bv=AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048); a=np.zeros(2048,dtype=np.float32); DataStructs.ConvertToNumpyArray(bv,a); out[row["compound_id"]]=a
    return out


def dose_onehot(labels: np.ndarray) -> np.ndarray:
    values=["0.04","0.12","0.37","1.11","3.33","10"]; out=np.zeros((len(labels),6),dtype=np.float32)
    for i,x in enumerate(labels): out[i,values.index(str(x))]=1.0
    return out


def g_features(compounds: np.ndarray,doses: np.ndarray,ge_effect: np.ndarray,baseline: np.ndarray,fps: dict[str,np.ndarray]) -> np.ndarray:
    out=np.zeros((len(compounds),ge_effect.shape[1]+2048+baseline.shape[1]+6),dtype=np.float32)
    for i,c in enumerate(compounds): out[i,:ge_effect.shape[1]]=ge_effect[i]; out[i,ge_effect.shape[1]:ge_effect.shape[1]+2048]=fps[c]
    start=ge_effect.shape[1]+2048; out[:,start:start+baseline.shape[1]]=baseline; out[:,start+baseline.shape[1]:]=dose_onehot(doses); return out


def fit_g(x,y,vx,vy,seed,epochs,batch,lr,wd,device):
    B.set_seed(seed+777); gd=y.shape[1]; ge_dim=x.shape[1]-2048-gd-6
    mean=np.concatenate([x[:, :ge_dim].mean(0), x[:,ge_dim+2048:ge_dim+2048+gd].mean(0)]).astype(np.float32)
    scale=np.concatenate([np.maximum(x[:, :ge_dim].std(0),1e-6),np.maximum(x[:,ge_dim+2048:ge_dim+2048+gd].std(0),1e-6)]).astype(np.float32)
    def scale_x(z):
        z=z.copy(); z[:,:ge_dim]=(z[:,:ge_dim]-mean[:ge_dim])/scale[:ge_dim]; z[:,ge_dim+2048:ge_dim+2048+gd]=(z[:,ge_dim+2048:ge_dim+2048+gd]-mean[ge_dim:])/scale[ge_dim:]; return z
    ym=y.mean(0).astype(np.float32); ys=np.maximum(y.std(0).astype(np.float32),1e-6); y=(y-ym)/ys; vy=(vy-ym)/ys; x=scale_x(x); vx=scale_x(vx)
    m=GModel(x.shape[1],gd).to(device); opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd); g=torch.Generator(); g.manual_seed(B.stable_seed(seed,"g-loader")); loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=batch,shuffle=True,generator=g,num_workers=0); best=None; bv=float("inf"); be=-1
    for e in range(epochs):
        m.train()
        for a,b in loader:
            opt.zero_grad(set_to_none=True); loss=torch.mean((m(a.to(device))-b.to(device))**2); loss.backward(); opt.step()
        m.eval();
        with torch.no_grad(): val=float(torch.mean((m(torch.from_numpy(vx).to(device))-torch.from_numpy(vy).to(device))**2).cpu())
        if val<bv: bv=val; be=e; best={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        if e==0 or (e+1)%10==0: print(f"[G seed={seed} epoch={e}] valid_mse={val:.7f}",flush=True)
    m.load_state_dict(best); return m,{"continuous_mean":mean,"continuous_scale":scale,"target_mean":ym,"target_scale":ys,"ge_dim":ge_dim},{"best_epoch":be,"best_validation_mse":bv}


def predict_g(m,x,stats,device):
    x=x.copy(); gd=len(stats["target_mean"]); ge_dim=stats["ge_dim"]; x[:,:ge_dim]=(x[:,:ge_dim]-stats["continuous_mean"][:ge_dim])/stats["continuous_scale"][:ge_dim]; x[:,ge_dim+2048:ge_dim+2048+gd]=(x[:,ge_dim+2048:ge_dim+2048+gd]-stats["continuous_mean"][ge_dim:])/stats["continuous_scale"][ge_dim:]; m.eval();
    with torch.no_grad(): y=m(torch.from_numpy(x).to(device)).cpu().numpy()
    return y*stats["target_scale"]+stats["target_mean"]


def mse(a,b): return float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))


def subset_pairs_by_keys(rows, all_pairs, compounds, held_keys):
    idx=[]
    for i,(c,d,h) in enumerate(zip(all_pairs.compound,all_pairs.dose,all_pairs.held_index)):
        if (str(c),str(d),int(h)) in held_keys: idx.append(i)
    return B.subset_pairs(all_pairs,set(compounds)) if False else B.Pairs(all_pairs.support[idx],all_pairs.target[idx],all_pairs.compound[idx],all_pairs.dose[idx],all_pairs.held_index[idx],all_pairs.held_plate[idx],all_pairs.held_well[idx],[all_pairs.support_indices[i] for i in idx])


def main():
    a=args(); a.outdir.mkdir(parents=True, exist_ok=True); rows=B.load_rows(a.data)
    with np.load(a.data,allow_pickle=False) as d: baseline=d["baseline"].astype(np.float32)
    ge_map,ge_rep,ge_dim=load_ge(a.ge); fps=load_fingerprints(a.smiles)
    valid=np.load(a.p0_valid,allow_pickle=False); test=np.load(a.p0_test,allow_pickle=False)
    vc0=valid["compound"].astype(str); vd0=valid["dose"].astype(str); vh0=valid["held_index"].astype(np.int64); tc0=test["compound"].astype(str); td0=test["dose"].astype(str); th0=test["held_index"].astype(np.int64)
    valid_ge_mask=np.asarray([(c,d) in ge_map for c,d in zip(vc0,vd0)],dtype=bool); test_ge_mask=np.asarray([(c,d) in ge_map for c,d in zip(tc0,td0)],dtype=bool)
    vc,vd,vh=vc0[valid_ge_mask],vd0[valid_ge_mask],vh0[valid_ge_mask]; tc,td,th=tc0[test_ge_mask],td0[test_ge_mask],th0[test_ge_mask]
    train_ix=np.asarray([i for i,(c,d,s) in enumerate(zip(rows.compound,rows.dose,rows.split)) if s=="train" and (c,d) in ge_map and c in fps],dtype=np.int64)
    valid_row_ix=np.asarray([i for i,(c,d,s) in enumerate(zip(rows.compound,rows.dose,rows.split)) if s=="valid" and (c,d) in ge_map and c in fps],dtype=np.int64)
    def rows_x(ix): return g_features(rows.compound[ix],rows.dose[ix],np.vstack([ge_map[(c,d)] for c,d in zip(rows.compound[ix],rows.dose[ix])]),baseline[ix],fps)
    x=rows_x(train_ix); y=rows.delta[train_ix]; vx=rows_x(valid_row_ix); vy=rows.delta[valid_row_ix]
    device=B.select_device(a.device); m,stats,gmeta=fit_g(x,y,vx,vy,a.seed,a.epochs,a.batch_size,a.learning_rate,a.weight_decay,device)
    valid_ge=np.vstack([ge_map[(c,d)] for c,d in zip(vc,vd)]); test_ge=np.vstack([ge_map[(c,d)] for c,d in zip(tc,td)])
    xv=g_features(vc,vd,valid_ge,baseline[vh],fps); xt=g_features(tc,td,test_ge,baseline[th],fps); gv=predict_g(m,xv,stats,device); gt=predict_g(m,xt,stats,device)
    p0v=valid["posterior"].astype(np.float32)[valid_ge_mask]; vv=valid["virtual_prior"].astype(np.float32)[valid_ge_mask]; targetv=valid["target"].astype(np.float32)[valid_ge_mask]; p0t=test["posterior"].astype(np.float32)[test_ge_mask]; vt=test["virtual_prior"].astype(np.float32)[test_ge_mask]
    betas=np.arange(0,float(a.beta_max)+.0001,.05); bm={f"{b:.2f}":mse(p0v+b*(gv-vv),targetv) for b in betas}; beta=float(betas[int(np.argmin([bm[f"{b:.2f}"] for b in betas]))])
    # Controls: replace only GE evidence, retaining target molecule, dose and
    # held-plate baseline.  The foreign pool matches CP/GE repeat counts.
    cp_rep={(c,d):int(np.sum((rows.compound==c)&(rows.dose==d))) for c,d in zip(rows.compound,rows.dose)}; test_compounds=sorted(set(tc)); foreign=[]
    for c,d in zip(tc,td):
        cand=[z for z in test_compounds if z!=c and ge_rep.get((z,d),0)==ge_rep.get((c,d),0) and cp_rep.get((z,d),0)==cp_rep.get((c,d),0)]
        foreign.append(cand[0] if cand else next(z for z in test_compounds if z!=c))
    foreign_ge=np.vstack([ge_map[(c,d)] for c,d in zip(foreign,td)]); xf=g_features(tc,td,foreign_ge,baseline[th],fps); gf=predict_g(m,xf,stats,device)
    shuffled_ge=test_ge.copy(); rng=np.random.default_rng(B.stable_seed(a.seed,"ge-shuffle"))
    for d in sorted(set(td)):
        ii=np.flatnonzero(td==d); shuffled_ge[ii]=test_ge[ii[rng.permutation(len(ii))]]
    xs=g_features(tc,td,shuffled_ge,baseline[th],fps); gs=predict_g(m,xs,stats,device)
    pred={"P0":p0t,"correct_GE":p0t+beta*(gt-vt),"shuffled_GE":p0t+beta*(gs-vt),"matched_foreign_GE":p0t+beta*(gf-vt)}
    all_test=B.make_pairs(rows,a.budget,a.dose); keyset={(str(c),str(d),int(h)) for c,d,h in zip(tc,td,th)}; tp=subset_pairs_by_keys(rows,all_test,set(tc),keyset); order=[(str(c),str(d),int(h)) for c,d,h in zip(tp.compound,tp.dose,tp.held_index)]
    # Align p0 arrays to the deterministic all-pairs order returned above.
    loc={(str(c),str(d),int(h)):i for i,(c,d,h) in enumerate(zip(tc,td,th))}; ix=np.asarray([loc[k] for k in order],dtype=np.int64); pred={k:v[ix] for k,v in pred.items()};
    pred["support_mean"] = tp.support
    null_z,usable,exact=B.make_matched_null(rows,tp,a.null_rounds,a.seed); summaries,vectors,raw=B.score_predictions(pred,tp,null_z,usable); contrasts={}
    for method in ("correct_GE","shuffled_GE","matched_foreign_GE"):
        contrasts[f"{method}_minus_P0_excess_z"]=B.bootstrap_difference(vectors[method],vectors["P0"],a.bootstrap_rounds,B.stable_seed(a.seed,f"boot|{method}|z")); contrasts[f"{method}_minus_P0_pcc"]=B.bootstrap_difference(raw[method],raw["P0"],a.bootstrap_rounds,B.stable_seed(a.seed,f"boot|{method}|pcc"))
    for control in ("shuffled_GE", "matched_foreign_GE"):
        contrasts[f"correct_GE_minus_{control}_excess_z"] = B.bootstrap_difference(vectors["correct_GE"], vectors[control], a.bootstrap_rounds, B.stable_seed(a.seed, f"boot|correct|{control}|z"))
        contrasts[f"correct_GE_minus_{control}_pcc"] = B.bootstrap_difference(raw["correct_GE"], raw[control], a.bootstrap_rounds, B.stable_seed(a.seed, f"boot|correct|{control}|pcc"))
    np.savez_compressed(a.outdir/"test_predictions.npz",compound=tp.compound.astype(str),dose=tp.dose.astype(str),held_index=tp.held_index.astype(np.int64),P0=pred["P0"].astype(np.float32),correct_GE=pred["correct_GE"].astype(np.float32),shuffled_GE=pred["shuffled_GE"].astype(np.float32),matched_foreign_GE=pred["matched_foreign_GE"].astype(np.float32),usable=usable.astype(bool),null_z=null_z.astype(np.float64))
    payload={"version":VERSION,"seed":a.seed,"dose":a.dose,"budget":a.budget,"device":str(device),"data":str(a.data),"ge":str(a.ge),"smiles":str(a.smiles),"p0_valid":str(a.p0_valid),"p0_test":str(a.p0_test),"training":{"train_rows":int(len(train_ix)),"valid_rows":int(len(valid_row_ix)),**gmeta},"ge_feature_dim":ge_dim,"mapped_test_pairs":int(len(test_ge)),"row_matched_null_usable_pairs":int(usable.sum()),"exact_well_null_usable_pairs":int(exact.sum()),"beta_grid":[float(x) for x in betas],"validation_beta_mse":bm,"selected_beta":beta,"controls":{"foreign_selection":"same test split, same dose, same GE repeat count and CP condition repeat count when available","shuffled_definition":"within-dose permutation of observed GE aggregate"},"summaries":summaries,"paired_bootstrap":contrasts,"definition":"P_GE=P0+beta(G_GE-V); P0, teacher, virtual prior and lambda are frozen before GE; beta is selected on validation only.","guardrails":["GE estimator fits only training compounds","no held-out CP target enters GE input","correct/shuffled/foreign controls retain target molecule, dose and baseline","exact-dose key is the six-level pert_id_dose recode","row-matched foreign null and exact-well availability recorded"]}
    (a.outdir/"metrics.json").write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False),encoding="utf-8"); print(json.dumps(payload,indent=2,sort_keys=True))

if __name__=="__main__": main()
