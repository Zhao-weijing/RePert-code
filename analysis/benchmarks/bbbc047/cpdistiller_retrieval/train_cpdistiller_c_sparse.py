#!/usr/bin/env python3
"""Train cpDistiller-C on frozen BBBC047 training-only inputs.

This launcher preserves the official cpDistiller-C encoder, loss, optimizer,
and row/column/batch neighbour definitions.  Its graph/triplet container is a
memory-safe replacement for the upstream dense ``N x N`` matrix and per-row
candidate caches, which cannot be instantiated at N=52,370 on the 62-GiB
server.  It deliberately never opens a validation or test profile table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import scanpy as sc
import torch

from cpDistiller.labeled_data import get_nn, mnn
from cpDistiller.main import cpDistiller_Model

warnings.filterwarnings(
    "ignore",
    message="Series.__getitem__ treating keys as positions is deprecated.*",
    category=FutureWarning,
)


SEED = 3407
MNN_K = 5
KNN_K = 10
EPOCHS = 50
BATCH_SIZE = 256
DIM_HIDDEN = 512
DIM_OUT = 50
CATEGORY = 10
TECHNICAL_FIELDS = ("batch", "row", "col")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_hash(pairs: Iterable[tuple[int, int]]) -> str:
    digest = hashlib.sha256()
    for left, right in sorted((int(a), int(b)) for a, b in pairs):
        digest.update(f"{left},{right}\n".encode("ascii"))
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-dir", type=Path, required=True)
    parser.add_argument("--vendor-dir", type=Path, required=True)
    parser.add_argument("--equivalence-audit", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def upstream_mnn_pairs(data: ad.AnnData, field: str) -> set[tuple[int, int]]:
    """Execute the same pairwise HNSW MNN calls as upstream ``batch_mnn``."""
    values = sorted(set(data.obs[field].astype(str)))
    pairs: set[tuple[int, int]] = set()
    for left_index, left_value in enumerate(values):
        left = data[data.obs[field].astype(str).to_numpy() == left_value]
        for right_value in values[left_index + 1 :]:
            right = data[data.obs[field].astype(str).to_numpy() == right_value]
            pairs |= mnn(
                left.obsm["X_pca"],
                right.obsm["X_pca"],
                left.obs["name_num"],
                right.obs["name_num"],
                knn=MNN_K,
            )
    return {(int(left), int(right)) for left, right in pairs}


def upstream_knn_pairs(data: ad.AnnData, field: str) -> set[tuple[int, int]]:
    """Execute the same within-group HNSW KNN calls as upstream ``batch_knn``."""
    values = sorted(set(data.obs[field].astype(str)))
    pairs: set[tuple[int, int]] = set()
    for value in values:
        subset = data[data.obs[field].astype(str).to_numpy() == value]
        pairs |= get_nn(
            subset.obsm["X_pca"],
            subset.obsm["X_pca"],
            subset.obs["name_num"],
            subset.obs["name_num"],
            knn=KNN_K,
            type="knn",
        )
    return {(int(left), int(right)) for left, right in pairs}


def intersect_technical_pairs(data: ad.AnnData, kind: str) -> set[tuple[int, int]]:
    """Return the exact upstream batch/row/col set intersection.

    Row and column intersections are evaluated first.  Batch HNSW calls are
    then limited to batch pairs represented by those candidates.  This changes
    execution order only; each retained pair is tested by the same upstream
    function and parameters as the released implementation.
    """
    if kind == "mnn":
        row_pairs = upstream_mnn_pairs(data, "row")
        print(json.dumps({"stage": "mnn_row", "pairs": len(row_pairs)}), flush=True)
        col_pairs = upstream_mnn_pairs(data, "col")
        print(json.dumps({"stage": "mnn_col", "pairs": len(col_pairs)}), flush=True)
    elif kind == "knn":
        row_pairs = upstream_knn_pairs(data, "row")
        print(json.dumps({"stage": "knn_row", "pairs": len(row_pairs)}), flush=True)
        col_pairs = upstream_knn_pairs(data, "col")
        print(json.dumps({"stage": "knn_col", "pairs": len(col_pairs)}), flush=True)
    else:
        raise ValueError(kind)
    candidates = row_pairs & col_pairs
    del row_pairs, col_pairs
    print(json.dumps({"stage": f"{kind}_row_col_intersection", "pairs": len(candidates)}), flush=True)
    batch_values = data.obs["batch"].astype(str).to_numpy()
    represented: dict[tuple[str, str], set[tuple[int, int]]] = {}
    for left, right in candidates:
        left_batch, right_batch = batch_values[left], batch_values[right]
        if kind == "mnn" and left_batch != right_batch:
            key = tuple(sorted((left_batch, right_batch)))
            represented.setdefault(key, set()).add((left, right))
        elif kind == "knn" and left_batch == right_batch:
            represented.setdefault((left_batch, left_batch), set()).add((left, right))
    print(json.dumps({"stage": f"{kind}_batch_groups", "groups": len(represented)}), flush=True)
    result: set[tuple[int, int]] = set()
    started = time.time()
    for group_index, (key, group_candidates) in enumerate(sorted(represented.items()), start=1):
        left_batch, right_batch = key
        left = data[batch_values == left_batch]
        if kind == "mnn":
            right = data[batch_values == right_batch]
            produced = mnn(
                left.obsm["X_pca"], right.obsm["X_pca"],
                left.obs["name_num"], right.obs["name_num"], knn=MNN_K,
            )
        else:
            produced = get_nn(
                left.obsm["X_pca"], left.obsm["X_pca"],
                left.obs["name_num"], left.obs["name_num"],
                knn=KNN_K, type="knn",
            )
        result |= group_candidates & {(int(left), int(right)) for left, right in produced}
        if group_index % 1000 == 0 or group_index == len(represented):
            print(
                json.dumps(
                    {
                        "stage": f"{kind}_batch_progress",
                        "completed_groups": group_index,
                        "total_groups": len(represented),
                        "retained_pairs": len(result),
                        "elapsed_seconds": round(time.time() - started, 1),
                    }
                ),
                flush=True,
            )
    return result


def adjacency(n_rows: int, pairs: set[tuple[int, int]]) -> list[set[int]]:
    result = [set() for _ in range(n_rows)]
    # Upstream writes both directions into matrix regardless of whether a KNN
    # pair is directed, so the sparse substitute is explicitly symmetric too.
    for left, right in pairs:
        if left == right:
            continue
        result[left].add(right)
        result[right].add(left)
    return result


class SparseTripletDataSet:
    """cpDistiller DataSet-compatible, sparse candidate sampler.

    Candidate priorities and uniform draws match ``DataSet.prepare_triplet``:
    MNN then KNN then self for treatment positives; any negative control for a
    treatment negative; any treatment for a negative-control negative.  The
    upstream's dense zero search is represented by rejection against the same
    sparse MNN/KNN union, avoiding O(N^2) candidate caches.
    """

    def __init__(
        self,
        data: ad.AnnData,
        mnn_neighbors: list[set[int]],
        knn_neighbors: list[set[int]],
        seed: int,
        batch_size: int,
    ) -> None:
        self.data = data
        self.mod = 1
        self.batch_size = batch_size
        self.sum_num = data.shape[0]
        self.label_row = data.obs["row"].cat.codes.to_numpy()
        self.label_col = data.obs["col"].cat.codes.to_numpy()
        self.label_batch = data.obs["batch"].cat.codes.to_numpy()
        self.control_choice = np.flatnonzero(data.obs["control"].to_numpy() == "negative")
        self.other_choice = np.flatnonzero(data.obs["control"].to_numpy() != "negative")
        if len(self.control_choice) == 0 or len(self.other_choice) == 0:
            raise RuntimeError("both treatment and negative-control rows are required")
        self.mnn_neighbors = mnn_neighbors
        self.knn_neighbors = knn_neighbors
        self.excluded_neighbors = [mnn_neighbors[i] | knn_neighbors[i] | {i} for i in range(self.sum_num)]
        self.rng = np.random.RandomState(seed)
        self.py_rng = random.Random(seed)

    def _uniform_non_neighbor(self, candidates: np.ndarray, anchor: int) -> int:
        excluded = self.excluded_neighbors[anchor]
        # Technical neighbours are tiny relative to each candidate pool; this
        # produces the upstream uniform choice over zero-labelled candidates.
        for _ in range(10_000):
            candidate = int(candidates[self.rng.randint(len(candidates))])
            if candidate not in excluded:
                return candidate
        available = [int(item) for item in candidates if int(item) not in excluded]
        if not available:
            raise RuntimeError(f"anchor {anchor} has no zero-labelled candidate")
        return int(available[self.rng.randint(len(available))])

    def _positive(self, anchor: int) -> int:
        if self.py_rng.random() > 0.5:
            candidates = self.mnn_neighbors[anchor]
            if not candidates:
                candidates = self.knn_neighbors[anchor]
        else:
            candidates = self.knn_neighbors[anchor]
        if not candidates:
            return anchor
        ordered = np.fromiter(candidates, dtype=np.int64)
        return int(ordered[self.rng.randint(len(ordered))])

    def train(self):
        order = self.rng.permutation(self.sum_num)
        positive = np.empty(self.sum_num, dtype=np.int64)
        negative = np.empty(self.sum_num, dtype=np.int64)
        controls = self.data.obs["control"].to_numpy()
        for anchor in range(self.sum_num):
            if controls[anchor] == "negative":
                positive[anchor] = int(self.control_choice[self.rng.randint(len(self.control_choice))])
                negative[anchor] = self._uniform_non_neighbor(self.other_choice, anchor)
            else:
                positive[anchor] = self._positive(anchor)
                negative[anchor] = self._uniform_non_neighbor(self.control_choice, anchor)
        for start in range(0, self.sum_num, self.batch_size):
            indexes = order[start : start + self.batch_size]
            yield (
                self.data.X[indexes],
                self.data.X[positive[indexes]],
                self.data.X[negative[indexes]],
                self.label_row[indexes],
                self.label_col[indexes],
                self.label_batch[indexes],
            )

    def eval(self, input_data: ad.AnnData, batch_size: int = 256):
        for start in range(0, input_data.shape[0], batch_size):
            yield input_data.X[start : start + batch_size]


def main() -> None:
    args = parse_args()
    npz_path = args.prep_dir / "TRAIN_INPUTS.npz"
    prep_audit_path = args.prep_dir / "TRAIN_INPUT_AUDIT.json"
    if not npz_path.exists() or not prep_audit_path.exists():
        raise FileNotFoundError("expected passed train-only preparation artifacts")
    prep_audit = json.loads(prep_audit_path.read_text(encoding="utf-8"))
    if prep_audit.get("status") != "PASS" or prep_audit.get("test_used_for_fit_or_selection") is not False:
        raise RuntimeError("train-input audit does not establish a test-free PASS")
    equivalence_audit = json.loads(args.equivalence_audit.read_text(encoding="utf-8"))
    if equivalence_audit.get("status") != "PASS":
        raise RuntimeError("released-versus-sparse graph equivalence audit did not pass")
    if args.outdir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.outdir}")
    if psutil.virtual_memory().available < 10 * 1024**3:
        raise RuntimeError("less than 10 GiB RAM available for sparse training")
    if not torch.cuda.is_available():
        raise RuntimeError("cpDistiller-C training is frozen to an available CUDA GPU")

    args.outdir.mkdir(parents=True)
    set_seed(SEED)
    archive = np.load(npz_path, allow_pickle=False)
    X = np.ascontiguousarray(archive["X"], dtype=np.float32)
    obs = pd.DataFrame(
        {
            "row": pd.Categorical(archive["row"].astype(str)),
            "col": pd.Categorical(archive["col"].astype(str)),
            "batch": pd.Categorical(archive["batch"].astype(str)),
            "control": pd.Categorical(archive["control"].astype(str)),
        }
    )
    data = ad.AnnData(X=X, obs=obs)
    data.obs["name_num"] = np.arange(data.shape[0])

    config = {
        "audit": "BBBC047-cpDistiller-C-memory-safe-sparse-train-v1-2026-09-19",
        "status": "RUNNING",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "input_npz_sha256": sha256_file(npz_path),
        "train_input_audit_sha256": sha256_file(prep_audit_path),
        "launcher_sha256": sha256_file(Path(__file__)),
        "equivalence_audit_sha256": sha256_file(args.equivalence_audit),
        "upstream_vendor_commit": git_commit(args.vendor_dir),
        "upstream_source_sha256": {
            "main.py": sha256_file(args.vendor_dir / "cpDistiller" / "main.py"),
            "model.py": sha256_file(args.vendor_dir / "cpDistiller" / "model.py"),
            "losses.py": sha256_file(args.vendor_dir / "cpDistiller" / "losses.py"),
            "labeled_data.py": sha256_file(args.vendor_dir / "cpDistiller" / "labeled_data.py"),
        },
        "implementation": {
            "encoder_loss_optimizer": "official cpDistiller-C source, unmodified",
            "technical_neighbors": "official HNSW MNN/KNN calls and batch-row-col intersection",
            "sparse_difference": "replaces upstream dense N-by-N matrix and O(N^2) candidate caches with symmetric sparse adjacency plus uniform rejection sampling over the same zero-labelled pools",
            "upstream_full_dense_feasible_on_36": False,
        },
        "model": {
            "mo": "cpDistiller-C",
            "epochs": EPOCHS,
            "dim_hidden": DIM_HIDDEN,
            "dim_out": DIM_OUT,
            "category": CATEGORY,
            "lr": [1e-3, 3e-3],
            "batch_size": BATCH_SIZE,
            "mod": 1,
            "mnn_k": MNN_K,
            "knn_k": KNN_K,
            "mnn_knn_fields": list(TECHNICAL_FIELDS),
        },
        "input": {
            "n_rows": int(data.shape[0]),
            "n_features": int(data.shape[1]),
            "n_treatments": int((data.obs["control"] == "treatment").sum()),
            "n_negative_controls": int((data.obs["control"] == "negative").sum()),
            "n_batches": int(data.obs["batch"].nunique()),
        },
        "test_profile_values_opened": False,
        "test_used_for_fit_or_selection": False,
    }
    (args.outdir / "TRAIN_CONFIG.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # This is the upstream ``labeled(..., deep=True, deep_dim=0)`` PCA branch.
    # ``deep=False, deep_dim=0`` in the released source leaves X_pca undefined.
    sc.tl.pca(data)
    print(json.dumps({"stage": "pca_complete", "shape": list(data.obsm["X_pca"].shape)}), flush=True)
    mnn_pairs = intersect_technical_pairs(data, "mnn")
    knn_pairs = intersect_technical_pairs(data, "knn")
    mnn_neighbors = adjacency(data.shape[0], mnn_pairs)
    knn_neighbors = adjacency(data.shape[0], knn_pairs)
    graph_audit = {
        "mnn_pair_count_directed": len(mnn_pairs),
        "knn_pair_count_directed_before_symmetric_matrix_write": len(knn_pairs),
        "mnn_pair_sha256": pair_hash(mnn_pairs),
        "knn_pair_sha256": pair_hash(knn_pairs),
        "mnn_nonempty_anchors": int(sum(bool(row) for row in mnn_neighbors)),
        "knn_nonempty_anchors": int(sum(bool(row) for row in knn_neighbors)),
        "pca_shape": list(map(int, data.obsm["X_pca"].shape)),
        "test_profile_values_opened": False,
    }
    (args.outdir / "SPARSE_GRAPH_AUDIT.json").write_text(json.dumps(graph_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    dataset = SparseTripletDataSet(data, mnn_neighbors, knn_neighbors, SEED, BATCH_SIZE)
    model = cpDistiller_Model(
        dataset=dataset,
        model_path=str(args.outdir),
        epochs=EPOCHS,
        seed=SEED,
        dim_hidden=DIM_HIDDEN,
        dim_out=DIM_OUT,
        lr=[1e-3, 3e-3],
        gpu_id=0,
        category=CATEGORY,
        name="cpDistillerC_memory_safe_sparse_v1",
        mo="cpDistiller-C",
    )
    model.train()
    config["status"] = "PASS"
    config["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (args.outdir / "TRAIN_CONFIG.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "outdir": str(args.outdir)}, sort_keys=True))


if __name__ == "__main__":
    main()
