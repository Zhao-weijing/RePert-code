#!/usr/bin/env python3
"""Compare the memory-safe graph against released cpDistiller on a toy set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from cpDistiller.labeled_data import labeled
from train_cpdistiller_c_sparse import adjacency, intersect_technical_pairs


def relation(neighbors: list[set[int]]) -> set[tuple[int, int]]:
    return {(left, right) for left, values in enumerate(neighbors) for right in values}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.RandomState(3407)
    labels = [
        (f"B{batch}", f"R{row}", f"C{col}")
        for batch in range(3)
        for row in range(3)
        for col in range(3)
        for _ in range(5)
    ]
    obs = pd.DataFrame(labels, columns=["batch", "row", "col"])
    for field in obs:
        obs[field] = pd.Categorical(obs[field])
    data = ad.AnnData(X=rng.normal(size=(len(obs), 30)).astype(np.float32), obs=obs)
    labeled(
        data,
        Mnn=5,
        Knn=10,
        Mnn_list=["batch", "row", "col"],
        Knn_list=["batch", "row", "col"],
        deep=True,
        deep_dim=0,
    )
    dense = np.asarray(data.obsp["matrix"])
    released_mnn = set(zip(*np.where(dense == 3)))
    released_knn = set(zip(*np.where(dense == 2)))

    # Reuse the identical PCA coordinates while independently rebuilding the
    # graph through the memory-safe execution order.
    data.obs["name_num"] = np.arange(data.shape[0])
    sparse_mnn_pairs = intersect_technical_pairs(data, "mnn")
    sparse_knn_pairs = intersect_technical_pairs(data, "knn")
    sparse_mnn = relation(adjacency(data.shape[0], sparse_mnn_pairs))
    sparse_knn = relation(adjacency(data.shape[0], sparse_knn_pairs))
    audit = {
        "audit": "cpDistiller-released-dense-vs-memory-safe-sparse-v1",
        "n_rows": data.shape[0],
        "released_mnn_relations": len(released_mnn),
        "sparse_mnn_relations": len(sparse_mnn),
        "released_knn_relations": len(released_knn),
        "sparse_knn_relations": len(sparse_knn),
        "mnn_equal": released_mnn == sparse_mnn,
        "knn_equal": released_knn == sparse_knn,
    }
    audit["status"] = "PASS" if audit["mnn_equal"] and audit["knn_equal"] else "FAIL"
    print(json.dumps(audit, indent=2, sort_keys=True))
    if args.output:
        args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not audit["mnn_equal"] or not audit["knn_equal"]:
        raise RuntimeError("sparse graph differs from released dense graph")


if __name__ == "__main__":
    main()
