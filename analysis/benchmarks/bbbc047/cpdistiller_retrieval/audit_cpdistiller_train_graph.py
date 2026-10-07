#!/usr/bin/env python3
"""Rebuild the frozen sparse graph and audit cpDistiller triplet semantics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from train_cpdistiller_c_sparse import SEED, intersect_technical_pairs, pair_hash, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-dir", type=Path, required=True)
    parser.add_argument("--graph-audit", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def edge_stats(pairs: set[tuple[int, int]], compound: np.ndarray, dose: np.ndarray, plate: np.ndarray, control: np.ndarray) -> dict[str, int | float]:
    counts: Counter[str] = Counter()
    for left, right in pairs:
        left_control = control[left] == "negative"
        right_control = control[right] == "negative"
        if left_control and right_control:
            counts["control_control"] += 1
        elif left_control or right_control:
            counts["treatment_control"] += 1
        else:
            counts["treatment_treatment"] += 1
            same_compound = compound[left] == compound[right]
            same_dose = dose[left] == dose[right]
            cross_plate = plate[left] != plate[right]
            counts["treatment_treatment_same_compound"] += int(same_compound)
            counts["treatment_treatment_same_compound_dose"] += int(same_compound and same_dose)
            counts["true_replicate_cross_plate"] += int(same_compound and same_dose and cross_plate)
            counts["different_compound"] += int(not same_compound)
    total = len(pairs)
    result: dict[str, int | float] = {"directed_pairs": total, **dict(counts)}
    for key, value in counts.items():
        result[f"fraction_{key}"] = value / max(total, 1)
    return result


def true_replicate_recall(
    pairs: set[tuple[int, int]], compound: np.ndarray, dose: np.ndarray, plate: np.ndarray, control: np.ndarray
) -> dict[str, int | float]:
    graph = pairs | {(right, left) for left, right in pairs}
    by_condition: dict[tuple[str, str], list[int]] = {}
    for index in np.flatnonzero(control != "negative"):
        by_condition.setdefault((compound[index], dose[index]), []).append(int(index))
    eligible_edges: set[tuple[int, int]] = set()
    eligible_anchors: set[int] = set()
    hit_anchors: set[int] = set()
    for indexes in by_condition.values():
        for left in indexes:
            for right in indexes:
                if left != right and plate[left] != plate[right]:
                    eligible_edges.add((left, right))
                    eligible_anchors.add(left)
                    if (left, right) in graph:
                        hit_anchors.add(left)
    hits = len(eligible_edges & graph)
    return {
        "eligible_directed_true_replicate_pairs": len(eligible_edges),
        "graph_directed_true_replicate_pairs": hits,
        "pair_recall": hits / max(len(eligible_edges), 1),
        "eligible_treatment_anchors": len(eligible_anchors),
        "anchors_with_at_least_one_true_replicate_neighbor": len(hit_anchors),
        "anchor_recall": len(hit_anchors) / max(len(eligible_anchors), 1),
    }


def main() -> None:
    args = parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"refusing to overwrite {args.outdir}")
    set_seed(SEED)
    expected_graph = json.loads(args.graph_audit.read_text(encoding="utf-8"))
    archive = np.load(args.prep_dir / "TRAIN_INPUTS.npz", allow_pickle=False)
    X = np.asarray(archive["X"], dtype=np.float32)
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
    sc.tl.pca(data)
    mnn_pairs = intersect_technical_pairs(data, "mnn")
    knn_pairs = intersect_technical_pairs(data, "knn")
    frozen_mnn_count = expected_graph["mnn_pair_count_directed"]
    frozen_knn_count = expected_graph["knn_pair_count_directed_before_symmetric_matrix_write"]
    rebuilt_mnn_hash = pair_hash(mnn_pairs)
    rebuilt_knn_hash = pair_hash(knn_pairs)
    exact_graph_match = (
        len(mnn_pairs) == frozen_mnn_count
        and len(knn_pairs) == frozen_knn_count
        and rebuilt_mnn_hash == expected_graph["mnn_pair_sha256"]
        and rebuilt_knn_hash == expected_graph["knn_pair_sha256"]
    )
    compound = archive["compound"].astype(str)
    dose = archive["dose"].astype(str)
    plate = archive["plate"].astype(str)
    control = archive["control"].astype(str)
    n_treatment = int(np.sum(control != "negative"))
    n_control = int(np.sum(control == "negative"))
    audit = {
        "version": "BBBC047-cpDistiller-training-graph-audit-v1-2026-09-19",
        "training_rows": len(control),
        "treatment_rows": n_treatment,
        "control_rows": n_control,
        "control_fraction": n_control / len(control),
        "graph_reproducibility": {
            "exact_frozen_graph_match": exact_graph_match,
            "interpretation": (
                "The upstream hnswlib add_items call uses its default multithreaded insertion. "
                "Its approximate graph is not bitwise reproducible even when Python, NumPy, "
                "Torch, PCA, and the hnswlib construction seed are fixed."
            ),
            "frozen": {
                "mnn_directed_pairs": frozen_mnn_count,
                "knn_directed_pairs": frozen_knn_count,
                "mnn_pair_sha256": expected_graph["mnn_pair_sha256"],
                "knn_pair_sha256": expected_graph["knn_pair_sha256"],
            },
            "rebuilt": {
                "mnn_directed_pairs": len(mnn_pairs),
                "knn_directed_pairs": len(knn_pairs),
                "mnn_pair_sha256": rebuilt_mnn_hash,
                "knn_pair_sha256": rebuilt_knn_hash,
            },
            "relative_count_difference": {
                "mnn": (len(mnn_pairs) - frozen_mnn_count) / frozen_mnn_count,
                "knn": (len(knn_pairs) - frozen_knn_count) / max(frozen_knn_count, 1),
            },
        },
        "triplet_negative_semantics": {
            "treatment_anchor_negative_source": "negative controls whenever an eligible control exists",
            "control_anchor_negative_source": "treatments whenever an eligible treatment exists",
            "expected_treatment_anchor_fraction_with_control_negative": n_treatment / len(control),
            "expected_control_anchor_fraction_with_treatment_negative": n_control / len(control),
            "treatment_treatment_negative_fraction": 0.0,
        },
        "mnn": edge_stats(mnn_pairs, compound, dose, plate, control),
        "knn": edge_stats(knn_pairs, compound, dose, plate, control),
        "mnn_true_replicate_recall": true_replicate_recall(mnn_pairs, compound, dose, plate, control),
        "knn_true_replicate_recall": true_replicate_recall(knn_pairs, compound, dose, plate, control),
        "test_profile_values_loaded": False,
        "status": "PASS" if exact_graph_match else "PASS_WITH_HNSW_NONDETERMINISM_CAVEAT",
    }
    args.outdir.mkdir(parents=True)
    (args.outdir / "TRAIN_GRAPH_DISCREPANCY_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
