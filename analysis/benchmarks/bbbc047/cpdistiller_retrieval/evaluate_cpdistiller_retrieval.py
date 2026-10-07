#!/usr/bin/env python3
"""Evaluate frozen cpDistiller-C embeddings on frozen BBBC047 retrieval roles."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import torch

from cpDistiller.model import GMVAE


VERSION = "BBBC047-cpDistiller-retrieval-v1-2026-09-19"
SEED = 3407
BOOTSTRAP_ROUNDS = 10_000
FEATURE_PREFIXES = ("Cells_", "Cytoplasm_", "Nuclei_")
EXPECTED_FEATURES = 1783
ARMS = ("RawMean", "CFRA", "cpDistiller-C")
METRICS = ("mAP_at_33", "held_rank", "top1", "query_held_cosine", "mean_query_foreign_cosine")
EXPECTED_COMPARATOR_HASHES = {
    1: "40f92af9a60930445397c48a2e144c9597cad5dde2744de3bdba1d6ec7d3c18a",
    2: "00ec44bc9dad65dc74ad5d9948576514f54bb74d7d8041d90e3166448736a84f",
    3: "ce0af845c9d0a570c8d1ab714368212706da0b607d067290c95727bcf16d9256",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-profiles", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--prep-dir", type=Path, required=True)
    parser.add_argument("--train-run-dir", type=Path, required=True)
    parser.add_argument("--roles-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--rows-root", type=Path, required=True)
    parser.add_argument("--comparator-root", type=Path, required=True)
    parser.add_argument("--frozen-evaluator", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{VERSION}|{label}".encode()).digest()[:8], "big")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("cpdistiller_frozen_external", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_test_mapping(path: Path) -> tuple[dict[int, dict[str, str]], set[str]]:
    rows: dict[int, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"source_row_index", "split", "compound", "dose", "plate", "well", "row", "column", "batch"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"mapping missing fields: {sorted(missing)}")
        for row in reader:
            if row["split"] != "test":
                continue
            source_index = int(row["source_row_index"])
            if source_index in rows:
                raise RuntimeError(f"duplicate source row {source_index}")
            rows[source_index] = row
    if not rows:
        raise RuntimeError("empty test source mapping")
    return rows, {row["plate"] for row in rows.values()}


def source_schema(path: Path) -> tuple[list[int], int, int]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    meta_count = next(index for index, name in enumerate(header) if not name.startswith("Metadata_"))
    payload = header[meta_count:]
    feature_offsets = [index for index, name in enumerate(payload) if name.startswith(FEATURE_PREFIXES)]
    if len(feature_offsets) != EXPECTED_FEATURES or feature_offsets != list(range(EXPECTED_FEATURES)):
        raise RuntimeError("source CellProfiler feature schema changed")
    return feature_offsets, meta_count, len(payload)


def frozen_transforms(prep_npz: Path, test_plates: set[str]) -> tuple[dict[str, tuple[np.ndarray, ...]], tuple[np.ndarray, ...], dict[str, Any]]:
    archive = np.load(prep_npz, allow_pickle=False)
    plate_labels = archive["preprocessing_plate"].astype(str)
    arrays = [
        np.asarray(archive[name], dtype=np.float32)
        for name in ("plate_control_center", "plate_control_mad", "plate_post_mean", "plate_post_std")
    ]
    by_plate = {
        plate: tuple(values[index] for values in arrays)
        for index, plate in enumerate(plate_labels)
    }
    fallback = tuple(np.median(values.astype(np.float64), axis=0).astype(np.float32) for values in arrays)
    global_values = (
        np.asarray(archive["global_mean"], dtype=np.float32),
        np.asarray(archive["global_std"], dtype=np.float32),
    )
    unseen = sorted(test_plates - set(by_plate))
    audit = {
        "rule": "training plate parameters where available; per-feature median of training plate parameters for unseen plates",
        "n_training_plates": len(by_plate),
        "n_test_plates_metadata_only": len(test_plates),
        "n_unseen_test_plates": len(unseen),
        "unseen_test_plates": unseen,
    }
    return by_plate, fallback + global_values, audit


def transform_block(raw: np.ndarray, parameters: tuple[np.ndarray, ...]) -> np.ndarray:
    center, mad, post_mean, post_std, global_mean, global_std = parameters
    block = np.asarray(raw, dtype=np.float64)
    invalid = ~np.isfinite(block)
    if invalid.any():
        block[invalid] = np.broadcast_to(center, block.shape)[invalid]
    block = (block - center) / (mad + 1e-18)
    block = (block - post_mean) / post_std
    block = (block - global_mean) / global_std
    return block.astype(np.float32)


def embed_test_wells(
    source_path: Path,
    mapping: dict[int, dict[str, str]],
    by_plate: dict[str, tuple[np.ndarray, ...]],
    fallback_and_global: tuple[np.ndarray, ...],
    model: GMVAE,
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, str]], dict[str, Any]]:
    feature_offsets, meta_count, payload_count = source_schema(source_path)
    selected = sorted(mapping)
    positions = {source_index: output_index for output_index, source_index in enumerate(selected)}
    raw = np.empty((len(selected), EXPECTED_FEATURES), dtype=np.float32)
    found: set[int] = set()
    with gzip.open(source_path, "rt", newline="", encoding="utf-8") as handle:
        next(handle)
        for source_index, line in enumerate(handle):
            if source_index not in positions:
                continue
            values = next(csv.reader([line]))
            if len(values) != meta_count + payload_count:
                raise RuntimeError(f"source row {source_index}: width mismatch")
            raw[positions[source_index]] = np.asarray(
                [values[meta_count + offset] for offset in feature_offsets], dtype=np.float32
            )
            found.add(source_index)
    if found != set(selected):
        raise RuntimeError("some mapped test source rows were not found")
    rows = [mapping[index] for index in selected]
    global_mean, global_std = fallback_and_global[-2:]
    fallback = fallback_and_global[:4]
    transformed = np.empty_like(raw)
    fallback_rows = 0
    plates = np.asarray([row["plate"] for row in rows])
    for plate in sorted(set(plates)):
        mask = plates == plate
        local = by_plate.get(plate)
        if local is None:
            local = fallback
            fallback_rows += int(mask.sum())
        transformed[mask] = transform_block(raw[mask], local + (global_mean, global_std))
    if not np.isfinite(transformed).all():
        raise RuntimeError("non-finite test values after frozen transformation")
    output: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(transformed), 256):
        tensor = torch.from_numpy(transformed[start : start + 256]).to(device)
        with torch.no_grad():
            output.append(model(tensor)["mu"].cpu().numpy().astype(np.float32))
    return np.concatenate(output), rows, {"n_test_wells": len(rows), "n_fallback_wells": fallback_rows}


def aggregate_physical(embeddings: np.ndarray, rows: list[dict[str, str]]) -> dict[tuple[str, str, str], np.ndarray]:
    grouped: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    for embedding, row in zip(embeddings, rows):
        grouped[(row["compound"], row["dose"], row["plate"])].append(embedding)
    return {key: np.mean(values, axis=0).astype(np.float32) for key, values in grouped.items()}


def parse_condition(value: str) -> tuple[str, str]:
    compound, dose = value.rsplit("::", 1)
    return compound, dose


def assemble_embeddings(roles: list[dict[str, str]], profiles: dict[tuple[str, str, str], np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    query, held, foreign, compounds, conditions = [], [], [], [], []
    for row in roles:
        compound, dose = row["compound_id"], row["dose"]
        support_ids = row["support_plate_ids"].split("|")
        support = [profiles[(compound, dose, plate)] for plate in support_ids]
        query.append(np.mean(support, axis=0))
        held.append(profiles[(compound, dose, row["held_plate_id"])])
        donors = row["foreign_condition_ids"].split("|")
        donor_plates = row["foreign_held_plate_ids"].split("|")
        foreign.append(np.stack([profiles[(*parse_condition(donor), plate)] for donor, plate in zip(donors, donor_plates)]))
        compounds.append(compound)
        conditions.append(row["condition_id"])
    return np.stack(query), np.stack(held), foreign, np.asarray(compounds), np.asarray(conditions)


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left.astype(np.float64) * right.astype(np.float64), axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    result = np.zeros(len(left), dtype=np.float64)
    valid = denominator > 1e-12
    result[valid] = numerator[valid] / denominator[valid]
    return np.clip(result, -1.0, 1.0)


def retrieval_metrics(query: np.ndarray, held: np.ndarray, foreign: list[np.ndarray]) -> dict[str, np.ndarray]:
    held_cosine = cosine(query, held)
    foreign_mean = np.empty(len(query), dtype=np.float64)
    rank = np.empty(len(query), dtype=np.float64)
    for index, gallery in enumerate(foreign):
        repeated = np.repeat(query[index : index + 1], len(gallery), axis=0)
        scores = cosine(repeated, gallery)
        foreign_mean[index] = float(scores.mean())
        rank[index] = 1.0 + float(np.sum(scores > held_cosine[index]))
    return {
        "mAP_at_33": 1.0 / rank,
        "held_rank": rank,
        "top1": (rank == 1).astype(np.float64),
        "query_held_cosine": held_cosine,
        "mean_query_foreign_cosine": foreign_mean,
    }


def compound_values(values: np.ndarray, compounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for compound, value in zip(compounds, values):
        grouped[str(compound)].append(float(value))
    ids = np.asarray(sorted(grouped))
    return ids, np.asarray([np.mean(grouped[key]) for key in ids], dtype=np.float64)


def bootstrap_summary(values: np.ndarray, compounds: np.ndarray, label: str) -> tuple[float, float, float, int]:
    _, reduced = compound_values(values, compounds)
    rng = np.random.default_rng(stable_seed(f"summary|{label}"))
    draws = rng.integers(0, len(reduced), size=(BOOTSTRAP_ROUNDS, len(reduced)))
    sampled = reduced[draws].mean(axis=1)
    return float(reduced.mean()), float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975)), len(reduced)


def bootstrap_delta(left: np.ndarray, right: np.ndarray, compounds: np.ndarray, label: str) -> tuple[float, float, float, int]:
    left_ids, left_values = compound_values(left, compounds)
    right_ids, right_values = compound_values(right, compounds)
    if not np.array_equal(left_ids, right_ids):
        raise RuntimeError("paired compound identities differ")
    delta = left_values - right_values
    rng = np.random.default_rng(stable_seed(f"delta|{label}"))
    draws = rng.integers(0, len(delta), size=(BOOTSTRAP_ROUNDS, len(delta)))
    sampled = delta[draws].mean(axis=1)
    return float(delta.mean()), float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975)), len(delta)


def main() -> None:
    args = parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"refusing to overwrite {args.outdir}")
    train_config_path = args.train_run_dir / "TRAIN_CONFIG.json"
    train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
    checkpoint = args.train_run_dir / "cpDistillerC_memory_safe_sparse_v1" / "final_model_ema.ckpt"
    if train_config.get("status") != "PASS" or not checkpoint.is_file():
        raise RuntimeError("frozen training run is incomplete")
    evaluator = load_module(args.frozen_evaluator)
    role_audit = evaluator.check_freeze(args.roles_root, args.protocol)
    mapping, test_plates = load_test_mapping(args.mapping)
    prep_npz = args.prep_dir / "TRAIN_INPUTS.npz"
    by_plate, fallback_and_global, transform_audit = frozen_transforms(prep_npz, test_plates)
    comparator_hashes = {}
    for budget, expected in EXPECTED_COMPARATOR_HASHES.items():
        path = args.comparator_root / f"budget{budget}_TEST_PREDICTIONS.npz"
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"frozen comparator hash mismatch at {budget}R")
        comparator_hashes[str(budget)] = actual

    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("frozen cpDistiller inference requires CUDA")
    model = GMVAE(EXPECTED_FEATURES, 512, 50, 10).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    model.eval()
    args.outdir.mkdir(parents=True)
    transform_path = args.outdir / "FROZEN_INFERENCE_FALLBACK.npz"
    np.savez_compressed(
        transform_path,
        fallback_control_center=fallback_and_global[0],
        fallback_control_mad=fallback_and_global[1],
        fallback_post_mean=fallback_and_global[2],
        fallback_post_std=fallback_and_global[3],
        global_mean=fallback_and_global[4],
        global_std=fallback_and_global[5],
    )
    preflight = {
        "version": VERSION,
        "stage": "test_preflight_frozen",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "train_config_sha256": sha256_file(train_config_path),
        "train_input_sha256": sha256_file(prep_npz),
        "mapping_sha256": sha256_file(args.mapping),
        "role_manifest_sha256": role_audit["manifest"]["sha256"],
        "frozen_comparator_hashes": comparator_hashes,
        "inference_fallback_sha256": sha256_file(transform_path),
        "transform": transform_audit,
        "checkpoint_selection": "official fixed 50-epoch EMA checkpoint; no validation or test endpoint selection",
        "test_profile_values_loaded": False,
        "test_used_for_fit_or_selection": False,
        "status": "PASS",
    }
    (args.outdir / "EVAL_PREFLIGHT.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # The first test-value access occurs only after every configuration and
    # model/comparator hash above has been frozen to disk.
    set_seed(SEED)
    embeddings, source_rows, inference_audit = embed_test_wells(
        args.source_profiles, mapping, by_plate, fallback_and_global, model, device
    )
    physical_embeddings = aggregate_physical(embeddings, source_rows)
    np.savez_compressed(
        args.outdir / "TEST_WELL_EMBEDDINGS.npz",
        embedding=embeddings,
        source_row_index=np.asarray([int(row["source_row_index"]) for row in source_rows]),
        compound=np.asarray([row["compound"] for row in source_rows]),
        dose=np.asarray([row["dose"] for row in source_rows]),
        plate=np.asarray([row["plate"] for row in source_rows]),
        well=np.asarray([row["well"] for row in source_rows]),
    )
    test_raw_profiles = evaluator.load_profiles(args.rows_root, "test")
    summaries: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    per_condition: list[dict[str, Any]] = []
    condition_counts: dict[str, int] = {}
    for budget in (1, 2, 3):
        roles = evaluator.read_roles(args.roles_root, "test", budget)
        raw_query, raw_held, compounds, raw_foreign, conditions, _ = evaluator.assemble(
            roles, test_raw_profiles, foreign=True
        )
        archive = np.load(args.comparator_root / f"budget{budget}_TEST_PREDICTIONS.npz", allow_pickle=False)
        if not np.array_equal(archive["condition"].astype(str), conditions.astype(str)):
            raise RuntimeError(f"frozen comparator condition order mismatch at {budget}R")
        if not np.allclose(archive["Mean"], raw_query, rtol=0, atol=1e-7):
            raise RuntimeError(f"frozen Raw/Mean query mismatch at {budget}R")
        cp_query, cp_held, cp_foreign, cp_compounds, cp_conditions = assemble_embeddings(roles, physical_embeddings)
        if not np.array_equal(cp_conditions, conditions) or not np.array_equal(cp_compounds, compounds):
            raise RuntimeError(f"cpDistiller role assembly mismatch at {budget}R")
        arm_metrics = {
            "RawMean": retrieval_metrics(raw_query, raw_held, raw_foreign),
            "CFRA": retrieval_metrics(np.asarray(archive["CFRA"], dtype=np.float32), raw_held, raw_foreign),
            "cpDistiller-C": retrieval_metrics(cp_query, cp_held, cp_foreign),
        }
        condition_counts[str(budget)] = len(conditions)
        for arm in ARMS:
            for metric in METRICS:
                point, low, high, n = bootstrap_summary(
                    arm_metrics[arm][metric], compounds, f"b{budget}|{arm}|{metric}"
                )
                summaries.append(
                    {
                        "budget": budget, "arm": arm, "metric": metric,
                        "point": point, "ci_low": low, "ci_high": high,
                        "n_compounds": n, "n_conditions": len(conditions),
                        "bootstrap_rounds": BOOTSTRAP_ROUNDS,
                    }
                )
            for index, condition in enumerate(conditions):
                per_condition.append(
                    {
                        "budget": budget, "arm": arm, "compound_id": compounds[index],
                        "condition_id": condition,
                        **{metric: float(arm_metrics[arm][metric][index]) for metric in METRICS},
                    }
                )
        for arm in ("cpDistiller-C", "CFRA"):
            for metric in METRICS:
                point, low, high, n = bootstrap_delta(
                    arm_metrics[arm][metric], arm_metrics["RawMean"][metric], compounds,
                    f"b{budget}|{arm}-RawMean|{metric}",
                )
                contrasts.append(
                    {
                        "budget": budget, "comparison": f"{arm} - RawMean", "metric": metric,
                        "point": point, "ci_low": low, "ci_high": high,
                        "n_compounds": n, "n_conditions": len(conditions),
                        "bootstrap_rounds": BOOTSTRAP_ROUNDS,
                    }
                )
    write_csv(args.outdir / "TEST_RETRIEVAL_ARM_SUMMARY.csv", summaries)
    write_csv(args.outdir / "TEST_RETRIEVAL_CONTRASTS.csv", contrasts)
    write_csv(args.outdir / "TEST_RETRIEVAL_PER_CONDITION.csv", per_condition)
    marker = {
        "version": VERSION,
        "stage": "test_retrieval_once",
        "historical_test_status": "supportive non-blind recomputation",
        "representation_scope": "cpDistiller-C is 50D retrieval and is not part of the frozen 775D E leaderboard",
        "physical_aggregation": "embed source wells independently, then mean within physical compound-dose-plate",
        "support_aggregation": "mean independently embedded support physical profiles",
        "gallery": "one frozen held positive plus the same 32 frozen foreign physical profiles",
        "bootstrap": {"rounds": BOOTSTRAP_ROUNDS, "unit": "compound after dose-condition aggregation", "ci": "95% percentile"},
        "condition_counts": condition_counts,
        "physical_test_profiles": len(physical_embeddings),
        "inference": inference_audit,
        "test_profile_values_loaded": True,
        "test_used_for_fit_or_selection": False,
        "status": "PASS",
    }
    (args.outdir / "TEST_RETRIEVAL_COMPLETE.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(marker, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
