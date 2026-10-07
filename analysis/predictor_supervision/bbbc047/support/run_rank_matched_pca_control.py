#!/usr/bin/env python3
"""Test-sealed BBBC047 rank-matched Raw-PCA control.

This runner is intentionally separate from the historical repeat-ablation
entry point.  It reuses the deterministic BBBC047 row, CFRA-teacher and
Student helpers, but freezes a *dual* held-repeat manifest before fitting and
keeps test values behind an explicit authorization marker.

Formal ``prepare``/``fit``/``test`` require an externally certified dual-repeat
eligibility marker and matching manifest.  ``smoke`` uses a deterministic
synthetic data set and is non-scientific.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
EXPERIMENTS = Path(__file__).resolve().parents[4] / "analysis"
LEGACY_PATH = (Path(__file__).resolve().parents[4] / "analysis/predictor_supervision/bbbc047/support/run_cfra_target_replacement.py")
if not LEGACY_PATH.is_file():
    raise FileNotFoundError(f"legacy target-replacement source is required: {LEGACY_PATH}")
spec = importlib.util.spec_from_file_location("bbbc047_rank_matched_legacy", LEGACY_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"unable to import {LEGACY_PATH}")
LEGACY = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = LEGACY
spec.loader.exec_module(LEGACY)


VERSION = "BBBC047-rank-matched-pca-control-v1-2026-09-12"
PROTOCOL_SEED = 3407
BUDGETS = (1, 2, 3)
PREDICTOR_SEEDS = (3407, 42, 2025, 1337, 7331)
METHODS = ("raw_mean", "raw_pca", "cfra")
FOLDS = 5
CP_DIM = 775
FP_DIM = 2048
EXPECTED = {"train": 12175, "valid": 4059, "test": 4060}
FROZEN_RANKS = {1: 8, 2: 8, 3: 9}
FROZEN_TARGET_HASHES = {
    1: "4c42de05212569b7665d22df21ccafb81947bdc1c1b4d8f816b10a5978a8601b",
    2: "aff1000ef7ada1c672843bbefa4c6a842eb98971cf2e0c99c0eb8593a928ddc0",
    3: "d443e6a0db662cf0cc40483182601b34931a71f32820205827a72a684e23a684",
}

DEFAULT_ROWS = Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0")
DEFAULT_H5 = DEFAULT_ROWS.parent / "Paired_CP_GE_PlateMedian_v1_model_compat.h5"
DEFAULT_AGG = DEFAULT_ROWS / "molecule_aggregates.npz"
DEFAULT_LOCK = Path("/path/to/data/AIDD/MVCPert_5_27/source/baseline/artifacts/split_locks/BBBC047_smiles_split_seed3407_official_v1.json")
DEFAULT_RANK_SOURCE = Path("/path/to/data/AIDD/MVCPert_5_27/runs/bbbc047_repeat_ablation_20260912_v1")


@dataclass
class DualExamples:
    """One deterministic support condition and two independent held plates."""

    dataset: str
    modality: str
    split: str
    budget: int
    compound: np.ndarray
    dose: np.ndarray
    condition: np.ndarray
    support: np.ndarray
    held_a: np.ndarray
    held_b: np.ndarray
    foreign_a: np.ndarray
    foreign_b: np.ndarray
    foreign_ok: np.ndarray
    foreign_compound: np.ndarray
    support_plates: list[tuple[str, ...]]
    held_a_plates: list[tuple[str, ...]]
    held_b_plates: list[tuple[str, ...]]

    def __len__(self) -> int:
        return int(len(self.compound))


@dataclass
class SyntheticVirtual:
    smiles: np.ndarray
    control: np.ndarray
    fingerprint: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("rank", "prepare", "fit", "test", "smoke", "dry-run"), required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--rank-source-root", type=Path, default=DEFAULT_RANK_SOURCE)
    parser.add_argument("--dual-reference-root", type=Path)
    parser.add_argument("--rank-freeze", type=Path, default=HERE / "RANK_FREEZE.json")
    parser.add_argument("--rows-root", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--model-h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--aggregate-npz", type=Path, default=DEFAULT_AGG)
    parser.add_argument("--split-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--protocol-file", type=Path, default=HERE / "PROTOCOL.md")
    parser.add_argument("--teacher-epochs", type=int, default=80)
    parser.add_argument("--student-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--bootstrap-rounds", type=int, default=10000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true", help="relax protocol limits; prefer --phase smoke")
    args = parser.parse_args()
    relaxed = args.smoke or args.phase in ("smoke", "dry-run")
    if not relaxed and (args.teacher_epochs, args.student_epochs, args.batch_size) != (80, 40, 256):
        parser.error("formal protocol locks teacher_epochs/student_epochs/batch_size to 80/40/256")
    if not relaxed and args.bootstrap_rounds != 10000:
        parser.error("formal protocol locks bootstrap-rounds to 10000")
    if args.bootstrap_rounds < 1:
        parser.error("bootstrap-rounds must be positive")
    if args.phase in ("rank", "prepare", "fit", "test") and args.output_root is None:
        parser.error("--output-root is required for rank/prepare/fit/test")
    if args.phase in ("prepare", "fit", "test") and not args.smoke and args.dual_reference_root is None:
        parser.error("formal prepare/fit/test require --dual-reference-root")
    return args


def device_for(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_rows(rows_root: Path, splits: Iterable[str], *, smoke: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in splits:
        path = rows_root / f"{split}_cp_plate_rows.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as z:
            required = {"smiles", "dose", "plate", "delta"}
            if not required.issubset(z.files):
                raise RuntimeError(f"missing keys in {path}: {sorted(required - set(z.files))}")
            result[split] = LEGACY.Rows("BBBC047", "CP", split, z["smiles"], z["dose"], z["plate"], z["delta"])
        if not smoke and len(set(result[split].compound.tolist())) != EXPECTED[split]:
            raise RuntimeError(f"{split} compound count mismatch")
    sets = {split: set(values.compound.tolist()) for split, values in result.items()}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if left in sets and right in sets and sets[left] & sets[right]:
            raise RuntimeError(f"compound overlap: {left}/{right}")
    return result


def assert_lock(rows: dict[str, Any], split_lock: Path, *, smoke: bool = False) -> None:
    if smoke:
        return
    lock = read_json(split_lock)
    for split, values in rows.items():
        observed = set(values.compound.tolist())
        expected = {str(x) for x in lock[f"{split}_smiles"]}
        if observed != expected:
            raise RuntimeError(f"{split} rows do not match the official compound lock")


def _stable_order(plates: Iterable[str], label: str) -> list[str]:
    return sorted(plates, key=lambda p: (LEGACY.stable_int(PROTOCOL_SEED, f"dual-plate|{label}|{p}"), p))


def build_dual_examples(target: Any, budget: int, *, include_foreign: bool) -> DualExamples:
    groups = target.plate_groups()
    eligible = [key for key, plates in groups.items() if len(plates) >= budget + 2]
    by_compound: dict[str, list[tuple[str, str]]] = {}
    for key in sorted(eligible):
        by_compound.setdefault(str(key[0]), []).append(key)
    label = f"{target.dataset}|{target.modality}|{target.split}|dual-b{budget}"
    chosen = [
        min(candidates, key=lambda key: (LEGACY.stable_int(PROTOCOL_SEED, f"dual-condition|{label}|{key[0]}|{key[1]}"), key))
        for _, candidates in sorted(by_compound.items())
    ]
    chosen = sorted(chosen)
    foreign_index: dict[tuple[str, tuple[str, ...]], list[tuple[str, str]]] = {}
    if include_foreign:
        for key, plates in groups.items():
            foreign_index.setdefault((str(key[1]), tuple(sorted(plates))), []).append(key)
        for key in foreign_index:
            foreign_index[key].sort()

    compounds: list[str] = []
    doses: list[str] = []
    conditions: list[str] = []
    supports: list[np.ndarray] = []
    held_a: list[np.ndarray] = []
    held_b: list[np.ndarray] = []
    foreign_a: list[np.ndarray] = []
    foreign_b: list[np.ndarray] = []
    foreign_ok: list[bool] = []
    foreign_compounds: list[str] = []
    support_plates: list[tuple[str, ...]] = []
    held_a_plates: list[tuple[str, ...]] = []
    held_b_plates: list[tuple[str, ...]] = []
    for compound, dose in chosen:
        plate_label = f"{label}|{compound}|{dose}"
        ordered = _stable_order(groups[(compound, dose)], plate_label)
        sp = tuple(ordered[:budget])
        ap = (ordered[budget],)
        bp = (ordered[budget + 1],)
        signature = (str(dose), tuple(sorted(groups[(compound, dose)])))
        candidates = [x for x in foreign_index.get(signature, []) if x[0] != compound] if include_foreign else []
        donor = candidates[LEGACY.stable_int(PROTOCOL_SEED, f"dual-foreign|{plate_label}") % len(candidates)] if candidates else None
        if donor is None:
            fa = np.full(target.dim, np.nan, dtype=np.float32)
            fb = np.full(target.dim, np.nan, dtype=np.float32)
            donor_id = ""
            ok = False
        else:
            fa = target.condition_mean(donor, ap).astype(np.float32)
            fb = target.condition_mean(donor, bp).astype(np.float32)
            donor_id = str(donor[0])
            ok = True
        compounds.append(str(compound))
        doses.append(str(dose))
        conditions.append(f"{compound}::{dose}")
        supports.append(target.condition_mean((compound, dose), sp))
        held_a.append(target.condition_mean((compound, dose), ap))
        held_b.append(target.condition_mean((compound, dose), bp))
        foreign_a.append(fa)
        foreign_b.append(fb)
        foreign_ok.append(ok)
        foreign_compounds.append(donor_id)
        support_plates.append(sp)
        held_a_plates.append(ap)
        held_b_plates.append(bp)
    if not compounds:
        raise RuntimeError(f"no dual-repeat examples for {target.split} budget {budget}")
    shape = (len(compounds), target.dim)
    return DualExamples(
        dataset=target.dataset,
        modality=target.modality,
        split=target.split,
        budget=budget,
        compound=np.asarray(compounds, dtype=str),
        dose=np.asarray(doses, dtype=str),
        condition=np.asarray(conditions, dtype=str),
        support=np.asarray(supports, dtype=np.float32).reshape(shape),
        held_a=np.asarray(held_a, dtype=np.float32).reshape(shape),
        held_b=np.asarray(held_b, dtype=np.float32).reshape(shape),
        foreign_a=np.asarray(foreign_a, dtype=np.float32).reshape(shape),
        foreign_b=np.asarray(foreign_b, dtype=np.float32).reshape(shape),
        foreign_ok=np.asarray(foreign_ok, dtype=bool),
        foreign_compound=np.asarray(foreign_compounds, dtype=str),
        support_plates=support_plates,
        held_a_plates=held_a_plates,
        held_b_plates=held_b_plates,
    )


def build_bundles(rows: dict[str, Any], *, include_foreign: bool) -> dict[int, dict[str, DualExamples]]:
    return {
        budget: {
            split: build_dual_examples(values, budget, include_foreign=include_foreign)
            for split, values in rows.items()
        }
        for budget in BUDGETS
    }


def manifest_rows(bundles: dict[int, dict[str, DualExamples]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for budget in BUDGETS:
        for split, examples in bundles[budget].items():
            for i in range(len(examples)):
                output.append({
                    "dataset": "BBBC047",
                    "split": split,
                    "budget": budget,
                    "protocol_seed": PROTOCOL_SEED,
                    "compound_id": str(examples.compound[i]),
                    "dose": str(examples.dose[i]),
                    "condition": str(examples.condition[i]),
                    "support_rep_ids": "|".join(examples.support_plates[i]),
                    "held_A_rep_id": "|".join(examples.held_a_plates[i]),
                    "held_B_rep_id": "|".join(examples.held_b_plates[i]),
                })
    return output


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (value.item() if isinstance(value, np.generic) else value) for key, value in row.items()})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def protocol_hash(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return sha256_file(path)


def _first(row: Mapping[str, str], names: Iterable[str]) -> str:
    for name in names:
        if name in row and str(row[name]).strip() != "":
            return str(row[name]).strip()
    return ""


def _plate_list(value: str) -> str:
    return "|".join(part.strip() for part in value.replace(",", "|").split("|") if part.strip())


def canonical_dose(value: Any) -> str:
    text = str(value).strip()
    try:
        return f"{float(text):.2f}"
    except ValueError:
        return text


def source_manifest_key(row: Mapping[str, str]) -> tuple[str, int, str, str, str, str, str]:
    split = _first(row, ("split", "outer_split"))
    budget_raw = _first(row, ("budget", "repeat_budget"))
    compound = _first(row, ("compound_id", "compound", "smiles"))
    dose = _first(row, ("dose", "nominal_dose"))
    support = _plate_list(_first(row, ("support_rep_ids", "support_plates", "support_rep_id")))
    held_a = _first(row, ("held_A_rep_id", "held_a_rep_id", "held_rep_a", "held_repeat_A", "held_A"))
    held_b = _first(row, ("held_B_rep_id", "held_b_rep_id", "held_rep_b", "held_repeat_B", "held_B"))
    if not held_a or not held_b:
        packed = _first(row, ("held_rep_id", "held_rep_ids", "held_plates", "held_repeat_ids"))
        packed_parts = [part.strip() for part in packed.replace(",", "|").split("|") if part.strip()]
        if len(packed_parts) >= 2:
            held_a, held_b = packed_parts[0], packed_parts[1]
    if not all((split, budget_raw, compound, dose, support, held_a, held_b)):
        raise RuntimeError(f"dual manifest row lacks required fields: {dict(row)}")
    try:
        budget = int(float(budget_raw))
    except ValueError as exc:
        raise RuntimeError(f"invalid budget in dual manifest row: {dict(row)}") from exc
    dose = canonical_dose(dose)
    return split, budget, compound, dose, support, _plate_list(held_a), _plate_list(held_b)


def load_dual_reference(reference_root: Path, local_manifest: list[dict[str, Any]]) -> dict[str, Any]:
    if not reference_root.is_dir():
        raise RuntimeError(f"dual-reference-root is not a directory: {reference_root}")
    eligibility_path = reference_root / "DUAL_REPEAT_ELIGIBILITY.json"
    if not eligibility_path.is_file():
        fallback = reference_root / "PREPARE_COMPLETE.json"
        if fallback.is_file() and read_json(fallback).get("dual_repeat_eligible") is True:
            eligibility_path = fallback
    if not eligibility_path.is_file():
        raise RuntimeError("dual-repeat eligibility marker is absent")
    eligibility = read_json(eligibility_path)
    if eligibility.get("dual_repeat_eligible") is not True:
        raise RuntimeError("dual-repeat eligibility marker does not assert dual_repeat_eligible=true")
    if eligibility.get("test_loaded") or eligibility.get("test_values_opened"):
        raise RuntimeError("dual-repeat eligibility marker was produced after test access")
    source_path = reference_root / "MANIFEST.csv"
    if not source_path.is_file():
        raise RuntimeError(f"dual-repeat manifest is absent: {source_path}")
    source_rows = read_csv(source_path)
    expected = {
        (str(row["split"]), int(row["budget"]), str(row["compound_id"]), canonical_dose(row["dose"]),
         _plate_list(str(row["support_rep_ids"])), _plate_list(str(row["held_A_rep_id"])), _plate_list(str(row["held_B_rep_id"])))
        for row in local_manifest
    }
    observed: list[tuple[str, int, str, str, str, str, str]] = []
    for row in source_rows:
        # The source may carry a separate test assignment, but test rows are
        # intentionally irrelevant to prepare/fit and are not parsed here.
        if _first(row, ("split", "outer_split")) not in ("train", "valid"):
            continue
        dataset = _first(row, ("dataset", "dataset_id"))
        if dataset and dataset != "BBBC047":
            raise RuntimeError(f"dual manifest dataset mismatch: {dataset}")
        seed_text = _first(row, ("protocol_seed", "seed"))
        if seed_text and int(float(seed_text)) != PROTOCOL_SEED:
            raise RuntimeError(f"dual manifest protocol seed mismatch: {seed_text}")
        key = source_manifest_key(row)
        if key[1] in BUDGETS:
            observed.append(key)
    if len(observed) != len(set(observed)):
        raise RuntimeError("dual-repeat source manifest contains duplicate train/valid keys")
    missing = expected - set(observed)
    extra = set(observed) - expected
    if missing or extra:
        raise RuntimeError(f"dual-repeat manifest mismatch: missing={len(missing)} extra={len(extra)}")
    marker_hash = eligibility.get("manifest_sha256")
    source_hash = sha256_file(source_path)
    if marker_hash and marker_hash != source_hash:
        raise RuntimeError("eligibility marker manifest_sha256 does not match MANIFEST.csv")
    return {
        "root": str(reference_root),
        "eligibility_marker": str(eligibility_path),
        "eligibility_sha256": sha256_file(eligibility_path),
        "manifest": str(source_path),
        "manifest_sha256": source_hash,
        "rows": len(observed),
    }


def rank_ledger_from_targets(rank_source_root: Path, *, smoke: bool = False, synthetic_targets: dict[int, np.ndarray] | None = None) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for budget in BUDGETS:
        if synthetic_targets is not None:
            cfra = np.asarray(synthetic_targets[budget], dtype=np.float64)
            path_text = "synthetic-smoke"
            sha = hashlib.sha256(cfra.astype(np.float32).tobytes()).hexdigest()
        else:
            path = rank_source_root / f"budget{budget}" / "training_targets.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as values:
                if "cfra" not in values.files:
                    raise RuntimeError(f"{path} has no cfra target")
                cfra = np.asarray(values["cfra"], dtype=np.float64)
            path_text = str(path)
            sha = sha256_file(path)
        if cfra.ndim != 2 or cfra.shape[1] != CP_DIM or len(cfra) < 2:
            raise RuntimeError(f"invalid CFRA target shape for budget {budget}: {cfra.shape}")
        centered = cfra - cfra.mean(axis=0, keepdims=True)
        covariance = (centered.T @ centered) / float(len(cfra) - 1)
        eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
        tolerance = max(float(eigenvalues[0]) * 1e-12, 1e-15)
        positive = eigenvalues[eigenvalues > tolerance]
        if len(positive) == 0:
            raise RuntimeError(f"no positive CFRA covariance eigenvalues for budget {budget}")
        probabilities = positive / positive.sum()
        effective = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
        frozen = int(np.floor(effective + 0.5))
        if frozen < 1:
            frozen = 1
        records[str(budget)] = {
            "source_path": path_text,
            "source_sha256": sha,
            "n_train_compounds": int(cfra.shape[0]),
            "target_dim": int(cfra.shape[1]),
            "n_positive_eigenvalues": int(len(positive)),
            "effective_rank_entropy": effective,
            "frozen_pca_rank": frozen,
            "cfra_variance_retained_at_frozen_rank": float(positive[:frozen].sum() / positive.sum()),
            "positive_eigenvalues_descending": [float(x) for x in positive.tolist()],
        }
    return {
        "version": VERSION,
        "phase": "rank",
        "train_only": True,
        "rank_rule": "floor(effective_rank_entropy + 0.5)",
        "covariance_definition": "(R - mean(R))^T (R - mean(R)) / (n_train - 1)",
        "budgets": list(BUDGETS),
        "records": records,
        "test_loaded": False,
        "test_values_opened": False,
        "test_used_for_selection": False,
        "status": "PASS",
    }


def rank_phase(args: argparse.Namespace, root: Path) -> None:
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing rank output root: {root}")
    if args.smoke:
        smoke_rows = synthetic_rows(("train",))
        smoke_bundles = build_bundles(smoke_rows, include_foreign=False)
        ledger = rank_ledger_from_targets(
            args.rank_source_root,
            smoke=True,
            synthetic_targets={budget: smoke_bundles[budget]["train"].support for budget in BUDGETS},
        )
    else:
        ledger = rank_ledger_from_targets(args.rank_source_root)
        observed = {budget: int(ledger["records"][str(budget)]["frozen_pca_rank"]) for budget in BUDGETS}
        if observed != FROZEN_RANKS:
            raise RuntimeError(f"train-only rank changed from frozen ledger: {observed} != {FROZEN_RANKS}")
        observed_hashes = {budget: ledger["records"][str(budget)]["source_sha256"] for budget in BUDGETS}
        if observed_hashes != FROZEN_TARGET_HASHES:
            raise RuntimeError("train-only rank source hash changed from frozen ledger")
    ledger["protocol_sha256"] = protocol_hash(args.protocol_file)
    ledger["rank_source_root"] = "synthetic-smoke" if args.smoke else str(args.rank_source_root)
    root.mkdir(parents=True)
    json_dump(root / "RANK_FREEZE.json", ledger)
    print(json.dumps({"phase": "rank", "root": str(root), "frozen_ranks": {b: ledger["records"][str(b)]["frozen_pca_rank"] for b in BUDGETS}}, indent=2))


def load_rank_freeze(args: argparse.Namespace, *, root: Path | None = None, smoke: bool = False) -> dict[str, Any]:
    candidates = []
    if root is not None:
        candidates.append(root / "RANK_FREEZE.json")
    candidates.extend((args.rank_freeze, HERE / "RANK_FREEZE.json"))
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise RuntimeError("rank freeze ledger is absent")
    ledger = read_json(path)
    if ledger.get("version") != VERSION or ledger.get("protocol_sha256") not in (None, protocol_hash(args.protocol_file)):
        raise RuntimeError("rank freeze does not match current protocol")
    if ledger.get("train_only") is not True or ledger.get("test_loaded") or ledger.get("test_values_opened"):
        raise RuntimeError("rank freeze is not train-only/test-sealed")
    for budget in BUDGETS:
        record = ledger.get("records", {}).get(str(budget), {})
        frozen = int(record.get("frozen_pca_rank", 0))
        if frozen < 1:
            raise RuntimeError(f"invalid frozen rank for budget {budget}")
        if not smoke and frozen != FROZEN_RANKS[budget]:
            raise RuntimeError(f"frozen rank changed for budget {budget}: {frozen} != {FROZEN_RANKS[budget]}")
        if not smoke and record.get("source_sha256") != FROZEN_TARGET_HASHES[budget]:
            raise RuntimeError(f"rank source hash changed for budget {budget}")
    ledger["_path"] = str(path)
    return ledger


def pca_reconstruct(raw_target: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    values = np.asarray(raw_target, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != CP_DIM:
        raise RuntimeError(f"invalid Raw target shape for PCA: {values.shape}")
    mean = values.mean(axis=0)
    centered = values - mean
    covariance = (centered.T @ centered) / float(max(len(values) - 1, 1))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    if rank > eigenvectors.shape[1]:
        raise RuntimeError(f"PCA rank {rank} exceeds target dimension")
    components = eigenvectors[:, :rank].astype(np.float32)
    reconstruction = mean + (centered @ components.astype(np.float64)) @ components.astype(np.float64).T
    total = float(eigenvalues.sum())
    retained = float(eigenvalues[:rank].sum() / total) if total > 0 else 1.0
    return reconstruction.astype(np.float32), mean.astype(np.float32), components, eigenvalues.astype(np.float32), retained


def load_virtual_splits(model_h5: Path, aggregate_npz: Path, split_lock: Path, splits: Iterable[str]) -> dict[str, Any]:
    """Read exactly the requested virtual splits; never touch test during fit."""
    requested = tuple(splits)
    lock = read_json(split_lock)
    if not model_h5.is_file() or not aggregate_npz.is_file():
        raise FileNotFoundError("model H5 and aggregate NPZ are required")
    with h5py.File(model_h5, "r") as handle:
        all_smiles = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in handle["canonical_smiles"][:]]
        if len(all_smiles) != len(set(all_smiles)):
            raise RuntimeError("duplicate H5 canonical_smiles")
        index = {value: i for i, value in enumerate(all_smiles)}
        output: dict[str, Any] = {}
        with np.load(aggregate_npz, allow_pickle=False) as aggregate:
            for split in requested:
                names = [str(x) for x in lock[f"{split}_smiles"]]
                if len(names) != EXPECTED[split] or len(set(names)) != len(names):
                    raise RuntimeError(f"invalid {split} split lock")
                try:
                    rows = np.asarray([index[name] for name in names], dtype=np.int64)
                except KeyError as exc:
                    raise RuntimeError(f"{split} compound missing in H5: {exc}") from exc
                order = np.argsort(rows)
                inverse = np.argsort(order)
                sorted_rows = rows[order]
                control = handle["control_CP"][sorted_rows].astype(np.float32)[inverse]
                target = handle["target_CP"][sorted_rows].astype(np.float32)[inverse]
                delta = target - control
                agg_names = [str(x) for x in aggregate[f"{split}_smiles"]]
                agg_index = {name: i for i, name in enumerate(agg_names)}
                if set(agg_index) != set(names):
                    raise RuntimeError(f"aggregate/H5 {split} compound mismatch")
                agg_rows = np.asarray([agg_index[name] for name in names], dtype=np.int64)
                aggregate_delta = aggregate[f"{split}_cp"][agg_rows].astype(np.float32)
                if float(np.max(np.abs(delta - aggregate_delta))) > 1e-5:
                    raise RuntimeError(f"aggregate/H5 {split} delta mismatch")
                output[split] = LEGACY.PredictorSplit(
                    compound=np.asarray(names, dtype=str),
                    dose=np.asarray(["" for _ in names], dtype=str),
                    control=control,
                    fingerprint=LEGACY.VIRTUAL.fingerprints(names),
                    target=delta,
                )
    return output


def subset_virtual(base: Any, examples: DualExamples, target: np.ndarray) -> Any:
    index = {str(value): i for i, value in enumerate(base.compound if hasattr(base, "compound") else base.smiles)}
    rows = np.asarray([index[str(x)] for x in examples.compound], dtype=np.int64)
    if len(rows) != len(set(examples.compound.tolist())):
        raise RuntimeError("one predictor row per compound is required")
    return LEGACY.PredictorSplit(
        compound=examples.compound.copy(),
        dose=examples.dose.copy(),
        control=base.control[rows].astype(np.float32),
        fingerprint=base.fingerprint[rows].astype(np.float32),
        target=np.asarray(target, dtype=np.float32),
    )


def synthetic_rows(splits: Iterable[str]) -> dict[str, Any]:
    """Small deterministic BBBC047-shaped data for smoke-only execution."""
    output: dict[str, Any] = {}
    sizes = {"train": 12, "valid": 8, "test": 8}
    split_offset = {"train": 0, "valid": 1000, "test": 2000}
    for split in splits:
        rng = np.random.default_rng(PROTOCOL_SEED + split_offset[split])
        n = sizes[split]
        compounds = np.asarray([f"SMOKE_{split}_{i:03d}" for i in range(n)], dtype=str)
        doses = np.asarray(["1.00"] * n, dtype=str)
        plates = np.asarray([f"P{p}" for p in range(5) for _ in range(n)], dtype=str)
        row_compounds = np.tile(compounds, 5)
        row_doses = np.tile(doses, 5)
        latent = rng.normal(0, 0.3, size=(n, 8))
        loading = rng.normal(0, 0.2, size=(8, CP_DIM))
        baseline = latent @ loading + rng.normal(0, 0.05, size=(n, CP_DIM))
        deltas = np.vstack([baseline + rng.normal(0, 0.08, size=(n, CP_DIM)) for _ in range(5)]).astype(np.float32)
        output[split] = LEGACY.Rows("BBBC047", "CP", split, row_compounds, row_doses, plates, deltas)
    return output


def synthetic_virtual(rows: dict[str, Any]) -> dict[str, SyntheticVirtual]:
    output: dict[str, SyntheticVirtual] = {}
    for split, values in rows.items():
        names = np.asarray(sorted(set(values.compound.tolist())), dtype=str)
        controls = np.zeros((len(names), CP_DIM), dtype=np.float32)
        fingerprints = np.zeros((len(names), FP_DIM), dtype=np.float32)
        for i, name in enumerate(names):
            rng = np.random.default_rng(LEGACY.stable_int(PROTOCOL_SEED, f"smoke-input|{name}") % (2**63 - 1))
            controls[i] = rng.normal(0, 1, CP_DIM).astype(np.float32)
            fingerprints[i] = (rng.random(FP_DIM) > 0.985).astype(np.float32)
        output[split] = SyntheticVirtual(names, controls, fingerprints)
    return output


def preflight_audit(args: argparse.Namespace, root: Path, rows: dict[str, Any], bundles: dict[int, dict[str, DualExamples]], source: dict[str, Any] | None) -> dict[str, Any]:
    split_sets = {split: set(values.compound.tolist()) for split, values in rows.items()}
    source_hashes = {
        "protocol": protocol_hash(args.protocol_file),
        "split_lock": None if args.smoke else sha256_file(args.split_lock),
        "train_cp_plate_rows": None if args.smoke else sha256_file(args.rows_root / "train_cp_plate_rows.npz"),
        "valid_cp_plate_rows": None if args.smoke else sha256_file(args.rows_root / "valid_cp_plate_rows.npz"),
        "legacy_target_replacement": sha256_file(LEGACY_PATH),
    }
    return {
        "version": VERSION,
        "phase": "prepare",
        "protocol_sha256": source_hashes["protocol"],
        "protocol_seed": PROTOCOL_SEED,
        "budgets": list(BUDGETS),
        "student_methods": list(METHODS),
        "predictor_seeds": list(PREDICTOR_SEEDS),
        "outer_compounds": {split: len(values) for split, values in split_sets.items()},
        "split_intersections": {"train_valid": len(split_sets.get("train", set()) & split_sets.get("valid", set())), "train_test": 0, "valid_test": 0},
        "eligible_compounds": {f"budget{budget}": {split: len(set(bundles[budget][split].compound.tolist())) for split in rows} for budget in BUDGETS},
        "manifest_rows": len(manifest_rows(bundles)),
        "dual_repeat_eligible": bool(source is not None or args.smoke),
        "dual_reference": source,
        "source_hashes": source_hashes,
        "rank_freeze": str(args.rank_freeze),
        "test_loaded": False,
        "test_values_opened": False,
        "test_used_for_selection": False,
        "bootstrap_unit": "compound",
        "bootstrap_rounds": args.bootstrap_rounds,
        "status": "PASS",
    }


def require_prepare(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    marker = root / "PREPARE_COMPLETE.json"
    if not marker.is_file():
        raise RuntimeError(f"missing prepare marker: {marker}")
    audit = read_json(marker)
    if audit.get("version") != VERSION or audit.get("protocol_sha256") != protocol_hash(args.protocol_file):
        raise RuntimeError("prepare marker does not match current protocol")
    if audit.get("dual_repeat_eligible") is not True:
        raise RuntimeError("prepare marker is not dual-repeat eligible")
    if audit.get("test_loaded") or audit.get("test_values_opened"):
        raise RuntimeError("prepare marker claims test was opened")
    return audit


def dual_target_metrics(prediction: np.ndarray, examples: DualExamples) -> dict[str, np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float32)
    pcc_a = LEGACY.rowwise_corr(pred, examples.held_a)
    pcc_b = LEGACY.rowwise_corr(pred, examples.held_b)
    z_a = LEGACY.fisher_z(pcc_a)
    z_b = LEGACY.fisher_z(pcc_b)
    foreign_a = np.full(len(examples), np.nan, dtype=float)
    foreign_b = np.full(len(examples), np.nan, dtype=float)
    if np.any(examples.foreign_ok):
        ix = examples.foreign_ok
        foreign_a[ix] = LEGACY.rowwise_corr(pred[ix], examples.foreign_a[ix])
        foreign_b[ix] = LEGACY.rowwise_corr(pred[ix], examples.foreign_b[ix])
    fz_a = LEGACY.fisher_z(foreign_a)
    fz_b = LEGACY.fisher_z(foreign_b)
    norm_pred = np.linalg.norm(pred.astype(np.float64), axis=1)
    norm_a = np.linalg.norm(examples.held_a.astype(np.float64), axis=1)
    norm_b = np.linalg.norm(examples.held_b.astype(np.float64), axis=1)
    ratio_a = np.divide(norm_pred, norm_a, out=np.full(len(pred), np.nan), where=norm_a > 1e-12)
    ratio_b = np.divide(norm_pred, norm_b, out=np.full(len(pred), np.nan), where=norm_b > 1e-12)
    return {
        "held_pcc_A": pcc_a,
        "held_pcc_B": pcc_b,
        "A_A": z_a,
        "A_B": z_b,
        "worst_A": np.minimum(z_a, z_b),
        "held_pcc_dual": np.nanmean(np.stack([pcc_a, pcc_b]), axis=0),
        "foreign_pcc_A": foreign_a,
        "foreign_pcc_B": foreign_b,
        "E_A": z_a - fz_a,
        "E_B": z_b - fz_b,
        "E_dual": np.nanmean(np.stack([z_a - fz_a, z_b - fz_b]), axis=0),
        "norm_ratio_A": ratio_a,
        "norm_ratio_B": ratio_b,
        "norm_ratio_dual": np.nanmean(np.stack([ratio_a, ratio_b]), axis=0),
    }


def fit_phase(args: argparse.Namespace, root: Path) -> None:
    require_prepare(args, root)
    if (root / "FIT_COMPLETE.json").exists() or (root / "TEST_COMPLETE.json").exists():
        raise FileExistsError("fit/test marker already exists; use a fresh versioned output root")
    rank_freeze = load_rank_freeze(args, root=root, smoke=args.smoke)
    if args.smoke:
        rows = synthetic_rows(("train", "valid"))
        virtual = synthetic_virtual(rows)
    else:
        rows = load_rows(args.rows_root, ("train", "valid"))
        assert_lock(rows, args.split_lock)
        virtual = load_virtual_splits(args.model_h5, args.aggregate_npz, args.split_lock, ("train", "valid"))
    bundles = build_bundles(rows, include_foreign=False)
    local_manifest_hash = sha256_file(root / "MANIFEST.csv")
    if local_manifest_hash != read_json(root / "PREPARE_COMPLETE.json").get("manifest_sha256"):
        raise RuntimeError("manifest hash changed between prepare and fit")
    device = device_for(args.device)
    all_logs: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    target_hashes: dict[str, Any] = {}
    pca_model_hashes: dict[str, str] = {}
    checkpoint_hashes: dict[str, str] = {}
    for budget in BUDGETS:
        train_examples = bundles[budget]["train"]
        valid_examples = bundles[budget]["valid"]
        cfra_target, teacher_meta, assignment = LEGACY.build_oof_cfra(rows, train_examples, budget, args, device)
        raw_target = train_examples.support.astype(np.float32)
        rank = int(rank_freeze["records"][str(budget)]["frozen_pca_rank"])
        pca_target, pca_mean, pca_components, pca_eigenvalues, raw_retained = pca_reconstruct(raw_target, rank)
        if not (raw_target.shape == cfra_target.shape == pca_target.shape):
            raise RuntimeError(f"target shape mismatch for budget {budget}")
        if not np.isfinite(pca_target).all() or not np.isfinite(cfra_target).all():
            raise RuntimeError(f"non-finite target for budget {budget}")
        if float(np.max(np.abs(pca_target.mean(axis=0) - raw_target.mean(axis=0)))) > 1e-4:
            raise RuntimeError(f"Raw-PCA reconstruction does not preserve Raw target mean for budget {budget}")
        budget_dir = root / f"budget{budget}"
        budget_dir.mkdir(parents=True, exist_ok=False)
        target_path = budget_dir / "training_targets.npz"
        np.savez_compressed(target_path, compound=train_examples.compound, dose=train_examples.dose, raw_mean=raw_target, raw_pca=pca_target, cfra=cfra_target)
        pca_model_path = budget_dir / "raw_pca_model.npz"
        np.savez_compressed(pca_model_path, mean=pca_mean, components=pca_components, eigenvalues=pca_eigenvalues, rank=np.asarray(rank), retained_variance=np.asarray(raw_retained))
        pca_model_hashes[str(budget)] = sha256_file(pca_model_path)
        target_hashes[str(budget)] = {
            "path": str(target_path),
            "sha256": sha256_file(target_path),
            "n_compounds": len(train_examples),
            "rank": rank,
            "cfra_effective_rank": rank_freeze["records"][str(budget)]["effective_rank_entropy"],
            "cfra_variance_retained_at_rank": rank_freeze["records"][str(budget)]["cfra_variance_retained_at_frozen_rank"],
            "raw_pca_variance_retained_at_rank": raw_retained,
            "teacher": teacher_meta,
            "cfra_fold_assignment": assignment,
            "test_values_opened": False,
        }
        json_dump(budget_dir / "CFRA_TEACHER_METADATA.json", {"teacher": teacher_meta, "fold_assignment": assignment})
        stats = LEGACY.common_stats(subset_virtual(virtual["train"], train_examples, raw_target), raw_target)
        np.savez_compressed(budget_dir / "student_normalization.npz", **stats)
        labels = {"raw_mean": raw_target, "raw_pca": pca_target, "cfra": cfra_target}
        for seed in PREDICTOR_SEEDS:
            for method in METHODS:
                train_split = subset_virtual(virtual["train"], train_examples, labels[method])
                valid_target = (valid_examples.held_a + valid_examples.held_b) / 2.0
                valid_split = subset_virtual(virtual["valid"], valid_examples, valid_target)
                model, fitted_stats, logs = LEGACY.fit_student(train_split, valid_split, labels[method], stats, seed, args, device, method)
                all_logs.extend({**row, "budget": budget} for row in logs)
                best_row = min(logs, key=lambda row: float(row["validation_mse"]))
                ckpt_dir = budget_dir / method
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / f"seed{seed}.pt"
                state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                torch.save({"version": VERSION, "budget": budget, "method": method, "seed": seed, "best_epoch": int(best_row["epoch"]), "validation_mse": float(best_row["validation_mse"]), "state_dict": state, "stats": {key: np.asarray(value, dtype=np.float32) for key, value in fitted_stats.items()}}, ckpt_path)
                checkpoint_hashes[f"budget{budget}/{method}/seed{seed}"] = sha256_file(ckpt_path)
                prediction = LEGACY.predict_student(model, valid_split, fitted_stats, args.batch_size, device)
                vm = dual_target_metrics(prediction, valid_examples)
                validation_rows.append({"budget": budget, "method": method, "seed": seed, "n_compounds": len(valid_examples), "best_epoch": int(best_row["epoch"]), "validation_mse": float(best_row["validation_mse"]), **{f"{key}_mean": float(np.nanmean(value)) for key, value in vm.items()}})
    LEGACY.write_csv(root / "TRAINING_LOG.csv", all_logs)
    LEGACY.write_csv(root / "VALIDATION_METRICS.csv", validation_rows)
    json_dump(root / "TARGET_HASHES.json", target_hashes)
    json_dump(root / "CHECKPOINT_HASHES.json", checkpoint_hashes)
    fit_audit = {
        "version": VERSION, "phase": "fit", "protocol_sha256": protocol_hash(args.protocol_file), "manifest_sha256": sha256_file(root / "MANIFEST.csv"), "rank_freeze_sha256": sha256_file(Path(rank_freeze["_path"])), "budgets": list(BUDGETS), "methods": list(METHODS), "predictor_seeds": list(PREDICTOR_SEEDS), "teacher_epochs": args.teacher_epochs, "student_epochs": args.student_epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "device": str(device), "checkpoint_count": len(checkpoint_hashes), "target_hashes": target_hashes, "pca_model_hashes": pca_model_hashes, "checkpoint_hashes": checkpoint_hashes, "test_loaded": False, "test_values_opened": False, "test_used_for_selection": False, "bootstrap_unit": "compound", "bootstrap_rounds": args.bootstrap_rounds, "status": "PASS",
    }
    json_dump(root / "FIT_COMPLETE.json", fit_audit)
    json_dump(root / "SELECTION_FREEZE.json", {"version": VERSION, "selection_frozen": True, "protocol_sha256": protocol_hash(args.protocol_file), "manifest_sha256": sha256_file(root / "MANIFEST.csv"), "rank_freeze_sha256": sha256_file(Path(rank_freeze["_path"])), "pca_model_hashes": pca_model_hashes, "checkpoint_hashes_sha256": sha256_file(root / "CHECKPOINT_HASHES.json"), "checkpoint_count": len(checkpoint_hashes), "all_budgets": list(BUDGETS), "all_methods": list(METHODS), "all_predictor_seeds": list(PREDICTOR_SEEDS), "test_loaded": False, "test_used_for_selection": False, "status": "PASS"})
    print(json.dumps({"phase": "fit", "root": str(root), "checkpoint_count": len(checkpoint_hashes), "test_loaded": False}, indent=2))


def bootstrap_difference(left: np.ndarray, right: np.ndarray, compounds: np.ndarray, seed: int, rounds: int) -> tuple[float, float, float, int]:
    return LEGACY.bootstrap_difference(left, right, compounds, seed, rounds)


def test_phase(args: argparse.Namespace, root: Path) -> None:
    require_prepare(args, root)
    fit_marker = root / "FIT_COMPLETE.json"
    freeze_marker = root / "SELECTION_FREEZE.json"
    if not fit_marker.is_file() or not freeze_marker.is_file():
        raise RuntimeError("test requires FIT_COMPLETE and SELECTION_FREEZE")
    fit_audit = read_json(fit_marker)
    selection = read_json(freeze_marker)
    if fit_audit.get("test_loaded") or fit_audit.get("test_values_opened") or fit_audit.get("test_used_for_selection"):
        raise RuntimeError("fit audit is not test sealed")
    if selection.get("selection_frozen") is not True or selection.get("test_loaded"):
        raise RuntimeError("selection freeze is invalid")
    checkpoint_hashes = read_json(root / "CHECKPOINT_HASHES.json")
    expected_checkpoints = len(BUDGETS) * len(METHODS) * len(PREDICTOR_SEEDS)
    if len(checkpoint_hashes) != expected_checkpoints:
        raise RuntimeError(f"expected {expected_checkpoints} checkpoints, found {len(checkpoint_hashes)}")
    for key, expected_hash in checkpoint_hashes.items():
        budget, method, seed_name = key.split("/")
        path = root / budget / method / f"{seed_name}.pt"
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
    pca_hashes = selection.get("pca_model_hashes", {})
    if set(pca_hashes) != {str(budget) for budget in BUDGETS}:
        raise RuntimeError("selection freeze has incomplete PCA model hash ledger")
    for budget, expected_hash in pca_hashes.items():
        pca_path = root / f"budget{budget}" / "raw_pca_model.npz"
        if not pca_path.is_file() or sha256_file(pca_path) != expected_hash:
            raise RuntimeError(f"PCA model hash mismatch: {pca_path}")
    target_hashes = read_json(root / "TARGET_HASHES.json")
    if set(target_hashes) != {str(budget) for budget in BUDGETS}:
        raise RuntimeError("training target hash ledger is incomplete")
    for budget, record in target_hashes.items():
        target_path = root / f"budget{budget}" / "training_targets.npz"
        if not target_path.is_file() or sha256_file(target_path) != record.get("sha256"):
            raise RuntimeError(f"training target hash mismatch: {target_path}")
    if sha256_file(root / "MANIFEST.csv") != selection.get("manifest_sha256"):
        raise RuntimeError("manifest hash mismatch at test authorization")
    auth = {"version": VERSION, "decision": "TEST-AUTHORIZED", "selection_frozen": True, "protocol_sha256": protocol_hash(args.protocol_file), "manifest_sha256": sha256_file(root / "MANIFEST.csv"), "checkpoint_count": expected_checkpoints, "test_loaded_before_authorization": False, "test_values_opened_before_authorization": False, "test_used_for_selection": False, "status": "AUTHORIZED"}
    json_dump(root / "TEST_AUTHORIZED.json", auth)
    # First test data access occurs only after TEST_AUTHORIZED.json exists.
    if args.smoke:
        rows = synthetic_rows(("test",))
        virtual = synthetic_virtual(rows)
    else:
        rows = load_rows(args.rows_root, ("test",))
        assert_lock(rows, args.split_lock)
        virtual = load_virtual_splits(args.model_h5, args.aggregate_npz, args.split_lock, ("test",))
    bundles = build_bundles(rows, include_foreign=True)
    device = device_for(args.device)
    per_compound: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    prediction_index: list[dict[str, Any]] = []
    cache: dict[tuple[int, str, int], tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    metric_names = ("A_A", "A_B", "worst_A", "held_pcc_A", "held_pcc_B", "held_pcc_dual", "E_A", "E_B", "E_dual", "norm_ratio_A", "norm_ratio_B", "norm_ratio_dual")
    for budget in BUDGETS:
        examples = bundles[budget]["test"]
        test_target = (examples.held_a + examples.held_b) / 2.0
        test_split = subset_virtual(virtual["test"], examples, test_target)
        for seed in PREDICTOR_SEEDS:
            for method in METHODS:
                ckpt_path = root / f"budget{budget}" / method / f"seed{seed}.pt"
                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                model = LEGACY.Student().to(device)
                model.load_state_dict(checkpoint["state_dict"])
                stats = {key: np.asarray(value, dtype=np.float32) for key, value in checkpoint["stats"].items()}
                prediction = LEGACY.predict_student(model, test_split, stats, args.batch_size, device)
                metrics = dual_target_metrics(prediction, examples)
                cache[(budget, method, seed)] = (prediction, metrics)
                pred_path = root / f"budget{budget}" / f"seed{seed}_test_{method}_predictions.npz"
                np.savez_compressed(pred_path, compound=examples.compound, dose=examples.dose, held_A=examples.held_a, held_B=examples.held_b, prediction=prediction, foreign_A=examples.foreign_a, foreign_B=examples.foreign_b, foreign_ok=examples.foreign_ok.astype(np.uint8))
                prediction_index.append({"budget": budget, "method": method, "seed": seed, "path": str(pred_path), "sha256": sha256_file(pred_path), "shape": str(tuple(prediction.shape))})
                for i, compound in enumerate(examples.compound):
                    per_compound.append({"budget": budget, "method": method, "seed": seed, "compound_id": str(compound), "dose": str(examples.dose[i]), **{key: metrics[key][i] for key in metrics}})
        compounds = np.asarray(examples.compound, dtype=str)
        for left_method, right_method, comparison in (("raw_pca", "raw_mean", "raw_pca_minus_raw_mean"), ("raw_pca", "cfra", "raw_pca_minus_cfra")):
            for metric in metric_names:
                seed_points: list[float] = []
                for seed in PREDICTOR_SEEDS:
                    left = cache[(budget, left_method, seed)][1][metric]
                    right = cache[(budget, right_method, seed)][1][metric]
                    point, low, high, n = bootstrap_difference(left, right, compounds, LEGACY.stable_int(PROTOCOL_SEED, f"{VERSION}|test|b{budget}|{comparison}|{metric}|seed{seed}"), args.bootstrap_rounds)
                    seed_points.append(point)
                    contrasts.append({"budget": budget, "comparison": comparison, "metric": metric, "seed": seed, "mean_difference": point, "ci_low": low, "ci_high": high, "n_compounds": n})
                left_pred = np.mean(np.stack([cache[(budget, left_method, seed)][0] for seed in PREDICTOR_SEEDS]), axis=0)
                right_pred = np.mean(np.stack([cache[(budget, right_method, seed)][0] for seed in PREDICTOR_SEEDS]), axis=0)
                left_metric = dual_target_metrics(left_pred, examples)[metric]
                right_metric = dual_target_metrics(right_pred, examples)[metric]
                point, low, high, n = bootstrap_difference(left_metric, right_metric, compounds, LEGACY.stable_int(PROTOCOL_SEED, f"{VERSION}|test|b{budget}|{comparison}|{metric}|seed-averaged"), args.bootstrap_rounds)
                contrasts.append({"budget": budget, "comparison": comparison, "metric": metric, "seed": "seed_averaged", "mean_difference": point, "ci_low": low, "ci_high": high, "n_compounds": n, "positive_seed_count": sum(value > 0 for value in seed_points if np.isfinite(value))})
    LEGACY.write_csv(root / "TEST_PER_COMPOUND.csv", per_compound)
    LEGACY.write_csv(root / "TEST_CONTRASTS.csv", contrasts)
    LEGACY.write_csv(root / "TEST_PREDICTION_INDEX.csv", prediction_index)
    summary: list[dict[str, Any]] = []
    for budget in BUDGETS:
        for comparison in ("raw_pca_minus_raw_mean", "raw_pca_minus_cfra"):
            entries = {str(row["metric"]): row for row in contrasts if row["budget"] == budget and row["comparison"] == comparison and row["seed"] == "seed_averaged"}
            row: dict[str, Any] = {"budget": budget, "comparison": comparison}
            for metric in ("A_A", "A_B", "worst_A", "held_pcc_dual", "E_A", "E_B", "E_dual", "norm_ratio_dual"):
                item = entries[metric]
                row[f"{metric}_delta"] = item["mean_difference"]
                row[f"{metric}_ci_low"] = item["ci_low"]
                row[f"{metric}_ci_high"] = item["ci_high"]
            row["n_compounds"] = entries["A_A"]["n_compounds"]
            summary.append(row)
    LEGACY.write_csv(root / "TEST_SUMMARY.csv", summary)
    test_hashes = {"test_cp_plate_rows": None if args.smoke else sha256_file(args.rows_root / "test_cp_plate_rows.npz"), "model_h5": None if args.smoke else sha256_file(args.model_h5), "aggregate_npz": None if args.smoke else sha256_file(args.aggregate_npz)}
    json_dump(root / "TEST_COMPLETE.json", {"version": VERSION, "phase": "test", "protocol_sha256": protocol_hash(args.protocol_file), "manifest_sha256": sha256_file(root / "MANIFEST.csv"), "test_source_hashes": test_hashes, "test_loaded": True, "test_values_opened": True, "test_used_for_selection": False, "test_loaded_after_authorization": True, "bootstrap_unit": "compound", "bootstrap_rounds": args.bootstrap_rounds, "status": "PASS"})
    print(json.dumps({"phase": "test", "root": str(root), "test_loaded": True, "summary": summary}, indent=2))


def prepare_phase(args: argparse.Namespace, root: Path) -> None:
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {root}")
    if args.smoke:
        rows = synthetic_rows(("train", "valid"))
        source = None
    else:
        rows = load_rows(args.rows_root, ("train", "valid"))
        assert_lock(rows, args.split_lock)
    bundles = build_bundles(rows, include_foreign=False)
    local_manifest = manifest_rows(bundles)
    if not args.smoke:
        source = load_dual_reference(args.dual_reference_root, local_manifest)
    root.mkdir(parents=True)
    write_csv(root / "MANIFEST.csv", local_manifest)
    audit = preflight_audit(args, root, rows, bundles, source)
    audit["manifest_sha256"] = sha256_file(root / "MANIFEST.csv")
    audit["manifest_file"] = str(root / "MANIFEST.csv")
    audit["test_data_paths"] = {"test_cp_plate_rows": str(args.rows_root / "test_cp_plate_rows.npz"), "test_h5": str(args.model_h5)}
    rank_path = args.rank_freeze if args.rank_freeze.is_file() else None
    if rank_path is not None:
        audit["rank_freeze_sha256"] = sha256_file(rank_path)
    json_dump(root / "PREPARE_COMPLETE.json", audit)
    json_dump(root / "RUN_CONFIG.json", {"version": VERSION, "protocol_sha256": audit["protocol_sha256"], "rows_root": str(args.rows_root), "model_h5": str(args.model_h5), "aggregate_npz": str(args.aggregate_npz), "split_lock": str(args.split_lock), "dual_reference_root": None if args.dual_reference_root is None else str(args.dual_reference_root), "rank_freeze": str(args.rank_freeze), "teacher_epochs": args.teacher_epochs, "student_epochs": args.student_epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "bootstrap_rounds": args.bootstrap_rounds, "test_loaded": False})
    print(json.dumps({"phase": "prepare", "root": str(root), "manifest_sha256": audit["manifest_sha256"], "dual_repeat_eligible": audit["dual_repeat_eligible"], "test_loaded": False}, indent=2))


def smoke_phase(args: argparse.Namespace, root: Path | None) -> None:
    temp: tempfile.TemporaryDirectory[str] | None = None
    if root is None:
        temp = tempfile.TemporaryDirectory(prefix="bbbc047_rank_matched_pca_smoke_")
        root = Path(temp.name)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing smoke output root: {root}")
    args.smoke = True
    # A smoke rank ledger is derived from train-only synthetic support targets.
    rows = synthetic_rows(("train",))
    smoke_bundles = build_bundles(rows, include_foreign=False)
    synthetic_targets = {budget: smoke_bundles[budget]["train"].support for budget in BUDGETS}
    ledger = rank_ledger_from_targets(args.rank_source_root, smoke=True, synthetic_targets=synthetic_targets)
    ledger["protocol_sha256"] = protocol_hash(args.protocol_file)
    ledger["rank_source_root"] = "synthetic-smoke"
    # Keep the rank ledger beside (rather than inside) the fresh output root;
    # prepare deliberately refuses to enter an existing directory.
    args.rank_freeze = root.parent / f"{root.name}_RANK_FREEZE.json"
    json_dump(args.rank_freeze, ledger)
    prepare_phase(args, root)
    fit_phase(args, root)
    test_phase(args, root)
    print(json.dumps({"phase": "smoke", "root": str(root), "non_scientific": True}, indent=2))
    if temp is not None:
        temp.cleanup()


def dry_run_phase(args: argparse.Namespace) -> None:
    print(json.dumps({"version": VERSION, "phase": "dry-run", "budgets": list(BUDGETS), "methods": list(METHODS), "predictor_seeds": list(PREDICTOR_SEEDS), "frozen_ranks": FROZEN_RANKS, "teacher_epochs": args.teacher_epochs, "student_epochs": args.student_epochs, "batch_size": args.batch_size, "bootstrap_rounds": args.bootstrap_rounds, "formal_prepare_requires_dual_reference": True, "formal_test_requires_authorization_marker": True, "test_loaded": False}, indent=2))


def main() -> None:
    args = parse_args()
    if args.phase == "dry-run":
        dry_run_phase(args)
        return
    if args.phase == "smoke":
        smoke_phase(args, args.output_root)
        return
    root = args.output_root
    assert root is not None
    if args.phase == "rank":
        rank_phase(args, root)
    elif args.phase == "prepare":
        prepare_phase(args, root)
    elif args.phase == "fit":
        fit_phase(args, root)
    elif args.phase == "test":
        test_phase(args, root)


if __name__ == "__main__":
    main()
