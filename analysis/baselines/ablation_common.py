"""Shared, deterministic data/protocol/evaluation utilities for baseline ablations.

The execution script is intentionally self contained and can be copied to the
remote AIDD project.  This module has no torch dependency so that audits can be
run in a light-weight Python environment.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SEEDS = (3407, 42, 2025)
BETA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
PCA_CANDIDATES = (16, 32, 64, 128, 256, 512)
BOOTSTRAP_N = 2000
PROTOCOL_VERSION = "baseline-ablation-v1-2026-08-30"


def decode_scalar(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def stable_int(seed: int, text: str) -> int:
    raw = f"{int(seed)}|{text}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dose_key(value: Any, dataset: str) -> str:
    """Canonical nominal-dose key used for cross-modality matching.

    BBBC047 stores CP and GE doses as float32 and their decimal encodings differ
    slightly (for example 1.75 versus 1.75475); the frozen protocol therefore
    uses two decimal places for BBBC047.  cpg0004-LINCS retains its prepared
    nominal dose string exactly.
    """
    text = decode_scalar(value).strip()
    if dataset == "BBBC047":
        try:
            return f"{float(text):.2f}"
        except ValueError:
            return text
    return text


@dataclass
class Rows:
    dataset: str
    modality: str
    split: str
    compound: np.ndarray
    dose: np.ndarray
    plate: np.ndarray
    delta: np.ndarray

    def __post_init__(self) -> None:
        self.compound = np.asarray([decode_scalar(x) for x in self.compound], dtype=str)
        self.dose = np.asarray([dose_key(x, self.dataset) for x in self.dose], dtype=str)
        self.plate = np.asarray([decode_scalar(x) for x in self.plate], dtype=str)
        self.delta = np.asarray(self.delta, dtype=np.float32)
        if self.delta.ndim != 2 or len(self.compound) != self.delta.shape[0]:
            raise ValueError(f"invalid {self.dataset}/{self.modality}/{self.split} row shape")
        self._plate_groups: dict[tuple[str, str], dict[str, np.ndarray]] | None = None
        self._plate_means: dict[tuple[str, str], dict[str, np.ndarray]] | None = None

    @property
    def dim(self) -> int:
        return int(self.delta.shape[1])

    @property
    def n_rows(self) -> int:
        return int(self.delta.shape[0])

    def condition_keys(self) -> list[tuple[str, str]]:
        return list(self.plate_groups().keys())

    def plate_groups(self) -> dict[tuple[str, str], dict[str, np.ndarray]]:
        if self._plate_groups is None:
            raw: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
            for i, (compound, dose, plate) in enumerate(zip(self.compound, self.dose, self.plate)):
                raw[(compound, dose)][plate].append(i)
            self._plate_groups = {
                key: {plate: np.asarray(ix, dtype=np.int64) for plate, ix in sorted(plates.items())}
                for key, plates in raw.items()
            }
        return self._plate_groups

    def plate_means(self) -> dict[tuple[str, str], dict[str, np.ndarray]]:
        if self._plate_means is None:
            self._plate_means = {
                key: {plate: np.asarray(self.delta[ix].mean(axis=0), dtype=np.float32) for plate, ix in plates.items()}
                for key, plates in self.plate_groups().items()
            }
        return self._plate_means

    def condition_mean(self, key: tuple[str, str], plates: Sequence[str] | None = None) -> np.ndarray:
        by_plate = self.plate_means()[key]
        names = list(by_plate) if plates is None else list(plates)
        return np.asarray(np.mean([by_plate[p] for p in names], axis=0), dtype=np.float32)


@dataclass
class DatasetBundle:
    dataset: str
    rows: dict[str, dict[str, Rows]]

    def get(self, split: str, modality: str) -> Rows:
        return self.rows[split][modality]


@dataclass
class ExampleSet:
    dataset: str
    modality: str
    split: str
    budget: int
    seed: int
    compound: np.ndarray
    dose: np.ndarray
    condition: np.ndarray
    support: np.ndarray
    target: np.ndarray
    aggregate_target: np.ndarray
    source: np.ndarray | None
    support_plates: list[tuple[str, ...]]
    held_plates: list[tuple[str, ...]]
    foreign: np.ndarray
    foreign_ok: np.ndarray
    foreign_compound: np.ndarray
    foreign_condition: np.ndarray

    def __len__(self) -> int:
        return int(len(self.compound))


def _eligible_condition_keys(
    target: Rows,
    source: Rows | None,
    budget: int,
    require_source: bool,
) -> list[tuple[str, str]]:
    target_groups = target.plate_groups()
    source_keys = set(source.plate_groups()) if source is not None else None
    keys = []
    for key, plates in target_groups.items():
        if len(plates) < budget + 1:
            continue
        if require_source and (source_keys is None or key not in source_keys):
            continue
        keys.append(key)
    return keys


def _ordered_plates(plates: Iterable[str], seed: int, label: str) -> list[str]:
    return sorted(plates, key=lambda p: (stable_int(seed, f"plate|{label}|{p}"), p))


def _select_condition_keys(
    keys: Sequence[tuple[str, str]],
    all_conditions: bool,
    seed: int,
    label: str,
) -> list[tuple[str, str]]:
    ordered = sorted(keys, key=lambda k: (k[0], k[1]))
    if all_conditions:
        return sorted(ordered, key=lambda k: (stable_int(seed, f"condition-order|{label}|{k[0]}|{k[1]}"), k))
    by_compound: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in ordered:
        by_compound[key[0]].append(key)
    chosen = []
    for compound, candidates in sorted(by_compound.items()):
        chosen.append(min(candidates, key=lambda k: (stable_int(seed, f"condition-choice|{label}|{compound}|{k[0]}|{k[1]}"), k)))
    return sorted(chosen, key=lambda k: (k[0], k[1]))


def make_examples(
    target: Rows,
    source: Rows | None,
    budget: int,
    seed: int,
    *,
    all_conditions: bool,
    require_source: bool,
    include_foreign: bool = True,
) -> ExampleSet:
    """Build plate-isolated support/independent-heldout examples.

    The first ``budget`` deterministic plates are support and the next plate is
    held out.  The aggregate target is computed from all legal plates but is
    used as a label only for the training objective-ablation model.  Foreign
    nulls require the same full target plate signature and exact nominal dose.
    """
    target_groups = target.plate_groups()
    target_means = target.plate_means()
    source_means = source.plate_means() if source is not None else None
    keys = _eligible_condition_keys(target, source, budget, require_source)
    label = f"{target.dataset}|{target.modality}|{target.split}|b{budget}"
    chosen = _select_condition_keys(keys, all_conditions, seed, label)

    # Exact-dose/full-plate-set foreign index.  Matching the complete target
    # condition signature prevents using a donor that has a different repeat
    # layout or dose schedule.
    foreign_index: dict[tuple[str, tuple[str, ...]], list[tuple[str, str]]] = defaultdict(list)
    if include_foreign:
        for key, plates in target_groups.items():
            signature = (key[1], tuple(sorted(plates)))
            foreign_index[signature].append(key)
        for signature in foreign_index:
            foreign_index[signature].sort()

    compounds: list[str] = []
    doses: list[str] = []
    conditions: list[str] = []
    supports: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    aggregates: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    support_plates: list[tuple[str, ...]] = []
    held_plates: list[tuple[str, ...]] = []
    foreign_values: list[np.ndarray] = []
    foreign_ok: list[bool] = []
    foreign_compounds: list[str] = []
    foreign_conditions: list[str] = []

    for key in chosen:
        compound, dose = key
        ordered = _ordered_plates(target_groups[key], seed, f"{label}|{compound}|{dose}")
        sp = tuple(ordered[:budget])
        hp = tuple(ordered[budget : budget + 1])
        if len(hp) != 1:
            continue
        support = target.condition_mean(key, sp)
        held = target.condition_mean(key, hp)
        aggregate = target.condition_mean(key)

        donor_key: tuple[str, str] | None = None
        if include_foreign:
            signature = (dose, tuple(sorted(target_groups[key])))
            candidates = [x for x in foreign_index.get(signature, []) if x[0] != compound]
            if candidates:
                donor_key = candidates[stable_int(seed, f"foreign|{label}|{compound}|{dose}|{signature}") % len(candidates)]

        if donor_key is None:
            foreign_value = np.full(target.dim, np.nan, dtype=np.float32)
            donor_compound = ""
            donor_condition = ""
            ok = False
        else:
            foreign_value = target.condition_mean(donor_key, hp)
            donor_compound = donor_key[0]
            donor_condition = f"{donor_key[0]}::{donor_key[1]}"
            ok = True

        source_value: np.ndarray | None = None
        if source_means is not None and key in source_means:
            source_value = np.mean(list(source_means[key].values()), axis=0).astype(np.float32)
        elif require_source:
            continue

        compounds.append(compound)
        doses.append(dose)
        conditions.append(f"{compound}::{dose}")
        supports.append(support)
        targets.append(held)
        aggregates.append(aggregate)
        if source is not None:
            sources.append(source_value if source_value is not None else np.full(source.dim, np.nan, dtype=np.float32))
        support_plates.append(sp)
        held_plates.append(hp)
        foreign_values.append(foreign_value)
        foreign_ok.append(ok)
        foreign_compounds.append(donor_compound)
        foreign_conditions.append(donor_condition)

    source_array: np.ndarray | None
    if source is None:
        source_array = None
    else:
        source_array = np.asarray(sources, dtype=np.float32).reshape(len(compounds), source.dim)
    return ExampleSet(
        dataset=target.dataset,
        modality=target.modality,
        split=target.split,
        budget=budget,
        seed=seed,
        compound=np.asarray(compounds, dtype=str),
        dose=np.asarray(doses, dtype=str),
        condition=np.asarray(conditions, dtype=str),
        support=np.asarray(supports, dtype=np.float32).reshape(len(compounds), target.dim),
        target=np.asarray(targets, dtype=np.float32).reshape(len(compounds), target.dim),
        aggregate_target=np.asarray(aggregates, dtype=np.float32).reshape(len(compounds), target.dim),
        source=source_array,
        support_plates=support_plates,
        held_plates=held_plates,
        foreign=np.asarray(foreign_values, dtype=np.float32).reshape(len(compounds), target.dim),
        foreign_ok=np.asarray(foreign_ok, dtype=bool),
        foreign_compound=np.asarray(foreign_compounds, dtype=str),
        foreign_condition=np.asarray(foreign_conditions, dtype=str),
    )


def load_bundle(
    dataset: str,
    *,
    bbbc_root: Path,
    cpg_cp_path: Path,
    cpg_ge_path: Path,
    cpg_split_lock: Path,
) -> DatasetBundle:
    rows: dict[str, dict[str, Rows]] = {"train": {}, "valid": {}, "test": {}}
    if dataset == "BBBC047":
        for split in rows:
            for modality in ("CP", "GE"):
                path = bbbc_root / f"{split}_{modality.lower()}_plate_rows.npz"
                z = np.load(path, allow_pickle=True)
                rows[split][modality] = Rows(dataset, modality, split, z["smiles"], z["dose"], z["plate"], z["delta"])
    elif dataset == "cpg0004-LINCS":
        cpz = np.load(cpg_cp_path, allow_pickle=True)
        cp_split = np.asarray([decode_scalar(x) for x in cpz["split"]], dtype=str)
        for split in rows:
            mask = cp_split == split
            rows[split]["CP"] = Rows(
                dataset,
                "CP",
                split,
                cpz["compound_id"][mask],
                cpz["dose"][mask],
                cpz["plate"][mask],
                cpz["delta"][mask],
            )
        lock = json.loads(cpg_split_lock.read_text(encoding="utf-8"))
        split_by_compound = {
            compound: split
            for split in ("train", "valid", "test")
            for compound in lock[f"{split}_compounds"]
        }
        gez = np.load(cpg_ge_path, allow_pickle=True)
        ge_compounds = np.asarray([decode_scalar(x) for x in gez["compound_id"]], dtype=str)
        ge_split = np.asarray([split_by_compound.get(x, "") for x in ge_compounds], dtype=str)
        for split in rows:
            mask = ge_split == split
            rows[split]["GE"] = Rows(
                dataset,
                "GE",
                split,
                gez["compound_id"][mask],
                gez["dose"][mask],
                gez["det_plate"][mask],
                gez["delta"][mask],
            )
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    return DatasetBundle(dataset, rows)


def rowwise_corr(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"correlation shape mismatch: {left.shape} versus {right.shape}")
    lc = left - np.nanmean(left, axis=1, keepdims=True)
    rc = right - np.nanmean(right, axis=1, keepdims=True)
    numerator = np.nansum(lc * rc, axis=1)
    denominator = np.sqrt(np.nansum(lc * lc, axis=1) * np.nansum(rc * rc, axis=1))
    result = np.full(left.shape[0], np.nan, dtype=np.float64)
    good = np.isfinite(denominator) & (denominator > 1e-12)
    result[good] = numerator[good] / denominator[good]
    return np.clip(result, -1.0, 1.0)


def fisher_z(value: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    return np.arctanh(np.clip(arr, -0.999999, 0.999999))


def metric_arrays(prediction: np.ndarray, examples: ExampleSet) -> dict[str, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float32)
    same = rowwise_corr(prediction, examples.target)
    baseline = rowwise_corr(examples.support, examples.target)
    foreign = rowwise_corr(prediction[examples.foreign_ok], examples.foreign[examples.foreign_ok]) if np.any(examples.foreign_ok) else np.asarray([], dtype=float)
    foreign_full = np.full(len(examples), np.nan, dtype=float)
    foreign_full[examples.foreign_ok] = foreign
    same_z = fisher_z(same)
    foreign_z = fisher_z(foreign_full)
    return {
        "pcc": same,
        "foreign_pcc": foreign_full,
        "baseline_pcc": baseline,
        "delta_pcc": same - baseline,
        "excess_z": same_z - foreign_z,
        "excess_pcc": same - foreign_full,
        "same_z": same_z,
        "foreign_z": foreign_z,
    }


def _compound_means(values: np.ndarray, compounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    compounds = np.asarray(compounds, dtype=str)
    good = np.isfinite(values)
    if not np.any(good):
        return np.asarray([], dtype=str), np.asarray([], dtype=float)
    grouped: dict[str, list[float]] = defaultdict(list)
    for compound, value in zip(compounds[good], values[good]):
        grouped[str(compound)].append(float(value))
    ids = np.asarray(sorted(grouped), dtype=str)
    means = np.asarray([np.mean(grouped[x]) for x in ids], dtype=float)
    return ids, means


def bootstrap_mean(values: np.ndarray, compounds: np.ndarray, seed: int, n_boot: int = BOOTSTRAP_N) -> tuple[float, float, float, int]:
    ids, means = _compound_means(values, compounds)
    if len(means) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    point = float(np.mean(means))
    if len(means) == 1:
        return point, point, point, 1
    rng = np.random.default_rng(stable_int(seed, "bootstrap") % (2**63 - 1))
    draws = rng.integers(0, len(means), size=(int(n_boot), len(means)))
    boot = means[draws].mean(axis=1)
    return point, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)), len(means)


def bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    compounds: np.ndarray,
    seed: int,
    n_boot: int = BOOTSTRAP_N,
) -> tuple[float, float, float, int]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    good = np.isfinite(left) & np.isfinite(right)
    if not np.any(good):
        return float("nan"), float("nan"), float("nan"), 0
    grouped: dict[str, list[float]] = defaultdict(list)
    for compound, value in zip(np.asarray(compounds, dtype=str)[good], (left - right)[good]):
        grouped[str(compound)].append(float(value))
    ids = sorted(grouped)
    means = np.asarray([np.mean(grouped[x]) for x in ids], dtype=float)
    point = float(np.mean(means))
    if len(means) == 1:
        return point, point, point, 1
    rng = np.random.default_rng(stable_int(seed, "paired-bootstrap") % (2**63 - 1))
    draws = rng.integers(0, len(means), size=(int(n_boot), len(means)))
    boot = means[draws].mean(axis=1)
    return point, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)), len(means)


def summarize_metrics(
    method: str,
    prediction: np.ndarray,
    examples: ExampleSet,
    seed: int,
    split: str,
    *,
    bootstrap_n: int = BOOTSTRAP_N,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metrics = metric_arrays(prediction, examples)
    row: dict[str, Any] = {
        "dataset": examples.dataset,
        "target_modality": examples.modality,
        "budget": examples.budget,
        "seed": seed,
        "split": split,
        "method": method,
        "n_pairs": len(examples),
        "n_foreign": int(np.sum(examples.foreign_ok)),
    }
    for key in ("excess_z", "pcc", "foreign_pcc", "delta_pcc", "excess_pcc"):
        point, low, high, n_compounds = bootstrap_mean(metrics[key], examples.compound, seed + stable_int(seed, method + key), bootstrap_n)
        row[f"{key}_mean"] = point
        row[f"{key}_ci_low"] = low
        row[f"{key}_ci_high"] = high
        row[f"{key}_n_compounds"] = n_compounds
    return row, metrics


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        fieldnames = fields
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (np.generic,)):
                    value = value.item()
                clean[key] = value
            writer.writerow(clean)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def example_manifest(examples: ExampleSet) -> list[dict[str, Any]]:
    rows = []
    for i in range(len(examples)):
        rows.append(
            {
                "dataset": examples.dataset,
                "target_modality": examples.modality,
                "split": examples.split,
                "budget": examples.budget,
                "seed": examples.seed,
                "compound": examples.compound[i],
                "dose": examples.dose[i],
                "condition": examples.condition[i],
                "support_plates": "|".join(examples.support_plates[i]),
                "held_plates": "|".join(examples.held_plates[i]),
                "foreign_ok": int(examples.foreign_ok[i]),
                "foreign_compound": examples.foreign_compound[i],
                "foreign_condition": examples.foreign_condition[i],
                "source_available": int(examples.source is not None and np.all(np.isfinite(examples.source[i]))),
            }
        )
    return rows


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def setting_id(dataset: str, modality: str, budget: int) -> str:
    return f"{dataset}__{modality}__budget{budget}"


def all_settings() -> list[dict[str, Any]]:
    return [
        {"dataset": "BBBC047", "target_modality": "CP", "source_modality": "GE", "budget": 1, "label": "1R"},
        {"dataset": "BBBC047", "target_modality": "CP", "source_modality": "GE", "budget": 2, "label": "2R"},
        {"dataset": "BBBC047", "target_modality": "CP", "source_modality": "GE", "budget": 3, "label": "3R"},
        {"dataset": "BBBC047", "target_modality": "GE", "source_modality": "CP", "budget": 1, "label": "1R"},
        {"dataset": "BBBC047", "target_modality": "GE", "source_modality": "CP", "budget": 2, "label": "2R"},
        {"dataset": "cpg0004-LINCS", "target_modality": "CP", "source_modality": "GE", "budget": 1, "label": "1R"},
        {"dataset": "cpg0004-LINCS", "target_modality": "CP", "source_modality": "GE", "budget": 2, "label": "2R"},
        {"dataset": "cpg0004-LINCS", "target_modality": "CP", "source_modality": "GE", "budget": 3, "label": "3R"},
    ]

