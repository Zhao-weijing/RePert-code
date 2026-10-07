#!/usr/bin/env python3
"""Locked confirmation evaluator for the BBBC047 CFRA decomposition experiment.

The fit phase is deliberately separate from this module.  ``run_confirm``
validates the PREPARE/FIT/SELECTION ledgers, writes
``CONFIRMATION_AUTHORIZED.json``, and only then opens confirmation rows,
virtual inputs, held repeats, or frozen foreign targets.

Fit artifact contract
---------------------
The preferred contract is ``MODEL_ARTIFACTS.json`` (or
``MODEL_ARTIFACT_INDEX.json``) at the experiment root.  It may use either of
these equivalent forms for each budget and seed::

    {"budget1": {"M0": {"seed3407": "budget1/M0/seed3407.pt",
                         "seed42":  {"path": "..."}},
                 "M2": {"seed3407": ["..._a.pt", "..._b.pt"]},
                 "M3": {"seed3407": {"branches": ["...pca.pt", "...resid.pt"]}},
                 "M4": {"seed3407": {"branches": ["...cfra.pt", "...resid.pt"]}}}}

For formal outputs, every path is relative to the experiment root unless it
is absolute.  A checkpoint is a ``torch.save`` dictionary containing a
``state_dict`` for the protocol Student and ``stats`` containing the
train-only ``control_mean``, ``control_scale``, ``target_mean`` and
``target_scale`` arrays.  M2/M3/M4 branches are predicted independently and
combined in original CP space; alpha is therefore fixed to one.

The fallback resolver recognizes ``budget{b}/{M0..M4}/seed{seed}.pt`` and
``seed{seed}_{a,b,pca,residual,cfra}.pt`` names, but formal runs should use
the explicit index so an ambiguous artifact cannot silently pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
EXPERIMENTS = Path(__file__).resolve().parents[4] / "analysis"
PROTOCOL = HERE / "PROTOCOL.md"
PREPARE = (Path(__file__).resolve().parents[4] / "analysis/predictor_supervision/bbbc047/support/prepare_confirmation.py")
spec = importlib.util.spec_from_file_location("bbbc047_confirmation_prepare", PREPARE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"unable to import {PREPARE}")
PREP = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = PREP
spec.loader.exec_module(PREP)
SRC = PREP.SRC
LEGACY = SRC.LEGACY


VERSION = "BBBC047-CFRA-Decomposition-confirmation-v2-2026-09-13"
METHODS = ("M0", "M1", "M2", "M3", "M4")
METHOD_LABELS = {
    "M0": "Raw Student",
    "M1": "CFRA Student",
    "M2": "Raw-2x",
    "M3": "PCA-Decomposition",
    "M4": "CFRA-Decomposition",
}
# ``run_fit.py`` uses descriptive method names in its on-disk ledgers while
# the confirmation tables use the short protocol names.  Keep this alias
# table in one place so that a fit produced by either revision is accepted
# only when it has the same branch semantics.
METHOD_ALIASES = {
    "M0": ("M0", "M0_raw"),
    "M1": ("M1", "M1_cfra"),
    "M2": ("M2", "M2_raw_2x"),
    "M3": ("M3", "M3_pca_decomp"),
    "M4": ("M4", "M4_cfra_decomp"),
}
EXPECTED_BRANCHES = {"M0": 1, "M1": 1, "M2": 2, "M3": 2, "M4": 2}
BUDGETS = (1, 2, 3)
PRIMARY = (1, 2)
SEEDS = tuple(PREP.MASTER_SEEDS)
CP_DIM = 775
FP_DIM = 2048

DEFAULT_ROWS = PREP.DEFAULT_ROWS
DEFAULT_H5 = DEFAULT_ROWS.parent / "Paired_CP_GE_PlateMedian_v1_model_compat.h5"
DEFAULT_AGG = DEFAULT_ROWS / "molecule_aggregates.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            converted = {
                key: (value.item() if isinstance(value, np.generic) else value)
                for key, value in row.items()
            }
            writer.writerow(converted)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rows-root", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--model-h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--aggregate-npz", type=Path, default=DEFAULT_AGG)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-rounds", type=int, default=10000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.bootstrap_rounds < 1:
        parser.error("bootstrap-rounds must be positive")
    if not args.smoke and args.bootstrap_rounds != 10000:
        parser.error("formal protocol locks bootstrap-rounds to 10000")
    return args


def device_for(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def _same_hash(path: Path, expected: str | None, label: str) -> None:
    if expected and sha256_file(path) != str(expected):
        raise RuntimeError(f"{label} hash mismatch: {path}")


def _require_protocol(audit: Mapping[str, Any], protocol_hash: str, label: str) -> None:
    if audit.get("version") != VERSION:
        raise RuntimeError(f"{label} version mismatch: {audit.get('version')!r}")
    if audit.get("protocol_sha256") != protocol_hash:
        raise RuntimeError(f"{label} protocol hash mismatch")
    if audit.get("test_loaded") or audit.get("test_values_opened") or audit.get("test_used_for_selection"):
        raise RuntimeError(f"{label} is not test-sealed")
    if label == "PREPARE_COMPLETE":
        if audit.get("confirmation_model_scores_computed_in_prepare") is not False or audit.get("confirmation_used_for_training") is not False or audit.get("confirmation_used_for_selection") is not False:
            raise RuntimeError("PREPARE_COMPLETE violates data-custodian-only access")
    elif audit.get("confirmation_loaded") or audit.get("confirmation_values_opened") or audit.get("confirmation_used_for_selection"):
        raise RuntimeError(f"{label} claims confirmation was already opened/used")


def authorize_confirmation(root: Path) -> dict[str, Any]:
    """Validate all pre-confirmation ledgers before any confirmation read."""
    protocol_hash = sha256_file(PROTOCOL)
    prepare_path = root / "PREPARE_COMPLETE.json"
    fit_path = root / "FIT_COMPLETE.json"
    selection_path = root / "SELECTION_FREEZE.json"
    for required in (prepare_path, fit_path, selection_path):
        if not required.is_file():
            raise RuntimeError(f"confirmation requires ledger: {required}")
    prepare = read_json(prepare_path)
    fit = read_json(fit_path)
    selection = read_json(selection_path)
    if fit.get("status") != "PASS" or fit.get("phase") != "fit" or selection.get("status") != "PASS":
        raise RuntimeError("fit/selection ledger is not PASS")
    if fit.get("checkpoint_count") != 120 or selection.get("checkpoint_count") != 120:
        raise RuntimeError("expected 120 branch checkpoints")
    if fit.get("alpha") != 1.0 or selection.get("alpha") != 1.0:
        raise RuntimeError("alpha is not frozen at 1")
    for key in ("split_sha256", "protocol_sha256", "checkpoint_hashes_sha256", "pca_hashes_sha256", "branch_seed_sha256", "artifact_index_sha256"):
        if not fit.get(key) and key not in ("split_sha256",):
            raise RuntimeError(f"missing fit ledger hash: {key}")
    _require_protocol(prepare, protocol_hash, "PREPARE_COMPLETE")
    _require_protocol(fit, protocol_hash, "FIT_COMPLETE")
    if selection.get("version") != VERSION or selection.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("SELECTION_FREEZE does not match current protocol")
    if selection.get("selection_frozen") is not True:
        raise RuntimeError("selection is not frozen")
    if selection.get("test_loaded") or selection.get("confirmation_loaded") or selection.get("test_used_for_selection"):
        raise RuntimeError("selection freeze is not sealed")
    for name in ("COMPOUND_SPLIT.csv", "ANALYSIS_MANIFEST.csv"):
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"missing frozen preparation artifact: {path}")
    _same_hash(root / "COMPOUND_SPLIT.csv", prepare.get("split_sha256") or selection.get("split_sha256"), "split")
    _same_hash(root / "ANALYSIS_MANIFEST.csv", prepare.get("manifest_sha256") or selection.get("manifest_sha256"), "manifest")
    expected_manifest = sha256_file(root / "ANALYSIS_MANIFEST.csv")
    if fit.get("manifest_sha256") not in (None, expected_manifest):
        raise RuntimeError("FIT_COMPLETE manifest hash mismatch")
    if selection.get("manifest_sha256") not in (None, expected_manifest):
        raise RuntimeError("SELECTION_FREEZE manifest hash mismatch")
    # The fit marker and selection marker must carry complete ledgers.  Paths
    # are checked before authorization, but no confirmation values are opened.
    # Resolve the explicit artifact index first.  This is metadata/checkpoint
    # validation only and therefore remains on the safe side of the barrier.
    artifact_index_path = next(
        (root / name for name in ("MODEL_ARTIFACTS.json", "MODEL_ARTIFACT_INDEX.json", "ARTIFACT_INDEX.json") if (root / name).is_file()),
        None,
    )
    artifact_index = _artifact_index(root)
    indexed_paths = _flatten_artifact_paths(artifact_index)
    if artifact_index_path is None:
        raise RuntimeError("missing explicit MODEL_ARTIFACTS.json artifact index")
    expected_artifact_hash = selection.get("artifact_index_sha256") or fit.get("artifact_index_sha256")
    if expected_artifact_hash:
        _same_hash(artifact_index_path, str(expected_artifact_hash), "artifact index")
    checkpoint_ledger = root / "CHECKPOINT_HASHES.json"
    if not checkpoint_ledger.is_file():
        raise RuntimeError(f"missing checkpoint hash ledger: {checkpoint_ledger}")
    checkpoint_hashes = read_json(checkpoint_ledger)
    if not isinstance(checkpoint_hashes, dict) or not checkpoint_hashes:
        raise RuntimeError("empty checkpoint hash ledger")
    indexed_by_resolved = {str(path.resolve()): path for path in indexed_paths}
    matched_paths: set[str] = set()
    for key, expected in checkpoint_hashes.items():
        candidates = [path for path in _checkpoint_key_candidates(root, key) if path.is_file()]
        matches = [path for path in candidates if str(path.resolve()) in indexed_by_resolved]
        if len(matches) != 1:
            # If an older ledger has no artifact index correspondence, fail
            # closed instead of silently accepting an unindexed checkpoint.
            raise RuntimeError(f"checkpoint key is ambiguous or absent from artifact index: {key}")
        path = matches[0]
        resolved = str(path.resolve())
        if resolved in matched_paths:
            raise RuntimeError(f"checkpoint ledger reuses path: {path}")
        if sha256_file(path) != str(expected):
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        matched_paths.add(resolved)
    if matched_paths != set(indexed_by_resolved):
        missing = sorted(set(indexed_by_resolved) - matched_paths)
        extra = sorted(matched_paths - set(indexed_by_resolved))
        raise RuntimeError(f"artifact/checkpoint ledger coverage mismatch; missing={missing[:2]} extra={extra[:2]}")
    if fit.get("checkpoint_hashes") and fit["checkpoint_hashes"] != checkpoint_hashes:
        raise RuntimeError("FIT_COMPLETE checkpoint ledger differs from CHECKPOINT_HASHES.json")
    if selection.get("checkpoint_hashes_sha256"):
        _same_hash(checkpoint_ledger, selection["checkpoint_hashes_sha256"], "checkpoint ledger")
    artifact_index = root / "MODEL_ARTIFACTS.json"
    if not artifact_index.is_file() or not fit.get("artifact_index_sha256") or sha256_file(artifact_index) != fit["artifact_index_sha256"]:
        raise RuntimeError("artifact index hash missing or mismatched")
    auth = {
        "version": VERSION,
        "decision": "CONFIRMATION-AUTHORIZED",
        "protocol_sha256": protocol_hash,
        "prepare_sha256": sha256_file(prepare_path),
        "fit_sha256": sha256_file(fit_path),
        "selection_sha256": sha256_file(selection_path),
        "split_sha256": sha256_file(root / "COMPOUND_SPLIT.csv"),
        "manifest_sha256": expected_manifest,
        "artifact_index_sha256": sha256_file(artifact_index_path),
        "checkpoint_ledger_sha256": sha256_file(checkpoint_ledger),
        "checkpoint_count": len(indexed_paths),
        "confirmation_loaded_before_authorization": False,
        "confirmation_values_opened_before_authorization": False,
        "confirmation_used_for_selection": False,
        "old_official_test_loaded": False,
        "status": "AUTHORIZED",
    }
    # This write is the explicit information barrier.  No confirmation rows,
    # H5 values, held vectors, or foreign profiles are read above this line.
    write_json(root / "CONFIRMATION_AUTHORIZED.json", auth)
    return auth


def _resolve_path(root: Path, value: Any) -> Path:
    if isinstance(value, Mapping):
        for key in ("path", "file", "checkpoint", "artifact"):
            if value.get(key):
                return _resolve_path(root, value[key])
        raise RuntimeError(f"artifact mapping lacks path: {value}")
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _branches(value: Any) -> list[Any]:
    """Return branch path values while accepting the historical index forms.

    Some fit runs wrote a bare path, some wrote ``{"path": ...}``, and the
    decomposition runs wrote ``{"branches": [...]}``.  A mapping whose keys
    are branch names (for example ``{"cfra": ..., "residual": ...}``) is
    also a valid explicit representation.  We preserve insertion order for
    the latter because branch order is semantically meaningful for M3/M4.
    """
    if isinstance(value, Mapping):
        for key in ("branches", "paths", "checkpoints", "artifacts"):
            if key in value:
                value = value[key]
                break
        else:
            for key in ("path", "file", "checkpoint", "artifact"):
                if value.get(key):
                    return [value[key]]
            # Explicit branch-name mappings are accepted only when every
            # value is path-like.  This avoids silently treating arbitrary
            # metadata mappings as checkpoints.
            if value and all(isinstance(item, (str, Path)) for item in value.values()):
                return list(value.values())
            raise RuntimeError(f"artifact mapping lacks path/branches: {value}")
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _mapping_value(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    """Look up one of ``keys`` case-insensitively, preserving exact priority."""
    for key in keys:
        if key in mapping:
            return mapping[key]
    folded = {str(key).casefold(): value for key, value in mapping.items()}
    for key in keys:
        if str(key).casefold() in folded:
            return folded[str(key).casefold()]
    return None


def _checkpoint_key_candidates(root: Path, key: Any) -> list[Path]:
    """Map checkpoint-ledger keys from current and legacy fit writers to paths.

    The current fit writer records keys such as
    ``budget1/M0_raw/raw/master3407`` while the file is
    ``budget1/M0_raw/raw_master3407.pt``.  The smoke writer records the
    simpler ``budget1/M0/seed3407`` key.  Resolve both without guessing from
    the confirmation data.
    """
    text = str(key)
    raw = Path(text)
    candidates: list[Path] = []

    def add(path: Path) -> None:
        value = path if path.is_absolute() else root / path
        if value not in candidates:
            candidates.append(value)

    if raw.suffix == ".pt":
        add(raw)
    else:
        add(Path(text + ".pt"))
    parts = [part for part in raw.parts if part not in ("", ".")]
    # Fit writer: .../<branch>/master<seed> -> .../<branch>_master<seed>.pt
    if len(parts) >= 2:
        add(Path(*parts[:-2]) / f"{parts[-2]}_{parts[-1]}.pt")
    # A few early smoke/compatibility writers used .../<method>/<branch>/<seed>
    # and omitted the branch separator in the file name.
    if len(parts) >= 3:
        add(Path(*parts[:-3]) / f"{parts[-3]}_{parts[-2]}_{parts[-1]}.pt")
    return candidates


def _flatten_artifact_paths(index: Mapping[str, Mapping[str, Mapping[str, list[Path]]]]) -> list[Path]:
    paths: list[Path] = []
    for budgets in index.values():
        for methods in budgets.values():
            for branches in methods.values():
                paths.extend(branches)
    # Duplicate paths would make branch provenance ambiguous.
    unique = {str(path): path for path in paths}
    if len(unique) != len(paths):
        raise RuntimeError("artifact index reuses one checkpoint for multiple branches")
    return list(unique.values())


def _artifact_index(root: Path) -> dict[str, dict[str, dict[str, list[Path]]]]:
    candidates = (root / "MODEL_ARTIFACTS.json", root / "MODEL_ARTIFACT_INDEX.json", root / "ARTIFACT_INDEX.json")
    index_path = next((path for path in candidates if path.is_file()), None)
    result: dict[str, dict[str, dict[str, list[Path]]]] = {}
    if index_path is not None:
        raw = read_json(index_path)
        raw = raw.get("models", raw)
        for budget in BUDGETS:
            if not isinstance(raw, Mapping):
                raise RuntimeError("artifact index root is not a mapping")
            braw = _mapping_value(raw, (f"budget{budget}", str(budget), f"b{budget}"))
            if braw is None:
                braw = {}
            if not isinstance(braw, Mapping):
                raise RuntimeError(f"artifact index budget{budget} is not a mapping")
            result[str(budget)] = {}
            for method in METHODS:
                mraw = _mapping_value(braw, METHOD_ALIASES[method])
                if mraw is None:
                    raise RuntimeError(f"artifact index missing budget{budget}/{method}")
                if not isinstance(mraw, Mapping):
                    raise RuntimeError(f"artifact index budget{budget}/{method} is not a seed mapping")
                result[str(budget)][method] = {}
                for seed in SEEDS:
                    value = _mapping_value(mraw, (f"seed{seed}", str(seed), f"master{seed}", f"seed_{seed}"))
                    if value is None:
                        raise RuntimeError(f"artifact index missing budget{budget}/{method}/seed{seed}")
                    paths = [_resolve_path(root, item) for item in _branches(value)]
                    expected = EXPECTED_BRANCHES[method]
                    if len(paths) != expected:
                        raise RuntimeError(
                            f"artifact index budget{budget}/{method}/seed{seed} has "
                            f"{len(paths)} branches; expected {expected}"
                        )
                    if any(not path.is_file() for path in paths):
                        missing = [str(path) for path in paths if not path.is_file()]
                        raise RuntimeError(f"artifact index points to missing checkpoint(s): {missing[:3]}")
                    result[str(budget)][method][str(seed)] = paths
        _flatten_artifact_paths(result)
        return result
    # Compatibility fallback: still fail closed when a method is ambiguous.
    for budget in BUDGETS:
        result[str(budget)] = {}
        bdir = root / f"budget{budget}"
        for method in METHODS:
            result[str(budget)][method] = {}
            mdir = bdir / method
            for seed in SEEDS:
                direct = mdir / f"seed{seed}.pt"
                if direct.is_file():
                    result[str(budget)][method][str(seed)] = [direct]
                    continue
                pats = {
                    "M0": [mdir / f"seed{seed}_raw.pt", mdir / f"seed{seed}_a.pt"],
                    "M1": [mdir / f"seed{seed}_cfra.pt"],
                    "M2": [mdir / f"seed{seed}_a.pt", mdir / f"seed{seed}_b.pt"],
                    "M3": [mdir / f"seed{seed}_pca.pt", mdir / f"seed{seed}_residual.pt"],
                    "M4": [mdir / f"seed{seed}_cfra.pt", mdir / f"seed{seed}_residual.pt"],
                }[method]
                found = [path for path in pats if path.is_file()]
                if len(found) != len(pats):
                    raise RuntimeError(f"artifact index absent and fallback incomplete for budget{budget}/{method}/seed{seed}")
                result[str(budget)][method][str(seed)] = found
    _flatten_artifact_paths(result)
    return result


def _load_virtual_confirmation(h5_path: Path, aggregate_path: Path, names: np.ndarray) -> Any:
    """Read only selected confirmation rows after authorization."""
    names = np.asarray(names, dtype=str)
    if len(names) != len(set(names.tolist())):
        raise RuntimeError("confirmation virtual names are not unique")
    if not h5_path.is_file() or not aggregate_path.is_file():
        raise FileNotFoundError("confirmation virtual input is absent")
    with h5py.File(h5_path, "r") as handle:
        all_names = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in handle["canonical_smiles"][:]]
        index = {name: i for i, name in enumerate(all_names)}
        try:
            rows = np.asarray([index[str(name)] for name in names], dtype=np.int64)
        except KeyError as exc:
            raise RuntimeError(f"confirmation compound absent from virtual H5: {exc}") from exc
        order = np.argsort(rows)
        inverse = np.argsort(order)
        selected = rows[order]
        control = np.asarray(handle["control_CP"][selected], dtype=np.float32)[inverse]
    # Only the selected aggregate entries are read.  It is used as a strict
    # H5 compatibility check, never as a confirmation target.
    with np.load(aggregate_path, allow_pickle=False) as aggregate:
        # The new confirmation is drawn from former train+valid.  Depending on
        # the source artifact, aggregates are stored as train/valid separately.
        all_agg_names: list[str] = []
        all_agg_values: list[np.ndarray] = []
        for split in ("train", "valid"):
            nkey, vkey = f"{split}_smiles", f"{split}_cp"
            if nkey not in aggregate or vkey not in aggregate:
                continue
            split_names = np.asarray(aggregate[nkey], dtype=str)
            mapping = {name: i for i, name in enumerate(split_names)}
            selected_names = [str(name) for name in names if str(name) in mapping]
            if selected_names:
                indices = np.asarray([mapping[name] for name in selected_names], dtype=np.int64)
                values = np.asarray(aggregate[vkey][indices], dtype=np.float32)
                all_agg_names.extend(selected_names)
                all_agg_values.extend(list(values))
        if set(all_agg_names) != set(names.tolist()):
            missing = sorted(set(names.tolist()) - set(all_agg_names))
            raise RuntimeError(f"confirmation names absent from aggregate NPZ: {missing[:3]}")
        # Keep the check useful but avoid depending on aggregate ordering.
        aggregate_map = {name: value for name, value in zip(all_agg_names, all_agg_values)}
        aggregate_delta = np.asarray([aggregate_map[str(name)] for name in names], dtype=np.float32)
        if aggregate_delta.shape != control.shape or not np.isfinite(aggregate_delta).all():
            raise RuntimeError("invalid selected confirmation aggregate profile")
    fingerprint = LEGACY.VIRTUAL.fingerprints(names.tolist()).astype(np.float32)
    if control.shape != (len(names), CP_DIM) or fingerprint.shape != (len(names), FP_DIM):
        raise RuntimeError("invalid selected virtual shape")
    return type("VirtualSelection", (), {"smiles": names, "control": control, "fingerprint": fingerprint})()


def _combine_source_rows(args: argparse.Namespace, confirmation: bool) -> Any:
    if args.smoke:
        rows = SRC.synthetic_rows(("train", "valid"))
        train, valid = rows["train"], rows["valid"]
    else:
        rows = SRC.load_rows(args.rows_root, ("train", "valid"), smoke=False)
        train, valid = rows["train"], rows["valid"]
    return SRC.LEGACY.Rows(
        "BBBC047", "CP", "confirmation_source",
        np.concatenate([train.compound, valid.compound]),
        np.concatenate([train.dose, valid.dose]),
        np.concatenate([train.plate, valid.plate]),
        np.concatenate([train.delta, valid.delta]),
    )


def _confirmation_compounds(root: Path) -> set[str]:
    rows = read_csv(root / "COMPOUND_SPLIT.csv")
    values = {str(row["compound_id"]) for row in rows if str(row.get("split", "")) == "confirmation"}
    if not values:
        raise RuntimeError("frozen confirmation compound set is empty")
    return values


def _checkpoint_prediction(path: Path, virtual: Any, names: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping) or "state_dict" not in checkpoint or "stats" not in checkpoint:
        raise RuntimeError(f"checkpoint lacks state_dict/stats: {path}")
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in checkpoint["stats"].items()}
    required = {"control_mean", "control_scale", "target_mean", "target_scale"}
    if not required.issubset(stats):
        raise RuntimeError(f"checkpoint stats incomplete: {path}")
    for key in required:
        if stats[key].shape != (CP_DIM,) or not np.isfinite(stats[key]).all():
            raise RuntimeError(f"checkpoint stats invalid: {path}:{key}")
    model = LEGACY.Student().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    dummy = np.zeros((len(names), CP_DIM), dtype=np.float32)
    split = LEGACY.PredictorSplit(
        compound=np.asarray(names, dtype=str),
        dose=np.asarray(["" for _ in names], dtype=str),
        control=np.asarray(virtual.control, dtype=np.float32),
        fingerprint=np.asarray(virtual.fingerprint, dtype=np.float32),
        target=dummy,
    )
    prediction = LEGACY.predict_student(model, split, stats, batch_size, device)
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.shape != (len(names), CP_DIM) or not np.isfinite(prediction).all():
        raise RuntimeError(f"invalid prediction from {path}: {prediction.shape}")
    return prediction


def _method_predictions(index: Mapping[str, Mapping[str, Mapping[str, list[Path]]]], budget: int, method: str, seed: int, virtual: Any, names: np.ndarray, device: torch.device, batch_size: int) -> tuple[np.ndarray, list[np.ndarray]]:
    paths = index[str(budget)][method][str(seed)]
    branches = [_checkpoint_prediction(path, virtual, names, device, batch_size) for path in paths]
    if method in ("M0", "M1"):
        if len(branches) != 1:
            raise RuntimeError(f"{method} expects exactly one checkpoint, got {len(branches)}")
        return branches[0], branches
    if method == "M2":
        if len(branches) != 2:
            raise RuntimeError("M2 requires two independent Raw branches")
        return np.mean(np.stack(branches, axis=0), axis=0).astype(np.float32), branches
    if method in ("M3", "M4"):
        if len(branches) != 2:
            raise RuntimeError(f"{method} requires two additive branches")
        return (branches[0] + branches[1]).astype(np.float32), branches
    raise RuntimeError(method)


def _metrics(prediction: np.ndarray, examples: Any) -> dict[str, np.ndarray]:
    return SRC.dual_target_metrics(prediction, examples)


def _compound_metric(metrics_by_seed: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(metrics_by_seed, axis=0).astype(np.float64)
    with np.errstate(invalid="ignore"):
        return np.nanmean(stacked, axis=0)


def paired_bootstrap_difference(left: np.ndarray, right: np.ndarray, seed: int, rounds: int) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if len(left) == 0:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_compounds": 0, "rounds": rounds}
    diff = left - right
    rng = np.random.default_rng(int(seed) % (2**63 - 1))
    indices = rng.integers(0, len(diff), size=(rounds, len(diff)), endpoint=False)
    draws = np.mean(diff[indices], axis=1)
    return {
        "estimate": float(np.mean(diff)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_compounds": int(len(diff)),
        "rounds": int(rounds),
    }


def _finite_mean(value: np.ndarray) -> float:
    return float(np.nanmean(value)) if np.any(np.isfinite(value)) else float("nan")


def _summary_rows(seed_metrics: dict[tuple[int, str, int], dict[str, np.ndarray]], budget: int, method: str) -> dict[str, Any]:
    metrics = ("A_A", "A_B", "E_A", "E_B", "E_dual", "held_pcc_A", "held_pcc_B")
    row: dict[str, Any] = {"budget": budget, "method": method, "method_label": METHOD_LABELS[method], "n_seeds": len(SEEDS)}
    for metric in metrics:
        values = _compound_metric([seed_metrics[(budget, method, seed)][metric] for seed in SEEDS])
        row[f"{metric}_mean"] = _finite_mean(values)
        row[f"{metric}_n"] = int(np.sum(np.isfinite(values)))
    row["A_mean"] = float(np.nanmean([row["A_A_mean"], row["A_B_mean"]]))
    row["E_mean"] = row["E_dual_mean"]
    return row


def acceptance(contrasts: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in contrasts:
        if row.get("seed") == "seed_averaged":
            by[(int(row["budget"]), str(row["comparison"]), str(row["metric"]))] = row
    primary = {}
    m4m0_ok: dict[int, bool] = {}
    for budget in PRIMARY:
        a = by[(budget, "M4-M0", "A")]
        e = by[(budget, "M4-M0", "E")]
        ea = by[(budget, "M4-M0", "E_A")]
        eb = by[(budget, "M4-M0", "E_B")]
        control: dict[str, bool] = {}
        for comparator in ("M4-M2", "M4-M3"):
            ca = by[(budget, comparator, "A")]
            ce = by[(budget, comparator, "E")]
            control[comparator] = bool(
                float(ca["estimate"]) >= 0 and float(ce["estimate"]) >= 0
                and (float(ca["ci_low"]) > 0 or float(ce["ci_low"]) > 0)
            )
        m4m0_ok[budget] = bool(float(a["ci_low"]) > 0 and float(e["ci_low"]) > 0)
        primary[budget] = {
            "M4_minus_M0_A": a,
            "M4_minus_M0_E": e,
            "M4_minus_M0_E_A": ea,
            "M4_minus_M0_E_B": eb,
            "M4_minus_M0_primary_ok": m4m0_ok[budget],
            "E_A_point_positive": bool(float(ea["estimate"]) > 0),
            "E_B_point_positive": bool(float(eb["estimate"]) > 0),
            "mechanism_controls": control,
        }
    all_primary = all(m4m0_ok.values())
    controls = all(all(primary[b]["mechanism_controls"].values()) for b in PRIMARY)
    strong = bool(
        all_primary
        and all(primary[b]["E_A_point_positive"] and primary[b]["E_B_point_positive"] for b in PRIMARY)
        and controls
        and all(
            float(by[(b, comparator, metric)]["estimate"]) >= 0
            for b in PRIMARY
            for comparator in ("M4-M2", "M4-M3")
            for metric in ("A", "E")
        )
    )
    partial = bool(all_primary and not strong)
    decision = "STRONG GO" if strong else "PARTIAL GO" if partial else "NO-GO"
    return {"decision": decision, "strong_go": strong, "partial_go": partial, "primary": primary, "mechanism_controls_pass": controls, "M4_minus_M0_primary_pass": all_primary}


def pareto_plot(summary: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, len(PRIMARY), figsize=(12, 5), squeeze=False)
    for ax, budget in zip(axes[0], PRIMARY):
        rows = [row for row in summary if int(row["budget"]) == budget]
        for row in rows:
            ax.scatter(row["A_mean"], row["E_mean"], s=65, label=row["method"])
            ax.annotate(row["method"], (row["A_mean"], row["E_mean"]), xytext=(4, 4), textcoords="offset points")
        ax.set_title(f"Budget {budget}R")
        ax.set_xlabel("A = mean same-repeat Fisher-z")
        ax.set_ylabel("E = same minus foreign Fisher-z")
        ax.grid(alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root
    if not root.is_dir():
        raise FileNotFoundError(root)
    auth = authorize_confirmation(root)
    index = _artifact_index(root)
    device = device_for(args.device)

    # Information barrier crossed above.  From here onward confirmation values
    # may be opened.  The formal path never touches test_cp_plate_rows.npz.
    confirmation_compounds = _confirmation_compounds(root)
    source_rows = _combine_source_rows(args, confirmation=True)
    confirmation_rows = PREP.subset(source_rows, confirmation_compounds, "confirmation")
    bundles = {budget: SRC.build_dual_examples(confirmation_rows, budget, include_foreign=True) for budget in BUDGETS}
    names_by_budget = {budget: np.asarray(bundles[budget].compound, dtype=str) for budget in BUDGETS}
    if args.smoke:
        virtual = SRC.synthetic_virtual({"confirmation": confirmation_rows})["confirmation"]
    else:
        virtual = _load_virtual_confirmation(args.model_h5, args.aggregate_npz, np.asarray(sorted(confirmation_compounds), dtype=str))
    # Reorder virtual inputs into each budget's deterministic compound order.
    virtual_index = {str(name): i for i, name in enumerate(virtual.smiles)}
    seed_metrics: dict[tuple[int, str, int], dict[str, np.ndarray]] = {}
    seed_predictions: dict[tuple[int, str, int], np.ndarray] = {}
    branch_predictions: list[dict[str, Any]] = []
    per_compound: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    metric_names = ("A_A", "A_B", "E_A", "E_B", "E_dual", "held_pcc_A", "held_pcc_B")
    for budget in BUDGETS:
        examples = bundles[budget]
        names = names_by_budget[budget]
        ix = np.asarray([virtual_index[str(name)] for name in names], dtype=np.int64)
        v = type("VirtualSelection", (), {"smiles": names, "control": virtual.control[ix], "fingerprint": virtual.fingerprint[ix]})()
        for method in METHODS:
            for seed in SEEDS:
                prediction, branches = _method_predictions(index, budget, method, seed, v, names, device, args.batch_size)
                metrics = _metrics(prediction, examples)
                seed_metrics[(budget, method, seed)] = metrics
                seed_predictions[(budget, method, seed)] = prediction
                for i, compound in enumerate(names):
                    per_compound.append({"budget": budget, "method": method, "method_label": METHOD_LABELS[method], "seed": seed, "compound_id": str(compound), "dose": str(examples.dose[i]), "foreign_ok": int(examples.foreign_ok[i]), **{metric: metrics[metric][i] for metric in metric_names}})
                for branch_index, branch in enumerate(branches):
                    branch_rows.append({"budget": budget, "method": method, "seed": seed, "branch": branch_index, "n_compounds": len(branch), "mean_norm": float(np.mean(np.linalg.norm(branch.astype(np.float64), axis=1))), "sum_norm": float(np.mean(np.linalg.norm(branch.astype(np.float64), axis=1)))})
    summary = [_summary_rows(seed_metrics, budget, method) for budget in BUDGETS for method in METHODS]
    contrasts: list[dict[str, Any]] = []
    comparison_pairs = (("M4-M0", "M4", "M0"), ("M4-M2", "M4", "M2"), ("M4-M3", "M4", "M3"))
    metrics_for_contrast = ("A", "E", "E_A", "E_B")
    for budget in BUDGETS:
        compound_arrays: dict[str, dict[str, np.ndarray]] = {}
        for method in METHODS:
            compound_arrays[method] = {}
            compound_arrays[method]["A"] = ( _compound_metric([seed_metrics[(budget, method, seed)]["A_A"] for seed in SEEDS]) + _compound_metric([seed_metrics[(budget, method, seed)]["A_B"] for seed in SEEDS]) ) / 2.0
            compound_arrays[method]["E"] = _compound_metric([seed_metrics[(budget, method, seed)]["E_dual"] for seed in SEEDS])
            compound_arrays[method]["E_A"] = _compound_metric([seed_metrics[(budget, method, seed)]["E_A"] for seed in SEEDS])
            compound_arrays[method]["E_B"] = _compound_metric([seed_metrics[(budget, method, seed)]["E_B"] for seed in SEEDS])
        for comparison, left_method, right_method in comparison_pairs:
            for metric in metrics_for_contrast:
                result = paired_bootstrap_difference(compound_arrays[left_method][metric], compound_arrays[right_method][metric], PREP.SRC.LEGACY.stable_int(PREP.SRC.LEGACY.PROTOCOL_SEED, f"{VERSION}|confirmation|b{budget}|{comparison}|{metric}"), args.bootstrap_rounds)
                contrasts.append({"budget": budget, "comparison": comparison, "metric": metric, "seed": "seed_averaged", **result})
                for seed in SEEDS:
                    if metric == "A":
                        left = (seed_metrics[(budget, left_method, seed)]["A_A"] + seed_metrics[(budget, left_method, seed)]["A_B"]) / 2.0
                        right = (seed_metrics[(budget, right_method, seed)]["A_A"] + seed_metrics[(budget, right_method, seed)]["A_B"]) / 2.0
                    elif metric == "E":
                        left, right = seed_metrics[(budget, left_method, seed)]["E_dual"], seed_metrics[(budget, right_method, seed)]["E_dual"]
                    else:
                        left, right = seed_metrics[(budget, left_method, seed)][metric], seed_metrics[(budget, right_method, seed)][metric]
                    seed_result = paired_bootstrap_difference(left, right, PREP.SRC.LEGACY.stable_int(PREP.SRC.LEGACY.PROTOCOL_SEED, f"{VERSION}|confirmation|b{budget}|{comparison}|{metric}|seed{seed}"), args.bootstrap_rounds)
                    contrasts.append({"budget": budget, "comparison": comparison, "metric": metric, "seed": seed, **seed_result})
    decision = acceptance(contrasts)
    # Branch decomposition retains branch-level norms and compares the additive
    # reconstruction to its constituent model at compound level.
    for budget in BUDGETS:
        names = names_by_budget[budget]
        examples = bundles[budget]
        for seed in SEEDS:
            m4 = seed_predictions[(budget, "M4", seed)]
            cfra = _method_predictions(index, budget, "M4", seed, type("VirtualSelection", (), {"smiles": names, "control": virtual.control[[virtual_index[str(name)] for name in names]], "fingerprint": virtual.fingerprint[[virtual_index[str(name)] for name in names]]})(), names, device, args.batch_size)[1][0]
            residual = _method_predictions(index, budget, "M4", seed, type("VirtualSelection", (), {"smiles": names, "control": virtual.control[[virtual_index[str(name)] for name in names]], "fingerprint": virtual.fingerprint[[virtual_index[str(name)] for name in names]]})(), names, device, args.batch_size)[1][1]
            for i, compound in enumerate(names):
                    branch_rows.append({"budget": budget, "method": "M4", "seed": seed, "branch": "compound", "compound_id": str(compound), "cfra_held_A_pcc": LEGACY.rowwise_corr(cfra[i:i+1], examples.held_a[i:i+1])[0], "cfra_held_B_pcc": LEGACY.rowwise_corr(cfra[i:i+1], examples.held_b[i:i+1])[0], "residual_held_A_pcc": LEGACY.rowwise_corr(residual[i:i+1], examples.held_a[i:i+1])[0], "residual_held_B_pcc": LEGACY.rowwise_corr(residual[i:i+1], examples.held_b[i:i+1])[0], "combined_held_A_pcc": LEGACY.rowwise_corr(m4[i:i+1], examples.held_a[i:i+1])[0], "combined_held_B_pcc": LEGACY.rowwise_corr(m4[i:i+1], examples.held_b[i:i+1])[0], "m4_sum_norm": float(np.linalg.norm(m4[i])), "cfra_norm": float(np.linalg.norm(cfra[i])), "residual_norm": float(np.linalg.norm(residual[i]))})
    write_csv(root / "CONFIRMATION_PER_COMPOUND.csv", per_compound)
    write_csv(root / "CONFIRMATION_SUMMARY.csv", summary)
    write_csv(root / "CONFIRMATION_CONTRASTS.csv", contrasts)
    write_csv(root / "BRANCH_DECOMPOSITION.csv", branch_rows)
    pareto_plot(summary, root / "PARETO.png")
    confirmation_source_hashes = {"train_cp_plate_rows": None if args.smoke else sha256_file(args.rows_root / "train_cp_plate_rows.npz"), "valid_cp_plate_rows": None if args.smoke else sha256_file(args.rows_root / "valid_cp_plate_rows.npz"), "model_h5": None if args.smoke else sha256_file(args.model_h5), "aggregate_npz": None if args.smoke else sha256_file(args.aggregate_npz)}
    audit = {"version": VERSION, "phase": "confirm", "status": "PASS", "decision": decision, "protocol_sha256": sha256_file(PROTOCOL), "authorization_sha256": sha256_file(root / "CONFIRMATION_AUTHORIZED.json"), "confirmation_loaded_after_authorization": True, "confirmation_values_opened_after_authorization": True, "confirmation_used_for_selection": False, "old_official_test_loaded": False, "budgets": list(BUDGETS), "primary_budgets": list(PRIMARY), "methods": list(METHODS), "master_seeds": list(SEEDS), "bootstrap_rounds": args.bootstrap_rounds, "bootstrap_unit": "compound", "compound_counts": {str(budget): len(bundles[budget]) for budget in BUDGETS}, "foreign_counts": {str(budget): int(np.sum(bundles[budget].foreign_ok)) for budget in BUDGETS}, "source_hashes": confirmation_source_hashes, "acceptance": decision}
    write_json(root / "CONFIRMATION_COMPLETE.json", audit)
    return {"audit": audit, "summary": summary, "contrasts": contrasts, "acceptance": decision}


def smoke(args: argparse.Namespace) -> dict[str, Any]:
    # Smoke uses a private temporary root with synthetic rows and checkpoints;
    # it exercises the authorization barrier and metric/plot paths without any
    # scientific claim.  Formal fit artifacts are never modified.
    root = args.output_root
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    protocol_hash = sha256_file(PROTOCOL)
    split_rows = []
    synthetic = SRC.synthetic_rows(("train", "valid"))
    compounds = sorted(set(synthetic["train"].compound.tolist() + synthetic["valid"].compound.tolist()))
    for i, compound in enumerate(compounds):
        split_rows.append({"split": "confirmation", "compound_id": compound, "order_hash": str(i)})
    write_csv(root / "COMPOUND_SPLIT.csv", split_rows)
    source = _combine_source_rows(args, confirmation=True)
    confirmation_view = PREP.subset(source, set(compounds), "confirmation")
    bundles = {b: SRC.build_dual_examples(confirmation_view, b, include_foreign=True) for b in BUDGETS}
    manifest = []
    for b, ex in bundles.items():
        for i in range(len(ex)):
            manifest.append({"split": "confirmation", "budget": b, "compound_id": str(ex.compound[i]), "dose": str(ex.dose[i]), "support_rep_ids": "|".join(ex.support_plates[i]), "held_A_rep_id": "|".join(ex.held_a_plates[i]), "held_B_rep_id": "|".join(ex.held_b_plates[i])})
    write_csv(root / "ANALYSIS_MANIFEST.csv", manifest)
    # Synthetic checkpoints all use the same valid protocol Student state; the
    # prediction/evaluation path, not scientific training, is what smoke tests.
    device = device_for("cpu")
    virtual = SRC.synthetic_virtual({"confirmation": confirmation_view})["confirmation"]
    for b in BUDGETS:
        bdir = root / f"budget{b}"
        for method in METHODS:
            mdir = bdir / method
            mdir.mkdir(parents=True)
            for seed in SEEDS:
                nbranches = 1 if method in ("M0", "M1") else 2
                paths = []
                for branch in range(nbranches):
                    path = mdir / (f"seed{seed}.pt" if nbranches == 1 else f"seed{seed}_{branch}.pt")
                    model = LEGACY.Student()
                    stats = {"control_mean": np.zeros(CP_DIM, np.float32), "control_scale": np.ones(CP_DIM, np.float32), "target_mean": np.zeros(CP_DIM, np.float32), "target_scale": np.ones(CP_DIM, np.float32)}
                    torch.save({"version": VERSION, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "stats": stats}, path)
                    paths.append(str(path.relative_to(root)))
                # Rename branch conventions for explicit index; M2/M3/M4 are
                # represented by the list and do not rely on fallback names.
    artifact_index: dict[str, Any] = {}
    for b in BUDGETS:
        artifact_index[f"budget{b}"] = {}
        for method in METHODS:
            artifact_index[f"budget{b}"][method] = {}
            nbranches = 1 if method in ("M0", "M1") else 2
            for seed in SEEDS:
                paths = [f"budget{b}/{method}/seed{seed}.pt"] if nbranches == 1 else [f"budget{b}/{method}/seed{seed}_0.pt", f"budget{b}/{method}/seed{seed}_1.pt"]
                artifact_index[f"budget{b}"][method][f"seed{seed}"] = paths
    write_json(root / "MODEL_ARTIFACTS.json", artifact_index)
    hashes = {}
    for b in BUDGETS:
        for method in METHODS:
            paths = artifact_index[f"budget{b}"][method]
            for seed, values in paths.items():
                for path in values:
                    rel = Path(path)
                    hashes[f"budget{b}/{method}/{rel.stem}"] = sha256_file(root / rel)
    write_json(root / "CHECKPOINT_HASHES.json", hashes)
    base = {"version": VERSION, "protocol_sha256": protocol_hash, "manifest_sha256": sha256_file(root / "ANALYSIS_MANIFEST.csv"), "split_sha256": sha256_file(root / "COMPOUND_SPLIT.csv"), "test_loaded": False, "test_values_opened": False, "test_used_for_selection": False, "confirmation_loaded": False, "confirmation_values_opened": False, "confirmation_used_for_selection": False}
    write_json(root / "PREPARE_COMPLETE.json", {**base, "phase": "prepare", "status": "PASS", "confirmation_model_scores_computed_in_prepare": False, "confirmation_used_for_training": False, "confirmation_used_for_selection": False})
    write_json(root / "FIT_COMPLETE.json", {**base, "phase": "fit", "status": "PASS", "checkpoint_hashes": hashes})
    write_json(root / "SELECTION_FREEZE.json", {**base, "phase": "selection", "status": "PASS", "selection_frozen": True, "checkpoint_hashes_sha256": sha256_file(root / "CHECKPOINT_HASHES.json")})
    return run(args)


def main() -> None:
    args = parse_args()
    if args.smoke:
        result = smoke(args)
    else:
        result = run(args)
    print(json.dumps({"phase": "confirm", "root": str(args.output_root), "decision": result["acceptance"]["decision"], "confirmation_loaded_after_authorization": True}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
