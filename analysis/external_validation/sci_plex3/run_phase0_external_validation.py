#!/usr/bin/env python3
"""Phase-0 audit and independent-repeat pseudo-bulk builder.

This module deliberately stops at data auditing and profile construction.  It
does not select HVGs, train IMR/LSO/IMCEB/CFRA, fit CFRA weights, or open a
confirmation split.  The implementation is intentionally conservative about
AnnData schemas: fields and layers are inferred only when their names/values
are recognisable, and an ambiguous schema fails closed with a machine-readable
reason in ``AUDIT.json``.

Examples
--------
Audit a downloaded scPerturb file, then build profiles only when the audit
gate passes::

    python run_phase0_external_validation.py run --dataset sciplex3 \
        --input /data/sciplex3.h5ad --output-root ./sciplex3_run

The two explicit stages are also available and are useful on a cluster::

    python run_phase0_external_validation.py audit --dataset papalexi \
        --input /data/papalexi.h5ad --output-root ./papalexi_run/00_audit
    python run_phase0_external_validation.py profiles --dataset papalexi \
        --input /data/papalexi.h5ad --audit-json ./papalexi_run/00_audit/AUDIT.json \
        --output-root ./papalexi_run/01_profiles

The profile HDF5 schema is documented in ``PROFILE_SCHEMA.md``.  It stores
one delta profile per condition and independent replicate, with row metadata
under ``obs/`` and feature names under ``var/``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _datetime
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_VERSION = "external-validation-phase0-v1-2026-09-03"
PROTOCOL_SEED = 3407
SCI_EXPECTED_REPLICATES = ("R1", "R2")
PAPA_EXPECTED_REPLICATES = ("R1", "R2", "R3")
SCI_EXPECTED_DRUGS = 188
SCI_EXPECTED_CELL_LINES = 3
SCI_EXPECTED_DOSES = 4
SCI_MIN_ELIGIBLE_DRUGS = 100
SCI_MIN_CONDITION_FRACTION = 0.50
PAPA_MIN_TARGETING_GUIDES = 60
SCI_THRESHOLDS = (25, 50, 100)
PAPA_THRESHOLDS = (15, 25)
DEFAULT_CPM = 10_000.0
DEFAULT_CHUNK_CELLS = 2048


class BlockedError(RuntimeError):
    """A fail-closed schema or eligibility problem."""

    def __init__(self, message: str, *, reasons: Sequence[str] | None = None):
        super().__init__(message)
        self.reasons = list(reasons or [message])


@dataclass
class MatrixSpec:
    """Description of an AnnData matrix without serialising the matrix itself."""

    kind: str
    key: str | None
    matrix: Any
    n_obs: int
    n_features: int
    feature_names: list[str]
    feature_indices: list[int]
    requested: str | None = None
    count_like: bool = False
    count_like_fraction: float = 0.0
    integer_fraction: float = 0.0
    nonnegative: bool = False
    feature_type_field: str | None = None
    feature_type_values: list[str] = field(default_factory=list)
    names_inferred: bool = False

    @property
    def selected_feature_names(self) -> list[str]:
        return self.feature_names

    def summary(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "key": self.key,
            "requested": self.requested,
            "n_obs": self.n_obs,
            "n_features_in_matrix": self.n_features,
            "n_features_selected": len(self.feature_indices),
            "feature_names_head": self.feature_names[:10],
            "feature_names_tail": self.feature_names[-10:] if self.feature_names else [],
            "names_inferred": self.names_inferred,
            "count_like": self.count_like,
            "count_like_fraction": self.count_like_fraction,
            "integer_fraction": self.integer_fraction,
            "nonnegative": self.nonnegative,
            "feature_type_field": self.feature_type_field,
            "feature_type_values": self.feature_type_values[:20],
        }


@dataclass
class AuditContext:
    dataset: str
    input_path: Path
    adata: Any
    obs: Any
    fields: dict[str, str | None]
    field_candidates: dict[str, list[str]]
    replicate_values_observed: list[str]
    replicate_map: dict[str, str]
    selected_replicates: list[str]
    excluded_replicates: list[str]
    rna: MatrixSpec
    adt: MatrixSpec | None
    # Papalexi may provide RNA and ADT in separate h5ad files.  ``adata``
    # remains the RNA/metadata authority; this handle is an obs-index-aligned
    # view of the optional ADT file and is never used for metadata grouping.
    adt_adata: Any | None
    adt_input_path: Path | None
    is_control: Any
    control_reason: str
    is_stimulated: Any
    stimulation_reason: str
    is_targeting: Any
    targeting_reason: str
    main_mask: Any
    audit_mask: Any
    audit_rows: list[dict[str, Any]]
    eligibility_rows: list[dict[str, Any]]
    eligibility_detail_rows: list[dict[str, Any]]
    gate_pass: bool
    gate_reasons: list[str]
    warnings: list[str]
    source_sha256: str | None
    adt_source_sha256: str | None
    source_size: int | None


def _import_runtime() -> tuple[Any, Any, Any, Any]:
    """Import optional scientific dependencies with an actionable error."""

    try:
        import anndata  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised on minimal hosts
        raise BlockedError(
            "AnnData input requires the 'anndata' package; install anndata, "
            "numpy, pandas, scipy, and h5py in the execution environment",
            reasons=[f"missing anndata dependency: {exc}"],
        ) from exc
    try:
        import h5py  # type: ignore
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        import scipy.sparse as sp  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise BlockedError(
            "Profile construction requires numpy, pandas, scipy, and h5py",
            reasons=[f"missing profile dependency: {exc}"],
        ) from exc
    return anndata, h5py, np, pd, sp


def _now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _norm(value: Any) -> str:
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def _string(value: Any) -> str:
    """Stable scalar serialisation for pandas/AnnData values."""

    if value is None:
        return "__NA__"
    try:
        # Avoid pandas' NA object without importing pandas at module import time.
        if bool(value != value):
            return "__NA__"
    except Exception:
        pass
    text = str(value).strip()
    return text if text else "__NA__"


def _jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalar values to JSON-safe primitives."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    # numpy scalar APIs are deliberately accessed by duck typing.
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: set[str] = set()
        for row in rows:
            keys.update(str(k) for k in row.keys())
        fieldnames = sorted(keys)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        return ";".join(_string(x) for x in value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_input(path: Path) -> None:
    if not path.is_file():
        raise BlockedError(f"input .h5ad file does not exist: {path}", reasons=[f"missing input: {path}"])
    if path.suffix.lower() != ".h5ad":
        raise BlockedError(f"input must be an .h5ad file, got {path.name}", reasons=["input extension is not .h5ad"])


def _read_anndata(path: Path) -> Any:
    anndata, _, _, _, _ = _import_runtime()
    try:
        # Backed mode avoids a second full in-memory copy for large scPerturb files.
        return anndata.read_h5ad(str(path), backed="r")
    except TypeError:
        return anndata.read_h5ad(str(path))
    except Exception as exc:
        raise BlockedError(
            f"could not read AnnData file {path}: {exc}", reasons=[f"AnnData read failure: {exc}"]
        ) from exc


def _close_adata(adata: Any) -> None:
    try:
        file_obj = getattr(adata, "file", None)
        close = getattr(file_obj, "close", None)
        if close is not None:
            close()
    except Exception:
        pass


def _align_auxiliary_adata_to_reference(reference: Any, auxiliary: Any, np: Any) -> Any:
    """Return an auxiliary AnnData view reordered to the RNA obs index.

    The RNA file owns all metadata and group definitions.  The auxiliary file
    is accepted only when both obs indices are unique and contain exactly the
    same identifiers (order may differ).  Converting identifiers to strings is
    intentional: it catches the common ``1`` versus ``"1"`` representation
    mismatch while allowing anndata/pandas index subclasses to compare
    consistently.  A duplicate or set mismatch is a hard Phase-0 block.
    """

    reference_index = [_string(value) for value in list(reference.obs.index)]
    auxiliary_index = [_string(value) for value in list(auxiliary.obs.index)]
    if len(reference_index) != len(set(reference_index)):
        raise BlockedError(
            "RNA obs index is not unique; cannot align a separate ADT file",
            reasons=["RNA obs index contains duplicates"],
        )
    if len(auxiliary_index) != len(set(auxiliary_index)):
        raise BlockedError(
            "ADT obs index is not unique; cannot align a separate ADT file",
            reasons=["ADT obs index contains duplicates"],
        )
    reference_set = set(reference_index)
    auxiliary_set = set(auxiliary_index)
    missing = sorted(reference_set - auxiliary_set)
    extra = sorted(auxiliary_set - reference_set)
    if missing or extra or len(reference_index) != len(auxiliary_index):
        details: list[str] = []
        if missing:
            details.append(f"missing_in_adt={missing[:10]}")
        if extra:
            details.append(f"extra_in_adt={extra[:10]}")
        if len(reference_index) != len(auxiliary_index):
            details.append(f"n_obs_rna={len(reference_index)} n_obs_adt={len(auxiliary_index)}")
        raise BlockedError(
            "RNA and auxiliary ADT obs indices are not a one-to-one match",
            reasons=["separate ADT alignment failed: " + "; ".join(details)],
        )
    auxiliary_position = {identifier: position for position, identifier in enumerate(auxiliary_index)}
    reorder = np.asarray([auxiliary_position[identifier] for identifier in reference_index], dtype=np.int64)
    # AnnData views preserve backed/lazy matrices; this avoids copying a large
    # protein matrix while making every subsequent row index refer to RNA obs.
    try:
        aligned = auxiliary[reorder, :]
    except Exception as exc:
        raise BlockedError(
            f"could not reorder ADT AnnData by RNA obs index: {exc}",
            reasons=["auxiliary ADT view could not be aligned to RNA rows"],
        ) from exc
    return aligned


FIELD_ALIASES: dict[str, dict[str, list[str]]] = {
    "sciplex3": {
        "drug_name": [
            "drug_name", "drug", "compound", "compound_name", "perturbation_name",
            "perturbation", "treatment", "drug_id", "compound_id",
        ],
        "cell_line": ["cell_line", "cellline", "cell_line_name", "cell_type", "celltype", "cell"],
        "dose": ["dose", "concentration", "dose_value", "pert_dose", "perturbation_dose", "dose_name"],
        "replicate": [
            "biological_replicate", "bio_replicate", "biological_rep", "replicate_id",
            "replicate", "repeat", "rep", "experiment_replicate", "culture_replicate",
        ],
        "plate": ["plate_id", "plate", "plate_name", "batch_plate", "well_plate"],
        "control_status": ["control_status", "is_control", "control", "treatment_status", "sample_type"],
    },
    "papalexi": {
        "guide_id": [
            "guide_id", "grna_id", "grna", "guide", "sg_rna", "sgrna", "sgRNA",
            "guide_identity", "guide_identity_id", "perturbation", "perturbation_id",
        ],
        "target_gene": [
            "target_gene", "target_gene_symbol", "gene_symbol", "perturbed_gene",
            "gene", "target", "gene_id", "perturbation_gene",
        ],
        "replicate": [
            "transduction_replicate", "transduction_rep", "viral_replicate", "biological_replicate",
            "biological_rep", "replicate_id", "replicate", "repeat", "rep", "experiment",
        ],
        "plate": ["plate_id", "plate", "plate_name", "batch_plate", "library_batch"],
        "stimulated": [
            "stimulated", "stimulation", "stimulus", "ifng", "ifn_gamma", "ifnγ",
            "interferon_gamma", "activation", "condition", "treatment_condition",
        ],
        "targeting_status": [
            "targeting", "is_targeting", "guide_type", "perturbation_type", "targeting_status",
            "gRNA_type", "grna_type", "control_status",
        ],
        "mixscape": ["mixscape_class", "mixscape_label", "mixscape_ko", "mixscape_class_global", "ko_status"],
        "cell_line": ["cell_line", "cellline", "cell_type", "celltype", "cell"],
    },
}


def _column_names(obs: Any) -> list[str]:
    return [str(value) for value in list(obs.columns)]


def _resolve_field(obs: Any, dataset: str, role: str, override: str | None) -> tuple[str | None, list[str]]:
    columns = _column_names(obs)
    if override:
        if override not in columns:
            raise BlockedError(
                f"requested obs field {override!r} for role {role!r} is absent",
                reasons=[f"missing requested obs field: {override}"],
            )
        return override, [override]
    aliases = FIELD_ALIASES[dataset].get(role, [])
    by_norm = {_norm(name): name for name in columns}
    matches: list[str] = []
    for alias in aliases:
        if _norm(alias) in by_norm and by_norm[_norm(alias)] not in matches:
            matches.append(by_norm[_norm(alias)])
    return (matches[0] if matches else None), matches


def _series_values(series: Any) -> list[str]:
    return [_string(value) for value in series.tolist()]


def _normalise_replicate(value: str, dataset: str) -> str:
    text = _string(value)
    compact = _norm(text)
    # A replicate column is already semantically constrained, so accepting
    # "replicate_1", "R1", and numeric 1 is safe and makes scPerturb exports
    # interoperable.  Values such as A/B are retained as labels.
    match = re.search(r"(?:replicate|repeat|rep|transduction|culture|r)?[_\- ]*([0-9]+)$", compact)
    if match:
        number = int(match.group(1))
        return f"R{number}"
    return text


def _choose_replicates(values: Sequence[str], dataset: str) -> tuple[dict[str, str], list[str], list[str], list[str]]:
    observed = sorted({value for value in values if value != "__NA__"})
    mapping = {value: _normalise_replicate(value, dataset) for value in observed}
    expected = SCI_EXPECTED_REPLICATES if dataset == "sciplex3" else PAPA_EXPECTED_REPLICATES
    canonical_values = set(mapping.values())
    if set(expected).issubset(canonical_values):
        selected = list(expected)
        excluded = sorted({mapping[value] for value in observed if mapping[value] not in expected})
        return mapping, selected, excluded, []
    if len(observed) == len(expected) and len(observed) > 0:
        # A/B/C labels are usable as three independent repeats only when there
        # are exactly the expected number; their order is frozen lexicographically.
        ordered = sorted(observed)
        remap = {value: expected[index] for index, value in enumerate(ordered)}
        return remap, list(expected), [], [
            f"replicate labels {ordered} were deterministically mapped to {list(expected)}"
        ]
    if dataset == "papalexi" and len(observed) > 3:
        extra = [value for value in observed if mapping[value] not in expected]
        if all(f"R{index}" in canonical_values for index in (1, 2, 3)):
            return mapping, list(expected), sorted(extra), [
                "replicate values outside R1/R2/R3 were excluded from the main Papalexi analysis; "
                "in particular, a fourth transduction is never reintroduced"
            ]
    raise BlockedError(
        f"could not identify the protocol replicates in values {observed}",
        reasons=[
            f"replicate schema mismatch: observed={observed}, expected={list(expected)}",
            "pass --replicate-field and/or normalise the source obs replicate labels",
        ],
    )


def _boolish(value: str) -> bool | None:
    compact = _norm(value)
    if compact in {"true", "yes", "y", "1", "positive", "pos", "targeting", "stimulated", "stim"}:
        return True
    if compact in {"false", "no", "n", "0", "negative", "neg", "nontargeting", "non_targeting", "unstimulated", "unstim"}:
        return False
    return None


def _control_from_values(values: Sequence[str], dataset: str, explicit: bool) -> tuple[list[bool | None], str]:
    result: list[bool | None] = []
    for value in values:
        compact = _norm(value)
        parsed = _boolish(value) if explicit else None
        if explicit and parsed is not None:
            result.append(parsed)
            continue
        # A control-status field can use prose, while a drug/guide identifier
        # uses the same markers as scPerturb's DMSO/NT controls.
        is_control = bool(re.search(r"(?:^|_)(dmso|vehicle|veh|control|ctrl|untreated|mock|negative|neg|nt|nontargeting|non_targeting|scramble)(?:$|_)", compact))
        if not is_control:
            is_control = compact in {"dmso", "vehicle", "control", "ctrl", "nt", "nontargeting", "non_targeting", "negativecontrol"}
        result.append(is_control)
    reason = "explicit control-status field" if explicit else "inferred from control/vehicle/NT identifier values"
    return result, reason


def _targeting_from_values(values: Sequence[str], field_name: str | None) -> tuple[list[bool | None], str]:
    result: list[bool | None] = []
    explicit = field_name is not None
    for value in values:
        parsed = _boolish(value) if explicit else None
        compact = _norm(value)
        if parsed is not None:
            result.append(parsed)
            continue
        is_nt = bool(re.search(r"(?:nontarget|non_target|negative|neg|scramble|control|ctrl|nt)", compact))
        if is_nt:
            result.append(False)
        elif compact in {"__na__", "nan", "none", "unknown"}:
            result.append(None)
        else:
            # For guide_type prose, targeting/perturbing are positive markers.
            is_targeting = bool(re.search(r"target|perturb|ko|knockout|gene", compact))
            result.append(is_targeting if is_targeting else None)
    return result, ("explicit targeting-status field" if explicit else "inferred from guide identifiers")


def _stimulated_from_values(values: Sequence[str], field_name: str | None, assume_stimulated: bool) -> tuple[list[bool | None], str, list[str]]:
    warnings: list[str] = []
    if field_name is None:
        if assume_stimulated:
            warnings.append("no stimulation field was found; --assume-stimulated marked every cell as stimulated")
            return [True for _ in values], "explicit --assume-stimulated override", warnings
        return [None for _ in values], "stimulation field absent", warnings
    result: list[bool | None] = []
    for value in values:
        parsed = _boolish(value)
        compact = _norm(value)
        if parsed is not None:
            result.append(parsed)
        elif "unstim" in compact or "nostim" in compact or "withoutifn" in compact or compact in {"0", "false"}:
            result.append(False)
        elif "stim" in compact or "ifn" in compact or "interferon" in compact or "activated" in compact:
            result.append(True)
        else:
            result.append(None)
    warnings.extend(
        f"unrecognised stimulation value {value!r} was excluded from the main analysis"
        for value, parsed in zip(values, result)
        if parsed is None
    )
    return result, f"inferred from {field_name!r} values", warnings


def _matrix_shape(matrix: Any) -> tuple[int, int]:
    shape = getattr(matrix, "shape", None)
    if shape is None or len(shape) != 2:
        raise BlockedError(f"AnnData matrix is not two-dimensional: shape={shape}", reasons=["matrix is not 2D"])
    return int(shape[0]), int(shape[1])


def _sample_matrix(matrix: Any, np: Any, sp: Any, n_rows: int, n_cols: int) -> tuple[Any, bool, float, float, bool]:
    if n_rows == 0 or n_cols == 0:
        return np.zeros((0, 0), dtype=np.float64), False, 0.0, 0.0, False
    rows = np.unique(np.linspace(0, n_rows - 1, num=min(32, n_rows), dtype=np.int64))
    cols = np.unique(np.linspace(0, n_cols - 1, num=min(256, n_cols), dtype=np.int64))
    try:
        sample = matrix[rows, :][:, cols]
    except Exception:
        try:
            sample = matrix[rows][:, cols]
        except Exception:
            sample = matrix[np.ix_(rows, cols)]
    if sp.issparse(sample):
        sample = sample.toarray()
    elif hasattr(sample, "to_numpy"):
        sample = sample.to_numpy()
    sample = np.asarray(sample, dtype=np.float64)
    finite = np.isfinite(sample)
    if not finite.any():
        return sample, False, 0.0, 0.0, False
    values = sample[finite]
    nonnegative = bool(np.all(values >= 0))
    integer_fraction = float(np.mean(np.isclose(values, np.rint(values), atol=1e-6, rtol=0)))
    positive = values[values > 0]
    count_like_fraction = integer_fraction if nonnegative else 0.0
    # Raw count matrices are nonnegative and overwhelmingly integer-valued.
    # All-zero matrices are accepted as count-like for schema purposes but a
    # zero library is recorded later and cannot yield a valid profile.
    return sample, count_like_fraction >= 0.98 and nonnegative, count_like_fraction, integer_fraction, nonnegative


def _var_names(adata: Any, source_kind: str) -> list[str]:
    if source_kind == "raw":
        raw = getattr(adata, "raw", None)
        if raw is None:
            return []
        names = getattr(raw, "var_names", None)
    else:
        names = getattr(adata, "var_names", None)
    if names is None:
        return []
    return [_string(value) for value in list(names)]


def _feature_type_field(var: Any) -> str | None:
    if var is None or not hasattr(var, "columns"):
        return None
    aliases = ["feature_types", "feature_type", "modality", "assay", "featuretype", "gene_type", "type"]
    by_norm = {_norm(name): str(name) for name in list(var.columns)}
    for alias in aliases:
        if _norm(alias) in by_norm:
            return by_norm[_norm(alias)]
    return None


def _feature_type_label(value: str) -> str:
    compact = _norm(value)
    if "antibody" in compact or compact in {"adt", "protein", "surfaceprotein", "surfaceantibody", "antibodycapture"}:
        return "adt"
    if "hto" in compact or "hash" in compact:
        return "hto"
    if "geneexpression" in compact or compact in {"rna", "gex", "gene", "mrna"}:
        return "rna"
    return "other"


def _matrix_for_source(adata: Any, source_kind: str, key: str | None) -> Any:
    if source_kind == "X":
        return adata.X
    if source_kind == "raw":
        raw = getattr(adata, "raw", None)
        if raw is None:
            raise BlockedError("requested adata.raw but adata.raw is absent", reasons=["adata.raw is absent"])
        return raw.X
    if source_kind == "layer":
        layers = getattr(adata, "layers", None)
        if layers is None or key not in layers:
            raise BlockedError(f"requested AnnData layer {key!r} is absent", reasons=[f"missing layer: {key}"])
        return layers[key]
    if source_kind == "obsm":
        obsm = getattr(adata, "obsm", None)
        if obsm is None or key not in obsm:
            raise BlockedError(f"requested AnnData obsm key {key!r} is absent", reasons=[f"missing obsm key: {key}"])
        return obsm[key]
    raise BlockedError(f"unknown matrix source kind {source_kind!r}", reasons=["internal source-kind error"])


def _make_matrix_spec(
    adata: Any,
    *,
    source_kind: str,
    key: str | None,
    requested: str | None,
    feature_indices: Sequence[int] | None = None,
    feature_names: Sequence[str] | None = None,
    names_inferred: bool = False,
) -> MatrixSpec:
    _, _, np, _, sp = _import_runtime()
    matrix = _matrix_for_source(adata, source_kind, key)
    n_obs, n_features = _matrix_shape(matrix)
    if n_obs != int(adata.n_obs):
        raise BlockedError(
            f"matrix {source_kind}:{key} has {n_obs} observations but AnnData has {adata.n_obs}",
            reasons=[f"row count mismatch for {source_kind}:{key}"],
        )
    raw_names = list(feature_names) if feature_names is not None else _var_names(adata, source_kind)
    if len(raw_names) != n_features:
        raw_names = [f"feature_{index:05d}" for index in range(n_features)]
        names_inferred = True
    indices = list(feature_indices) if feature_indices is not None else list(range(n_features))
    if not indices:
        raise BlockedError(f"matrix {source_kind}:{key} has no selected features", reasons=["zero selected features"])
    selected_names = [raw_names[index] for index in indices]
    _, count_like, count_fraction, integer_fraction, nonnegative = _sample_matrix(
        matrix, np, sp, n_obs, n_features
    )
    var = getattr(adata, "var", None) if source_kind != "raw" else getattr(getattr(adata, "raw", None), "var", None)
    type_field = _feature_type_field(var)
    type_values: list[str] = []
    if type_field is not None:
        try:
            type_values = sorted({_string(value) for value in var[type_field].tolist()})
        except Exception:
            type_values = []
    return MatrixSpec(
        kind=source_kind,
        key=key,
        matrix=matrix,
        n_obs=n_obs,
        n_features=n_features,
        feature_names=selected_names,
        feature_indices=indices,
        requested=requested,
        count_like=count_like,
        count_like_fraction=count_fraction,
        integer_fraction=integer_fraction,
        nonnegative=nonnegative,
        feature_type_field=type_field,
        feature_type_values=type_values,
        names_inferred=names_inferred,
    )


def _candidate_matrix_names(adata: Any) -> list[str]:
    layers = getattr(adata, "layers", None)
    if layers is None:
        return []
    try:
        return [str(key) for key in layers.keys()]
    except Exception:
        return []


def _resolve_rna(
    adata: Any,
    requested: str | None,
) -> MatrixSpec:
    """Resolve a raw count-like RNA matrix and remove ADT/HTO var features."""

    candidates: list[tuple[str, str | None]] = []
    if requested:
        if requested.lower() == "x":
            candidates.append(("X", None))
        elif requested.lower() == "raw":
            candidates.append(("raw", None))
        elif requested.startswith("layer:"):
            candidates.append(("layer", requested.split(":", 1)[1]))
        else:
            candidates.append(("layer", requested))
    else:
        preferred = ["counts", "raw_counts", "rawcounts", "rna_counts", "gene_expression", "expression_counts", "umi_counts"]
        layer_names = _candidate_matrix_names(adata)
        by_norm = {_norm(name): name for name in layer_names}
        for name in preferred:
            if _norm(name) in by_norm:
                candidates.append(("layer", by_norm[_norm(name)]))
        # Include remaining layers in a stable order, then X and raw.  The
        # count-like check decides among candidates; preference only breaks ties.
        for name in sorted(layer_names):
            item = ("layer", name)
            if item not in candidates:
                candidates.append(item)
        candidates.extend([("X", None), ("raw", None)])
    failures: list[str] = []
    for kind, key in candidates:
        try:
            matrix = _matrix_for_source(adata, kind, key)
            _, n_features = _matrix_shape(matrix)
            var = getattr(adata, "var", None) if kind != "raw" else getattr(getattr(adata, "raw", None), "var", None)
            names = _var_names(adata, kind)
            type_field = _feature_type_field(var)
            selected = list(range(n_features))
            if type_field is not None:
                labels = [_feature_type_label(_string(value)) for value in var[type_field].tolist()]
                rna_idx = [index for index, label in enumerate(labels) if label == "rna"]
                adt_idx = [index for index, label in enumerate(labels) if label == "adt"]
                if rna_idx:
                    selected = rna_idx
                elif adt_idx and len(adt_idx) == n_features:
                    failures.append(f"{kind}:{key} is annotated as ADT-only")
                    continue
            spec = _make_matrix_spec(
                adata,
                source_kind=kind,
                key=key,
                requested=requested,
                feature_indices=selected,
                feature_names=[names[index] for index in selected] if len(names) == n_features else None,
            )
            if not spec.count_like:
                failures.append(f"{kind}:{key} is not nonnegative integer-like raw counts")
                continue
            if len(set(spec.feature_names)) != len(spec.feature_names):
                failures.append(f"{kind}:{key} has duplicate RNA feature names")
                continue
            return spec
        except BlockedError as exc:
            failures.extend(exc.reasons)
        except Exception as exc:
            failures.append(f"{kind}:{key} inspection failed: {exc}")
    raise BlockedError(
        "no compatible raw RNA count matrix was found",
        reasons=[
            "RNA source candidates were inspected but none was count-like",
            *failures[:30],
            "pass --rna-layer/--rna-source with a raw integer count layer if auto-detection chose incorrectly",
        ],
    )


def _obsm_feature_names(adata: Any, key: str, n_features: int) -> tuple[list[str], bool]:
    obsm = getattr(adata, "obsm", None)
    try:
        value = obsm[key]
        columns = getattr(value, "columns", None)
        if columns is not None and len(columns) == n_features:
            return [_string(item) for item in list(columns)], False
    except Exception:
        pass
    uns = getattr(adata, "uns", {})
    candidate_keys = [f"{key}_var_names", f"{key}_features", f"{key}_names", f"{key}_feature_names"]
    for candidate in candidate_keys:
        try:
            values = list(uns[candidate])
            if len(values) == n_features:
                return [_string(value) for value in values], False
        except Exception:
            pass
    try:
        nested = uns[key]
        if isinstance(nested, Mapping):
            for candidate in ("var_names", "features", "feature_names", "names", "columns"):
                values = list(nested[candidate])
                if len(values) == n_features:
                    return [_string(value) for value in values], False
    except Exception:
        pass
    return [f"ADT_{index:04d}" for index in range(n_features)], True


def _resolve_adt(adata: Any, requested: str | None, rna: MatrixSpec | None) -> MatrixSpec | None:
    _, _, np, _, sp = _import_runtime()
    # A combined X/raw matrix with var feature_types is the most authoritative
    # source when it is available, because feature names remain aligned.
    if requested is None and rna is not None and rna.feature_type_field is not None:
        var = getattr(adata, "var", None) if rna.kind != "raw" else getattr(getattr(adata, "raw", None), "var", None)
        labels = [_feature_type_label(_string(value)) for value in var[rna.feature_type_field].tolist()]
        adt_idx = [index for index, label in enumerate(labels) if label == "adt"]
        if adt_idx:
            # rna.matrix may be the same layer/X; if it is a layer, the feature
            # type annotation is still valid because layers share var order.
            raw_names = _var_names(adata, rna.kind)
            names = [raw_names[index] for index in adt_idx] if len(raw_names) == len(labels) else None
            try:
                return _make_matrix_spec(
                    adata,
                    source_kind=rna.kind,
                    key=rna.key,
                    requested="var:ADT",
                    feature_indices=adt_idx,
                    feature_names=names,
                )
            except BlockedError:
                pass
    candidates: list[tuple[str, str]] = []
    if requested:
        if requested.startswith("obsm:"):
            candidates.append(("obsm", requested.split(":", 1)[1]))
        elif requested.startswith("layer:"):
            candidates.append(("layer", requested.split(":", 1)[1]))
        else:
            # Prefer an exact obsm key, then an exact layer key.
            obsm = getattr(adata, "obsm", {})
            layers = getattr(adata, "layers", {})
            if requested in obsm:
                candidates.append(("obsm", requested))
            elif requested in layers:
                candidates.append(("layer", requested))
            else:
                candidates.append(("obsm", requested))
    else:
        aliases = ("adt", "protein", "protein_expression", "protein_counts", "antibody_capture", "surface_protein", "surface_antibody")
        for container_name in ("obsm", "layers"):
            container = getattr(adata, container_name, {})
            try:
                keys = [str(key) for key in container.keys()]
            except Exception:
                keys = []
            by_norm = {_norm(key): key for key in keys}
            for alias in aliases:
                if _norm(alias) in by_norm and (container_name[:-1], by_norm[_norm(alias)]) not in candidates:
                    candidates.append((container_name[:-1], by_norm[_norm(alias)]))
            for key in sorted(keys):
                if any(token in _norm(key) for token in ("adt", "protein", "antibody", "surface")):
                    item = (container_name[:-1], key)
                    if item not in candidates:
                        candidates.append(item)
    failures: list[str] = []
    for kind, key in candidates:
        try:
            matrix = _matrix_for_source(adata, kind, key)
            n_obs, n_features = _matrix_shape(matrix)
            if n_obs != int(adata.n_obs) or n_features < 1:
                failures.append(f"{kind}:{key} shape={getattr(matrix, 'shape', None)} is not n_obs x n_adt")
                continue
            names, inferred = _obsm_feature_names(adata, key, n_features) if kind == "obsm" else (_var_names(adata, "X"), False)
            if len(names) != n_features:
                names = [f"ADT_{index:04d}" for index in range(n_features)]
                inferred = True
            spec = _make_matrix_spec(
                adata,
                source_kind=kind,
                key=key,
                requested=requested,
                feature_indices=list(range(n_features)),
                feature_names=names,
                names_inferred=inferred,
            )
            if not spec.count_like:
                failures.append(f"{kind}:{key} is not nonnegative integer-like ADT counts")
                continue
            if len(set(spec.feature_names)) != len(spec.feature_names):
                failures.append(f"{kind}:{key} has duplicate ADT feature names")
                continue
            return spec
        except BlockedError as exc:
            failures.extend(exc.reasons)
        except Exception as exc:
            failures.append(f"{kind}:{key} inspection failed: {exc}")
    # ADT is required for a full Papalexi audit but optional for sci-Plex3.
    if requested or failures:
        return None
    return None


def _series_for(obs: Any, field_name: str | None, n: int) -> list[str]:
    if field_name is None:
        return ["__NA__"] * n
    return _series_values(obs[field_name])


def _as_bool_array(values: Sequence[bool | None], np: Any) -> Any:
    return np.asarray([value is True for value in values], dtype=bool)


def _optional_bool_array(values: Sequence[bool | None], np: Any) -> Any:
    return np.asarray([value if value is not None else False for value in values], dtype=bool)


def _group_indices(mask: Any, keys: Sequence[Sequence[str]], np: Any) -> dict[tuple[str, ...], Any]:
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    selected = np.flatnonzero(mask)
    for index in selected.tolist():
        groups[tuple(key[index] for key in keys)].append(int(index))
    return {key: np.asarray(indices, dtype=np.int64) for key, indices in groups.items()}


def _unique_non_na(values: Sequence[str]) -> list[str]:
    return sorted({value for value in values if value != "__NA__"})


def _make_audit_rows(
    dataset: str,
    obs: Any,
    field_map: Mapping[str, str | None],
    rep_ids: Sequence[str],
    replicate_map: Mapping[str, str],
    is_control: Sequence[bool | None],
    is_stimulated: Sequence[bool | None],
    is_targeting: Sequence[bool | None],
    rna: MatrixSpec,
    adt: MatrixSpec | None,
    np: Any,
) -> list[dict[str, Any]]:
    n = int(len(obs))
    raw_reps = _series_for(obs, field_map.get("replicate"), n)
    canonical_reps = [replicate_map.get(value, _normalise_replicate(value, dataset)) if value != "__NA__" else "__NA__" for value in raw_reps]
    plate_values = _series_for(obs, field_map.get("plate"), n)
    if dataset == "sciplex3":
        values = {
            "drug_name": _series_for(obs, field_map.get("drug_name"), n),
            "cell_line": _series_for(obs, field_map.get("cell_line"), n),
            "dose": _series_for(obs, field_map.get("dose"), n),
        }
        keys = [values["drug_name"], values["cell_line"], values["dose"], canonical_reps, plate_values]
    else:
        values = {
            "guide_id": _series_for(obs, field_map.get("guide_id"), n),
            "target_gene": _series_for(obs, field_map.get("target_gene"), n),
            "stimulated": ["true" if value is True else "false" if value is False else "unknown" for value in is_stimulated],
        }
        keys = [values["guide_id"], values["target_gene"], canonical_reps, values["stimulated"], plate_values]
    groups = _group_indices(np.ones(n, dtype=bool), keys, np)
    rows: list[dict[str, Any]] = []
    for key, indices in sorted(groups.items(), key=lambda item: item[0]):
        if dataset == "sciplex3":
            drug, cell_line, dose, rep, plate = key
            row = {
                "drug_name": drug,
                "cell_line": cell_line,
                "dose": dose,
                "replicate_id": rep,
                "plate_id": plate,
                "control_treated": "control" if all(is_control[index] is True for index in indices) else "treated" if any(is_control[index] is False for index in indices) else "unknown",
                "is_control": all(is_control[index] is True for index in indices),
                "n_cells": int(len(indices)),
                "replicate_in_expected_set": rep in rep_ids,
                "rna_count_like": bool(rna.count_like),
                "adt_count_like": bool(adt.count_like) if adt is not None else False,
            }
        else:
            guide, gene, rep, stimulated, plate = key
            row = {
                "guide_id": guide,
                "target_gene": gene,
                "replicate_id": rep,
                "plate_id": plate,
                "stimulated": stimulated,
                "targeting": "targeting" if any(is_targeting[index] is True for index in indices) else "non_targeting" if all(is_targeting[index] is False for index in indices) else "unknown",
                "is_targeting": any(is_targeting[index] is True for index in indices),
                "mixscape_present": bool(field_map.get("mixscape")),
                "n_cells": int(len(indices)),
                "replicate_in_expected_set": rep in rep_ids,
                "rna_count_like": bool(rna.count_like),
                "adt_count_like": bool(adt.count_like) if adt is not None else False,
            }
        rows.append(row)
    return rows


def _sciplex_eligibility(
    audit_rows: Sequence[Mapping[str, Any]],
    expected_replicates: Sequence[str],
    thresholds: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_condition: dict[tuple[str, str, str], dict[str, int]] = defaultdict(dict)
    for row in audit_rows:
        if row.get("replicate_id") not in expected_replicates:
            continue
        if bool(row.get("is_control")):
            continue
        key = (_string(row.get("drug_name")), _string(row.get("cell_line")), _string(row.get("dose")))
        rep = _string(row.get("replicate_id"))
        by_condition[key][rep] = by_condition[key].get(rep, 0) + int(row.get("n_cells") or 0)
    detail: list[dict[str, Any]] = []
    drugs = sorted({key[0] for key in by_condition})
    cell_lines = sorted({key[1] for key in by_condition})
    doses = sorted({key[2] for key in by_condition})
    for drug in drugs:
        for cell_line in cell_lines:
            for dose in doses:
                key = (drug, cell_line, dose)
                counts = by_condition.get(key, {})
                row: dict[str, Any] = {
                    "drug_name": drug,
                    "cell_line": cell_line,
                    "dose": dose,
                    "expected_replicates": len(expected_replicates),
                    "n_replicates_present": sum(rep in counts for rep in expected_replicates),
                    "all_replicates_present": all(rep in counts for rep in expected_replicates),
                }
                for rep in expected_replicates:
                    row[f"n_cells_{rep}"] = int(counts.get(rep, 0))
                for threshold in thresholds:
                    eligible = all(counts.get(rep, 0) >= threshold for rep in expected_replicates)
                    row[f"eligible_n{threshold}"] = eligible
                detail.append(row)
    summary: list[dict[str, Any]] = []
    for drug in drugs:
        drug_rows = [row for row in detail if row["drug_name"] == drug]
        total = len(cell_lines) * len(doses)
        summary.append(
            {
                "drug_name": drug,
                "n_potential_conditions": total,
                "n_conditions_observed": sum(bool(row["all_replicates_present"]) for row in drug_rows),
                **{f"n_conditions_eligible_n{threshold}": sum(bool(row[f"eligible_n{threshold}"]) for row in drug_rows) for threshold in thresholds},
                **{f"all_conditions_eligible_n{threshold}": bool(drug_rows) and all(bool(row[f"eligible_n{threshold}"]) for row in drug_rows) for threshold in thresholds},
            }
        )
    return summary, detail


def _papalexi_eligibility(
    audit_rows: Sequence[Mapping[str, Any]],
    expected_replicates: Sequence[str],
    thresholds: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_guide: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    targeting: dict[tuple[str, str], bool] = {}
    for row in audit_rows:
        if row.get("replicate_id") not in expected_replicates or _string(row.get("stimulated")) != "true":
            continue
        if not bool(row.get("is_targeting")):
            continue
        guide = _string(row.get("guide_id"))
        gene = _string(row.get("target_gene"))
        key = (guide, gene)
        rep = _string(row.get("replicate_id"))
        by_guide[key][rep] = by_guide[key].get(rep, 0) + int(row.get("n_cells") or 0)
        targeting[key] = True
    detail: list[dict[str, Any]] = []
    for (guide, gene), counts in sorted(by_guide.items()):
        row: dict[str, Any] = {
            "guide_id": guide,
            "target_gene": gene,
            "targeting": True,
            "expected_replicates": len(expected_replicates),
            "n_replicates_present": sum(rep in counts for rep in expected_replicates),
            "all_replicates_present": all(rep in counts for rep in expected_replicates),
        }
        for rep in expected_replicates:
            row[f"n_cells_{rep}"] = int(counts.get(rep, 0))
        for threshold in thresholds:
            row[f"eligible_n{threshold}"] = all(counts.get(rep, 0) >= threshold for rep in expected_replicates)
        detail.append(row)
    summary = detail.copy()
    return summary, detail


def _build_context(args: argparse.Namespace, *, need_profiles: bool = False) -> AuditContext:
    _, _, np, pd, _ = _import_runtime()
    dataset = args.dataset
    input_path = args.input.resolve()
    _require_input(input_path)
    auxiliary_input_value = getattr(args, "adt_input", None)
    auxiliary_input_path = Path(auxiliary_input_value).resolve() if auxiliary_input_value else None
    if auxiliary_input_path is not None and dataset != "papalexi":
        raise BlockedError(
            "--adt-input is only valid for the Papalexi dataset",
            reasons=["separate ADT input supplied for non-Papalexi dataset"],
        )
    if auxiliary_input_path is not None:
        _require_input(auxiliary_input_path)
    adata = _read_anndata(input_path)
    adt_adata: Any | None = None
    try:
        if int(adata.n_obs) < 1 or int(adata.n_vars) < 1:
            raise BlockedError("AnnData contains no observations or variables", reasons=["empty AnnData object"])
        if auxiliary_input_path is not None:
            adt_raw = _read_anndata(auxiliary_input_path)
            try:
                adt_adata = _align_auxiliary_adata_to_reference(adata, adt_raw, np)
            except Exception:
                _close_adata(adt_raw)
                raise
        obs = adata.obs.copy()
        fields: dict[str, str | None] = {}
        field_candidates: dict[str, list[str]] = {}
        for role in FIELD_ALIASES[dataset]:
            override = getattr(args, f"{role}_field", None)
            field, candidates = _resolve_field(obs, dataset, role, override)
            fields[role] = field
            field_candidates[role] = candidates
        required = ["drug_name", "cell_line", "dose", "replicate"] if dataset == "sciplex3" else ["guide_id", "target_gene", "replicate"]
        missing = [role for role in required if fields.get(role) is None]
        if missing:
            raise BlockedError(
                f"required obs fields are absent: {missing}",
                reasons=[f"missing required obs field: {role}" for role in missing],
            )
        raw_replicate_values = _series_for(obs, fields["replicate"], len(obs))
        replicate_map, selected_reps, excluded_reps, rep_warnings = _choose_replicates(raw_replicate_values, dataset)
        canonical_reps = [replicate_map.get(value, _normalise_replicate(value, dataset)) if value != "__NA__" else "__NA__" for value in raw_replicate_values]
        selected_rep_mask = np.asarray([rep in selected_reps for rep in canonical_reps], dtype=bool)
        if not selected_rep_mask.any():
            raise BlockedError("none of the observations belong to protocol replicates", reasons=["no cells in selected independent replicates"])
        rna = _resolve_rna(adata, getattr(args, "rna_layer", None))
        adt_source_adata = adt_adata if adt_adata is not None else adata
        adt = _resolve_adt(
            adt_source_adata,
            getattr(args, "adt_key", None),
            None if adt_adata is not None else rna,
        )

        n = int(len(obs))
        control_field = fields.get("control_status") if dataset == "sciplex3" else None
        if dataset == "sciplex3":
            control_values = _series_for(obs, control_field or fields.get("drug_name"), n)
            is_control_values, control_reason = _control_from_values(control_values, dataset, control_field is not None)
            is_stimulated_values = [True for _ in range(n)]
            stimulation_reason = "not applicable to sci-Plex3"
            is_targeting_values = [False if value is True else None for value in is_control_values]
            targeting_reason = "not applicable to sci-Plex3"
            stimulation_warnings: list[str] = []
        else:
            guide_values = _series_for(obs, fields.get("guide_id"), n)
            targeting_values, targeting_reason = _targeting_from_values(
                _series_for(obs, fields.get("targeting_status"), n) if fields.get("targeting_status") else guide_values,
                fields.get("targeting_status"),
            )
            is_control_values = [False if value is True else True if value is False else None for value in targeting_values]
            control_reason = "non-targeting guide cells used as controls"
            stim_values, stimulation_reason, stimulation_warnings = _stimulated_from_values(
                _series_for(obs, fields.get("stimulated"), n),
                fields.get("stimulated"),
                bool(getattr(args, "assume_stimulated", False)),
            )
            is_stimulated_values = stim_values
        main_mask = selected_rep_mask & _as_bool_array(is_stimulated_values, np)
        audit_mask = selected_rep_mask | np.asarray([rep not in selected_reps for rep in canonical_reps], dtype=bool)
        # audit_mask is intentionally all observations with an identifiable
        # replicate, including R4/technical extras, so excluded material is
        # visible rather than silently disappearing.
        audit_mask = np.asarray([rep != "__NA__" for rep in canonical_reps], dtype=bool)
        audit_rows = _make_audit_rows(
            dataset,
            obs,
            fields,
            selected_reps,
            replicate_map,
            is_control_values,
            is_stimulated_values,
            is_targeting_values,
            rna,
            adt,
            np,
        )
        if dataset == "sciplex3":
            eligibility_rows, eligibility_detail_rows = _sciplex_eligibility(audit_rows, selected_reps, SCI_THRESHOLDS)
        else:
            eligibility_rows, eligibility_detail_rows = _papalexi_eligibility(audit_rows, selected_reps, PAPA_THRESHOLDS)
        gate_reasons: list[str] = []
        warnings = list(rep_warnings) + stimulation_warnings
        if not rna.count_like:
            gate_reasons.append("no raw integer-like RNA count matrix")
        if dataset == "sciplex3":
            if fields.get("plate") is None:
                warnings.append("plate field is absent; vehicle matching will use cell_line + biological replicate")
            if not any(bool(row.get("is_control")) for row in audit_rows):
                gate_reasons.append("no DMSO/vehicle control cells were identified")
            drug_count = len({str(row.get("drug_name")) for row in eligibility_rows})
            eligible_drugs = sum(
                1 for row in eligibility_rows if int(row.get("n_conditions_eligible_n50", 0)) > 0
            )
            detail_eligible = sum(bool(row.get("eligible_n50")) for row in eligibility_detail_rows)
            theoretical_conditions = SCI_EXPECTED_DRUGS * SCI_EXPECTED_CELL_LINES * SCI_EXPECTED_DOSES
            condition_fraction = detail_eligible / theoretical_conditions if theoretical_conditions else 0.0
            if eligible_drugs < SCI_MIN_ELIGIBLE_DRUGS:
                gate_reasons.append(
                    f"eligible drugs at n>=50 in at least one condition: {eligible_drugs} < {SCI_MIN_ELIGIBLE_DRUGS}"
                )
            if condition_fraction < SCI_MIN_CONDITION_FRACTION:
                gate_reasons.append(
                    f"n>=50 eligible condition fraction: {detail_eligible}/{theoretical_conditions}={condition_fraction:.4f} < {SCI_MIN_CONDITION_FRACTION:.2f}"
                )
            if drug_count != SCI_EXPECTED_DRUGS:
                warnings.append(f"observed {drug_count} drug identifiers; protocol reference expects {SCI_EXPECTED_DRUGS}")
        else:
            if fields.get("stimulated") is None and not getattr(args, "assume_stimulated", False):
                gate_reasons.append("stimulation field is absent; refuse to treat all cells as IFN-gamma stimulated")
            unknown_stim = sum(value is None for value in is_stimulated_values)
            if unknown_stim:
                gate_reasons.append(f"{unknown_stim} cells have unrecognised stimulation labels")
            if fields.get("targeting_status") is None and any(value is None for value in is_targeting_values):
                gate_reasons.append("targeting/non-targeting status could not be inferred for all guide values")
            if not any(value is False for value in is_targeting_values):
                gate_reasons.append("no non-targeting guide control cells were identified")
            if not any(value is True for value in is_targeting_values):
                gate_reasons.append("no targeting guide cells were identified")
            if adt is None:
                gate_reasons.append("no compatible ADT/protein count matrix was identified")
            eligible_count = sum(
                1 for row in eligibility_rows if bool(row.get("eligible_n25"))
            )
            if eligible_count < PAPA_MIN_TARGETING_GUIDES:
                gate_reasons.append(
                    f"eligible targeting gRNAs at n>=25 in R1/R2/R3: {eligible_count} < {PAPA_MIN_TARGETING_GUIDES}"
                )
            if excluded_reps:
                warnings.append(f"excluded replicate labels from main analysis: {excluded_reps}")
        gate_pass = not gate_reasons
        try:
            source_hash = None if getattr(args, "skip_source_hash", False) else _sha256(input_path)
        except Exception as exc:
            source_hash = None
            warnings.append(f"source SHA256 unavailable: {exc}")
        try:
            adt_source_hash = (
                None
                if auxiliary_input_path is None or getattr(args, "skip_source_hash", False)
                else _sha256(auxiliary_input_path)
            )
        except Exception as exc:
            adt_source_hash = None
            warnings.append(f"ADT source SHA256 unavailable: {exc}")
        return AuditContext(
            dataset=dataset,
            input_path=input_path,
            adata=adata,
            obs=obs,
            fields=fields,
            field_candidates=field_candidates,
            replicate_values_observed=sorted(set(raw_replicate_values)),
            replicate_map=replicate_map,
            selected_replicates=selected_reps,
            excluded_replicates=excluded_reps,
            rna=rna,
            adt=adt,
            adt_adata=adt_adata,
            adt_input_path=auxiliary_input_path,
            is_control=is_control_values,
            control_reason=control_reason,
            is_stimulated=is_stimulated_values,
            stimulation_reason=stimulation_reason,
            is_targeting=is_targeting_values,
            targeting_reason=targeting_reason,
            main_mask=main_mask,
            audit_mask=audit_mask,
            audit_rows=audit_rows,
            eligibility_rows=eligibility_rows,
            eligibility_detail_rows=eligibility_detail_rows,
            gate_pass=gate_pass,
            gate_reasons=gate_reasons,
            warnings=warnings,
            source_sha256=source_hash,
            adt_source_sha256=adt_source_hash,
            source_size=input_path.stat().st_size,
        )
    except Exception:
        _close_adata(adata)
        if adt_adata is not None:
            _close_adata(adt_adata)
        raise


def _audit_payload(ctx: AuditContext, args: argparse.Namespace) -> dict[str, Any]:
    n_cells = int(len(ctx.obs))
    if ctx.dataset == "sciplex3":
        treatment_cells = sum(1 for value in ctx.is_control if value is False)
        control_cells = sum(1 for value in ctx.is_control if value is True)
        eligible_n50 = sum(bool(row.get("eligible_n50")) for row in ctx.eligibility_detail_rows)
        eligibility_summary = {
            "n_drugs_observed": len(ctx.eligibility_rows),
            "n_conditions_detail": len(ctx.eligibility_detail_rows),
            "n_conditions_eligible_n25": sum(bool(row.get("eligible_n25")) for row in ctx.eligibility_detail_rows),
            "n_conditions_eligible_n50": eligible_n50,
            "n_conditions_eligible_n100": sum(bool(row.get("eligible_n100")) for row in ctx.eligibility_detail_rows),
            "n_conditions_theoretical": SCI_EXPECTED_DRUGS * SCI_EXPECTED_CELL_LINES * SCI_EXPECTED_DOSES,
        }
    else:
        treatment_cells = sum(1 for value in ctx.is_targeting if value is True)
        control_cells = sum(1 for value in ctx.is_targeting if value is False)
        eligibility_summary = {
            "n_targeting_guides_observed": len(ctx.eligibility_rows),
            "n_targeting_guides_eligible_n15": sum(bool(row.get("eligible_n15")) for row in ctx.eligibility_rows),
            "n_targeting_guides_eligible_n25": sum(bool(row.get("eligible_n25")) for row in ctx.eligibility_rows),
        }
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": _now(),
        "dataset": ctx.dataset,
        "input": {
            "path": str(ctx.input_path),
            "size_bytes": ctx.source_size,
            "sha256": ctx.source_sha256,
            "n_obs": n_cells,
            "n_vars": int(ctx.adata.n_vars),
        },
        "adt_input": {
            "path": str(ctx.adt_input_path) if ctx.adt_input_path is not None else None,
            "sha256": ctx.adt_source_sha256,
            "alignment": "ADT obs rows were reordered to the RNA obs.index; RNA obs is authoritative"
            if ctx.adt_input_path is not None
            else "ADT was resolved from the RNA AnnData object",
        },
        "obs_fields": ctx.fields,
        "obs_field_candidates": ctx.field_candidates,
        "replicates": {
            "observed_values": ctx.replicate_values_observed,
            "value_to_protocol_id": ctx.replicate_map,
            "selected": ctx.selected_replicates,
            "excluded": ctx.excluded_replicates,
            "independent_unit": "biological replicate for sci-Plex3; viral transduction replicate for Papalexi",
        },
        "matrices": {
            "rna": ctx.rna.summary(),
            "adt": ctx.adt.summary() if ctx.adt is not None else None,
        },
        "controls": {
            "interpretation": ctx.control_reason,
            "n_control_cells": control_cells,
            "n_treated_or_targeting_cells": treatment_cells,
            "plate_preferred": ctx.fields.get("plate") is not None,
            "fallback": "cell_line+replicate(+stimulated) when plate is unavailable or not matched",
        },
        "stimulation": {
            "interpretation": ctx.stimulation_reason,
            "main_filter": "stimulated only for Papalexi; all selected repeats for sci-Plex3",
            "unstimulated_in_main": False,
        },
        "eligibility": eligibility_summary,
        "gate": {
            "phase0_gate_pass": ctx.gate_pass,
            "reasons": ctx.gate_reasons,
            "warnings": ctx.warnings,
            "confirmation_loaded": False,
            "confirmation_available": False,
            "cfra_trained": False,
            "models_trained": False,
        },
        "fixed_protocol": {
            "seed": PROTOCOL_SEED,
            "sci_thresholds": list(SCI_THRESHOLDS),
            "papalexi_thresholds": list(PAPA_THRESHOLDS),
            "sci_primary_threshold": 50,
            "papalexi_primary_threshold": 25,
            "sci_split": "drug-level 60/20/20 is not opened in Phase 0",
            "papalexi_split": "gRNA-level 60/20/20 is not opened in Phase 0",
            "hvg_selection": "not performed in Phase 0; no confirmation data read",
        },
        "cli": {
            "argv": sys.argv,
            "rna_layer": getattr(args, "rna_layer", None),
            "adt_key": getattr(args, "adt_key", None),
            "adt_input": str(getattr(args, "adt_input", "")) if getattr(args, "adt_input", None) else None,
        },
    }


def _write_audit_outputs(ctx: AuditContext, output_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    prefix = "SCIPLEX3" if ctx.dataset == "sciplex3" else "PAPALEXI"
    audit_name = f"{ctx.dataset}_metadata_audit.csv"
    _write_csv(output_root / audit_name, ctx.audit_rows)
    _write_csv(output_root / f"{prefix}_ELIGIBILITY.csv", ctx.eligibility_rows)
    _write_csv(output_root / f"{prefix}_CONDITION_ELIGIBILITY.csv", ctx.eligibility_detail_rows)
    if ctx.dataset == "papalexi":
        adt_lines = [] if ctx.adt is None else ctx.adt.feature_names
        _write_text(output_root / "ADT_FEATURES.txt", "\n".join(adt_lines) + ("\n" if adt_lines else ""))
    payload = _audit_payload(ctx, args)
    _write_json(output_root / "AUDIT.json", payload)
    status = "PASS" if ctx.gate_pass else "BLOCKED"
    _write_text(
        output_root / "GATE_STATUS.txt",
        f"{status}\n" + ("\n".join(ctx.gate_reasons) if ctx.gate_reasons else "Phase 0 audit passed; profiles may be built.") + "\n",
    )
    return payload


def _minimal_blocked_payload(args: argparse.Namespace, error: Exception) -> dict[str, Any]:
    reasons = list(getattr(error, "reasons", [str(error)]))
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": _now(),
        "dataset": getattr(args, "dataset", None),
        "input": {"path": str(getattr(args, "input", ""))},
        "gate": {
            "phase0_gate_pass": False,
            "reasons": reasons,
            "confirmation_loaded": False,
            "confirmation_available": False,
            "cfra_trained": False,
            "models_trained": False,
        },
        "blocked_error": str(error),
        "cli": {"argv": sys.argv},
    }


def _load_matrix_rows(matrix: Any, indices: Any, feature_indices: Sequence[int], np: Any, sp: Any, chunk_cells: int) -> Any:
    indices = np.asarray(indices, dtype=np.int64)
    output = np.zeros(len(feature_indices), dtype=np.float64)
    if len(indices) == 0:
        return output
    feature_indices_arr = np.asarray(feature_indices, dtype=np.int64)
    for start in range(0, len(indices), chunk_cells):
        chunk_indices = indices[start : start + chunk_cells]
        try:
            block = matrix[chunk_indices, :]
        except Exception:
            try:
                block = matrix[chunk_indices]
            except Exception:
                block = matrix[np.asarray(chunk_indices)]
        if hasattr(block, "iloc"):
            block = block.iloc[chunk_indices]
        try:
            block = block[:, feature_indices_arr]
        except Exception:
            block = block[:, list(feature_indices_arr)]
        if sp.issparse(block):
            output += np.asarray(block.sum(axis=0)).ravel().astype(np.float64, copy=False)
        else:
            if hasattr(block, "to_numpy"):
                block = block.to_numpy()
            output += np.asarray(block, dtype=np.float64).sum(axis=0)
    return output


def _normalise_counts(counts: Any, np: Any) -> tuple[Any, float, bool]:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(counts))
    if not math.isfinite(total) or total <= 0:
        return np.zeros_like(counts, dtype=np.float32), total, False
    values = np.log1p((counts / total) * DEFAULT_CPM).astype(np.float32)
    return values, total, True


def _profile_group_indices(ctx: AuditContext, np: Any) -> tuple[dict[Any, Any], dict[Any, Any], list[dict[str, Any]]]:
    """Return treatment and control row groups at the independent-repeat unit."""

    n = int(len(ctx.obs))
    fields = ctx.fields
    rep_raw = _series_for(ctx.obs, fields.get("replicate"), n)
    rep_values = [ctx.replicate_map.get(value, _normalise_replicate(value, ctx.dataset)) for value in rep_raw]
    plate = _series_for(ctx.obs, fields.get("plate"), n)
    selected = np_array = __import__("numpy").asarray  # local import keeps module import dependency-free
    import numpy as np  # type: ignore
    selected_mask = np.asarray([rep in ctx.selected_replicates for rep in rep_values], dtype=bool)
    stim = np.asarray([value is True for value in ctx.is_stimulated], dtype=bool)
    main_mask = selected_mask & stim
    if ctx.dataset == "sciplex3":
        drug = _series_for(ctx.obs, fields.get("drug_name"), n)
        cell = _series_for(ctx.obs, fields.get("cell_line"), n)
        dose = _series_for(ctx.obs, fields.get("dose"), n)
        control = np.asarray([value is True for value in ctx.is_control], dtype=bool)
        treated = np.asarray([value is False for value in ctx.is_control], dtype=bool)
        treatment_keys = [drug, cell, dose, rep_values]
        control_keys = [cell, rep_values, plate]
        treatment_groups = _group_indices(main_mask & treated, treatment_keys, np)
        control_groups = _group_indices(main_mask & control, control_keys, np)
        rows: list[dict[str, Any]] = []
        for key, indices in sorted(treatment_groups.items(), key=lambda item: item[0]):
            drug_value, cell_value, dose_value, rep_value = key
            plates = sorted({plate[index] for index in indices})
            matched = []
            for plate_value in plates:
                matched.extend(control_groups.get((cell_value, rep_value, plate_value), []).tolist())
            match_level = "same_cell_line+replicate+plate"
            if not matched:
                # Only use the lower-resolution fallback when the source has no
                # usable plate field or the treatment rows explicitly have no plate.
                if fields.get("plate") is None or all(value == "__NA__" for value in plates):
                    matched = control_groups.get((cell_value, rep_value, "__all__"), []).tolist()
                    match_level = "same_cell_line+replicate"
                else:
                    rows.append({
                        "drug_name": drug_value,
                        "cell_line": cell_value,
                        "dose": dose_value,
                        "replicate_id": rep_value,
                        "plate_id": ";".join(plates),
                        "n_cells": int(len(indices)),
                        "n_control_cells": 0,
                        "control_match_level": "missing_same_plate_control",
                        "_treatment_indices": indices,
                        "_control_indices": np.asarray([], dtype=np.int64),
                    })
                    continue
            rows.append({
                "drug_name": drug_value,
                "cell_line": cell_value,
                "dose": dose_value,
                "replicate_id": rep_value,
                "plate_id": ";".join(plates),
                "n_cells": int(len(indices)),
                "n_control_cells": int(len(set(matched))),
                "control_match_level": match_level,
                "_treatment_indices": indices,
                "_control_indices": np.asarray(sorted(set(matched)), dtype=np.int64),
            })
        return treatment_groups, control_groups, rows
    guide = _series_for(ctx.obs, fields.get("guide_id"), n)
    gene = _series_for(ctx.obs, fields.get("target_gene"), n)
    targeting = np.asarray([value is True for value in ctx.is_targeting], dtype=bool)
    non_targeting = np.asarray([value is False for value in ctx.is_targeting], dtype=bool)
    stimulation_values = ["true" if value is True else "false" if value is False else "unknown" for value in ctx.is_stimulated]
    treatment_keys = [guide, gene, rep_values, stimulation_values]
    control_keys = [rep_values, stimulation_values, plate]
    treatment_groups = _group_indices(main_mask & targeting, treatment_keys, np)
    control_groups = _group_indices(main_mask & non_targeting, control_keys, np)
    rows = []
    for key, indices in sorted(treatment_groups.items(), key=lambda item: item[0]):
        guide_value, gene_value, rep_value, stim_value = key
        if stim_value != "true":
            continue
        plates = sorted({plate[index] for index in indices})
        matched: list[int] = []
        for plate_value in plates:
            matched.extend(control_groups.get((rep_value, stim_value, plate_value), []).tolist())
        match_level = "same_replicate+stimulated+plate"
        if not matched and (fields.get("plate") is None or all(value == "__NA__" for value in plates)):
            matched = control_groups.get((rep_value, stim_value, "__all__"), []).tolist()
            match_level = "same_replicate+stimulated"
        rows.append({
            "guide_id": guide_value,
            "target_gene": gene_value,
            "replicate_id": rep_value,
            "stimulated": stim_value,
            "plate_id": ";".join(plates),
            "n_cells": int(len(indices)),
            "n_control_cells": int(len(set(matched))),
            "control_match_level": match_level if matched else "missing_same_plate_control",
            "_treatment_indices": indices,
            "_control_indices": np.asarray(sorted(set(matched)), dtype=np.int64),
        })
    return treatment_groups, control_groups, rows


def _append_h5_rows(
    datasets: Mapping[str, Any],
    value_rows: list[Any],
    metadata_rows: list[Mapping[str, Any]],
    np: Any,
) -> None:
    if not value_rows:
        return
    values = np.asarray(value_rows, dtype=np.float32)
    start = int(datasets["delta"].shape[0])
    end = start + len(value_rows)
    datasets["delta"].resize(end, axis=0)
    datasets["delta"][start:end] = values
    for name, dataset in datasets.items():
        if name == "delta":
            continue
        dataset.resize(end, axis=0)
        dataset[start:end] = [str(row.get(name, "")) for row in metadata_rows]
    value_rows.clear()
    metadata_rows.clear()


def _build_profiles(ctx: AuditContext, output_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not ctx.gate_pass:
        raise BlockedError(
            "profile construction is locked until the Phase-0 audit gate passes",
            reasons=ctx.gate_reasons or ["Phase-0 gate did not pass"],
        )
    _, h5py, np, _, sp = _import_runtime()
    output_root.mkdir(parents=True, exist_ok=True)
    _, _, groups = _profile_group_indices(ctx, np)
    # Empty/missing matched controls are an audit issue, not a reason to
    # fabricate a profile.  The output remains valid but only includes rows
    # with a real control baseline.
    profile_rows = [row for row in groups if int(row["n_control_cells"]) > 0]
    if not profile_rows:
        raise BlockedError(
            "no treatment group has a matched control baseline",
            reasons=["vehicle/non-targeting pseudo-bulk could not be matched to any treatment group"],
        )
    rna_features = list(ctx.rna.feature_names)
    adt_features = list(ctx.adt.feature_names) if ctx.adt is not None else []
    requested_modalities = getattr(args, "modalities", "rna")
    modalities = [value.strip().lower() for value in requested_modalities.split(",") if value.strip()]
    if any(value not in {"rna", "adt", "protein"} for value in modalities):
        raise BlockedError("--modalities accepts comma-separated rna and adt", reasons=[f"invalid modalities: {modalities}"])
    if "protein" in modalities:
        modalities = ["adt" if value == "protein" else value for value in modalities]
    if "adt" in modalities and ctx.adt is None:
        raise BlockedError("ADT profiles were requested but no ADT source was audited", reasons=["missing ADT source"])
    if not modalities:
        raise BlockedError("at least one profile modality is required", reasons=["empty --modalities"])
    slug = "sciplex3" if ctx.dataset == "sciplex3" else "papalexi"
    out_path = output_root / f"{slug}_repeat_profiles.h5"
    if out_path.exists() and not getattr(args, "overwrite", False):
        raise BlockedError(
            f"refusing to overwrite existing profile file {out_path}; pass --overwrite explicitly",
            reasons=[f"output exists: {out_path}"],
        )
    feature_names = rna_features if modalities == ["rna"] else adt_features if modalities == ["adt"] else [f"RNA::{name}" for name in rna_features] + [f"ADT::{name}" for name in adt_features]
    if modalities == ["rna"]:
        dimensions = [len(rna_features)]
    elif modalities == ["adt"]:
        dimensions = [len(adt_features)]
    else:
        dimensions = [len(rna_features), len(adt_features)]
    n_features = sum(dimensions)
    if n_features < 1:
        raise BlockedError("profile source has zero selected features", reasons=["zero profile dimensions"])
    chunk_cells = max(1, int(getattr(args, "chunk_cells", DEFAULT_CHUNK_CELLS)))
    metadata_names = sorted(
        {
            key
            for row in profile_rows
            for key in row.keys()
            if not str(key).startswith("_")
        }
        | {"profile_id", "dataset", "n_rna_features", "n_adt_features", "rna_total_treated", "rna_total_control", "adt_total_treated", "adt_total_control"}
    )
    with h5py.File(out_path, "w") as handle:
        handle.attrs["schema_version"] = 1
        handle.attrs["protocol_version"] = PROTOCOL_VERSION
        handle.attrs["dataset"] = ctx.dataset
        handle.attrs["profile_value"] = "log1p(CPM=10000) treatment-minus-matched-control"
        handle.attrs["control_policy"] = "same plate preferred; fallback same independent replicate"
        handle.attrs["confirmation_loaded"] = False
        handle.attrs["cfra_trained"] = False
        handle.attrs["modalities"] = ",".join(modalities)
        handle.attrs["source_sha256"] = ctx.source_sha256 or ""
        values_group = handle.create_group("profiles")
        feature_group = handle.create_group("var")
        obs_group = handle.create_group("obs")
        values_group.create_dataset("delta", shape=(0, n_features), maxshape=(None, n_features), dtype="f4", chunks=(1, n_features), compression="gzip")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        feature_group.create_dataset("feature_names", data=np.asarray(feature_names, dtype=object), dtype=string_dtype)
        feature_group.attrs["rna_features"] = len(rna_features)
        feature_group.attrs["adt_features"] = len(adt_features)
        datasets: dict[str, Any] = {"delta": values_group["delta"]}
        for name in metadata_names:
            datasets[name] = obs_group.create_dataset(name, shape=(0,), maxshape=(None,), dtype=string_dtype, chunks=(1024,))
        value_buffer: list[Any] = []
        metadata_buffer: list[Mapping[str, Any]] = []
        manifest_rows: list[dict[str, Any]] = []
        for index, row in enumerate(profile_rows):
            treatment_indices = row["_treatment_indices"]
            control_indices = row["_control_indices"]
            rna_treated = _load_matrix_rows(ctx.rna.matrix, treatment_indices, ctx.rna.feature_indices, np, sp, chunk_cells)
            rna_control = _load_matrix_rows(ctx.rna.matrix, control_indices, ctx.rna.feature_indices, np, sp, chunk_cells)
            rna_treated_norm, rna_total_treated, ok_treated = _normalise_counts(rna_treated, np)
            rna_control_norm, rna_total_control, ok_control = _normalise_counts(rna_control, np)
            delta_parts: list[Any] = []
            metadata = {key: _csv_value(row.get(key, "")) for key in metadata_names}
            metadata.update(
                {
                    "profile_id": f"{slug}_{index:06d}",
                    "dataset": ctx.dataset,
                    "n_rna_features": len(rna_features),
                    "n_adt_features": len(adt_features),
                    "rna_total_treated": rna_total_treated,
                    "rna_total_control": rna_total_control,
                    "adt_total_treated": "",
                    "adt_total_control": "",
                    "profile_valid_library": bool(ok_treated and ok_control),
                }
            )
            if "rna" in modalities:
                delta_parts.append(rna_treated_norm - rna_control_norm)
            if "adt" in modalities:
                assert ctx.adt is not None
                adt_treated = _load_matrix_rows(ctx.adt.matrix, treatment_indices, ctx.adt.feature_indices, np, sp, chunk_cells)
                adt_control = _load_matrix_rows(ctx.adt.matrix, control_indices, ctx.adt.feature_indices, np, sp, chunk_cells)
                adt_treated_norm, adt_total_treated, adt_ok_treated = _normalise_counts(adt_treated, np)
                adt_control_norm, adt_total_control, adt_ok_control = _normalise_counts(adt_control, np)
                delta_parts.append(adt_treated_norm - adt_control_norm)
                metadata["adt_total_treated"] = adt_total_treated
                metadata["adt_total_control"] = adt_total_control
                metadata["profile_valid_library"] = bool(metadata["profile_valid_library"] and adt_ok_treated and adt_ok_control)
            metadata_buffer.append(metadata)
            value_buffer.append(np.concatenate(delta_parts).astype(np.float32, copy=False))
            manifest_rows.append(metadata.copy())
            if len(value_buffer) >= 32:
                _append_h5_rows(datasets, value_buffer, metadata_buffer, np)
        _append_h5_rows(datasets, value_buffer, metadata_buffer, np)
        handle.attrs["n_profiles"] = int(values_group["delta"].shape[0])
    manifest_fields = sorted({key for row in manifest_rows for key in row.keys()})
    _write_csv(output_root / "PROFILE_MANIFEST.csv", manifest_rows, manifest_fields)
    _write_json(
        output_root / "PROFILE_SCHEMA.json",
        {
            "schema_version": 1,
            "file": str(out_path),
            "dataset": ctx.dataset,
            "modalities": modalities,
            "feature_names": feature_names,
            "hdf5": {
                "profiles/delta": "float32 [n_profiles, n_features], treatment-minus-matched-control log1p(CPM)",
                "obs/*": "UTF-8 row metadata, one value per profile",
                "var/feature_names": "UTF-8 feature names; RNA first then ADT when both requested",
            },
            "confirmation_loaded": False,
            "cfra_trained": False,
        },
    )
    return {
        "profile_file": str(out_path),
        "manifest": str(output_root / "PROFILE_MANIFEST.csv"),
        "n_profiles": len(manifest_rows),
        "n_features": n_features,
        "modalities": modalities,
        "confirmation_loaded": False,
        "cfra_trained": False,
    }


def _read_audit_gate(path: Path, dataset: str, input_path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BlockedError(f"audit JSON does not exist: {path}", reasons=[f"missing audit JSON: {path}"])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BlockedError(f"invalid audit JSON {path}: {exc}", reasons=[f"invalid audit JSON: {exc}"]) from exc
    if payload.get("dataset") != dataset:
        raise BlockedError(
            f"audit dataset {payload.get('dataset')!r} does not match requested {dataset!r}",
            reasons=["audit dataset mismatch"],
        )
    recorded_path = Path(str(payload.get("input", {}).get("path", ""))).resolve()
    if recorded_path != input_path.resolve():
        raise BlockedError(
            f"audit input {recorded_path} does not match requested {input_path.resolve()}",
            reasons=["audit input path mismatch; rerun audit on the exact .h5ad file"],
        )
    gate = payload.get("gate", {})
    if gate.get("confirmation_loaded") or gate.get("cfra_trained") or gate.get("models_trained"):
        raise BlockedError(
            "audit metadata claims a model or confirmation was already opened",
            reasons=["refuse to use an audit artifact that violates the Phase-0 boundary"],
        )
    if not bool(gate.get("phase0_gate_pass")):
        raise BlockedError(
            "audit gate is blocked; profile construction is not allowed",
            reasons=[str(reason) for reason in gate.get("reasons", [])] or ["phase0_gate_pass=false"],
        )
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", choices=("sciplex3", "papalexi"), required=True)
    common.add_argument("--input", type=Path, required=True, help="preprocessed scPerturb .h5ad input")
    common.add_argument(
        "--adt-input",
        type=Path,
        default=None,
        help="Papalexi-only ADT/protein .h5ad; obs.index must match RNA input exactly (order may differ)",
    )
    common.add_argument("--rna-layer", default=None, help="layer name, X, or raw; auto-detected when omitted")
    common.add_argument("--adt-key", default=None, help="obsm:key, layer:key, or exact key; auto-detected when omitted")
    common.add_argument("--skip-source-hash", action="store_true", help="do not hash the full input file")
    common.add_argument("--replicate-field", default=None)
    common.add_argument("--plate-field", default=None)
    common.add_argument("--drug-name-field", default=None)
    common.add_argument("--cell-line-field", default=None)
    common.add_argument("--dose-field", default=None)
    common.add_argument("--control-status-field", default=None)
    common.add_argument("--guide-id-field", default=None)
    common.add_argument("--target-gene-field", default=None)
    common.add_argument("--stimulated-field", default=None)
    common.add_argument("--targeting-status-field", default=None)
    common.add_argument("--mixscape-field", default=None)
    common.add_argument("--assume-stimulated", action="store_true", help="only when the source is a known stimulated-only export")
    audit_parser = subparsers.add_parser("audit", parents=[common], help="inspect schema and write Phase-0 audit artifacts")
    audit_parser.add_argument("--output-root", type=Path, required=True)
    profiles_parser = subparsers.add_parser("profiles", parents=[common], help="build repeat profiles after a passing audit")
    profiles_parser.add_argument("--audit-json", type=Path, required=True)
    profiles_parser.add_argument("--output-root", type=Path, required=True)
    profiles_parser.add_argument("--modalities", default="rna", help="rna, adt, or rna,adt")
    profiles_parser.add_argument("--chunk-cells", type=int, default=DEFAULT_CHUNK_CELLS)
    profiles_parser.add_argument("--overwrite", action="store_true")
    run_parser = subparsers.add_parser("run", parents=[common], help="run audit, then profiles only if the gate passes")
    run_parser.add_argument("--output-root", type=Path, required=True)
    run_parser.add_argument("--modalities", default="rna", help="rna, adt, or rna,adt")
    run_parser.add_argument("--chunk-cells", type=int, default=DEFAULT_CHUNK_CELLS)
    run_parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if getattr(args, "chunk_cells", 1) < 1:
        parser.error("--chunk-cells must be positive")
    return args


def _run_audit(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    try:
        ctx = _build_context(args)
        payload = _write_audit_outputs(ctx, output_root, args)
        print(json.dumps({"status": "PASS" if ctx.gate_pass else "BLOCKED", "audit": str(output_root / "AUDIT.json"), "gate_reasons": ctx.gate_reasons}, indent=2))
        _close_adata(ctx.adata)
        return 0 if ctx.gate_pass else 2
    except Exception as exc:
        output_root.mkdir(parents=True, exist_ok=True)
        payload = _minimal_blocked_payload(args, exc)
        _write_json(output_root / "AUDIT.json", payload)
        _write_text(output_root / "GATE_STATUS.txt", "BLOCKED\n" + "\n".join(payload["gate"]["reasons"]) + "\n")
        print(json.dumps({"status": "BLOCKED", "audit": str(output_root / "AUDIT.json"), "gate_reasons": payload["gate"]["reasons"]}, indent=2), file=sys.stderr)
        return 2


def _run_profiles(args: argparse.Namespace) -> int:
    audit_payload = _read_audit_gate(args.audit_json.resolve(), args.dataset, args.input.resolve())
    try:
        ctx = _build_context(args, need_profiles=True)
        # Require the current source audit and fresh re-inspection to agree on
        # source hash whenever the audit recorded one.
        expected_hash = audit_payload.get("input", {}).get("sha256")
        if expected_hash and ctx.source_sha256 and expected_hash != ctx.source_sha256:
            raise BlockedError(
                "input SHA256 differs from the passing audit",
                reasons=["source file changed after audit; rerun Phase 0 audit"],
            )
        result = _build_profiles(ctx, args.output_root.resolve(), args)
        _write_json(args.output_root.resolve() / "PROFILE_RUN.json", result)
        print(json.dumps({"status": "PASS", **result}, indent=2))
        _close_adata(ctx.adata)
        return 0
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reasons": list(getattr(exc, "reasons", [str(exc)]))}, indent=2), file=sys.stderr)
        return 2


def _run_all(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    audit_root = root / "00_audit"
    profile_root = root / "01_profiles"
    audit_args = argparse.Namespace(**vars(args))
    audit_args.command = "audit"
    audit_args.output_root = audit_root
    code = _run_audit(audit_args)
    if code != 0:
        # The audit artifact is deliberately retained as the reasoned stop.
        print(f"Phase 0 gate blocked; profiles were not built. See {audit_root / 'AUDIT.json'}", file=sys.stderr)
        return code
    profile_args = argparse.Namespace(**vars(args))
    profile_args.command = "profiles"
    profile_args.audit_json = audit_root / "AUDIT.json"
    profile_args.output_root = profile_root
    return _run_profiles(profile_args)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "audit":
        return _run_audit(args)
    if args.command == "profiles":
        return _run_profiles(args)
    if args.command == "run":
        return _run_all(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
