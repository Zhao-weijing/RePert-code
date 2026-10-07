#!/usr/bin/env python3
"""V2 strict selected-repeat runner: treatment aggregates never enter inputs."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


HERE = Path(__file__).resolve().parent
V1 = (Path(__file__).resolve().parents[3] / "analysis/predictor_supervision/bbbc047/support/run_strict_selected_repeat_m2.py")
spec = importlib.util.spec_from_file_location("strict_selected_v1", V1)
if spec is None or spec.loader is None:
    raise RuntimeError(V1)
RUN = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = RUN
spec.loader.exec_module(RUN)


def strict_inputs(model_h5: Path, aggregate_npz: Path, split_lock: Path, splits: Iterable[str]) -> dict[str, Any]:
    """Load only immutable inputs; never read treatment target_CP or aggregate NPZ."""
    del aggregate_npz
    lock = json.loads(split_lock.read_text(encoding="utf-8"))
    requested = tuple(splits)
    with h5py.File(model_h5, "r") as handle:
        names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in handle["canonical_smiles"][:]]
        index = {name: i for i, name in enumerate(names)}
        if len(index) != len(names):
            raise RuntimeError("duplicate canonical_smiles")
        output: dict[str, Any] = {}
        for split in requested:
            selected = [str(x) for x in lock[f"{split}_smiles"]]
            rows = np.asarray([index[x] for x in selected], dtype=np.int64)
            order = np.argsort(rows); inverse = np.argsort(order)
            control = handle["control_CP"][rows[order]].astype(np.float32)[inverse]
            fingerprint = RUN.BASE.LEGACY.VIRTUAL.fingerprints(selected)
            output[split] = RUN.BASE.LEGACY.VIRTUAL.VirtualSplit(selected, control, np.zeros((len(selected), RUN.CP_DIM), dtype=np.float32), fingerprint)
    return output


RUN.VERSION = "BBBC047-strict-selected-repeat-1R-CFRA-M2-v2-2026-09-16"
RUN.HERE = HERE
RUN.BASE.load_virtual_splits = strict_inputs


if __name__ == "__main__":
    RUN.main()
