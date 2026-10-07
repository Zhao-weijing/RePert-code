#!/usr/bin/env python3
"""Create immutable train-/valid-only cpg CP artifacts from the prepared source.

The source preparation artifact stores all split rows in one compressed NPZ.
This one-time conversion emits two physical artifacts containing only the
requested split, so downstream validation code cannot materialise test rows
through the shared container.  No model is fit and no outcome is evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_SOURCE = Path("/path/to/data/AIDD/MVCPert_5_27/runs/cpg0004_lincs_cp_prepared_20260830/artifact/cp_plate_rows.npz")
DEFAULT_OUTDIR = Path("/path/to/data/AIDD/MVCPert_5_27/runs/cpg0004_lincs_cp_train_valid_only_20260831")
VERSION = "cpg-train-valid-physical-split-v1-2026-08-31"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.outdir.exists():
        raise FileExistsError(f"refusing to overwrite split-only input directory: {args.outdir}")
    source = np.load(args.source, allow_pickle=False)
    fields = ("compound_id", "dose", "plate", "delta", "split")
    missing = [field for field in fields if field not in source.files]
    if missing:
        raise RuntimeError(f"source artifact misses fields: {missing}")
    split = np.asarray(source["split"], dtype=str)
    args.outdir.mkdir(parents=True)
    summary: dict[str, object] = {
        "version": VERSION,
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "conversion_scope": "physical train/valid-only exports; no model fit or outcome evaluation",
        "exports": {},
    }
    for label in ("train", "valid"):
        mask = split == label
        if not np.any(mask):
            raise RuntimeError(f"source has no rows for split={label}")
        path = args.outdir / f"{label}_cp_plate_rows.npz"
        np.savez_compressed(
            path,
            compound_id=np.asarray(source["compound_id"])[mask],
            dose=np.asarray(source["dose"])[mask],
            plate=np.asarray(source["plate"])[mask],
            delta=np.asarray(source["delta"], dtype=np.float32)[mask],
            split=np.asarray(source["split"], dtype=str)[mask],
        )
        summary["exports"][label] = {
            "path": str(path),
            "sha256": sha256(path),
            "n_rows": int(mask.sum()),
            "n_compounds": int(len(set(np.asarray(source["compound_id"], dtype=str)[mask].tolist()))),
            "only_split": bool(np.all(np.asarray(source["split"], dtype=str)[mask] == label)),
        }
    (args.outdir / "SPLIT_EXPORT_AUDIT.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
