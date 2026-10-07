#!/usr/bin/env python3
"""Frozen utilities for the BBBC047 native-size backbone sensitivity."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "BBBC047-native-size-backbone-sensitivity-v1-2026-09-21"
COMPACT_SELECTION_VERSION = "BBBC047-modern-tabular-backbone-stress-test-v1-2026-09-20"
CP_DIM = 775
SEEDS = (3407, 42, 2025)
BUDGETS = (1, 2, 3)
MAX_EPOCHS = 80
EFFECTIVE_BATCH_SIZE = 256
EXPECTED_ROLE_SHA256 = "d72229c2cdb060306dffe7f90d8cde9c66f402a91fc1c5250781bf1e5467ace9"
RTDL_COMMIT = "e3ed46cac38568785289d8fa16b8cfa585bde27e"
TABM_COMMIT = "28e47ae301c92ec37787dde1ce923a0793f405b4"
ARMS = ("ResNet-wide512", "FT-Transformer-5block-default", "TabM-native-k32")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_int(label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{VERSION}|{label}".encode()).digest()[:8], "big")


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def torch_modules() -> tuple[Any, Any]:
    import torch
    from torch import nn
    return torch, nn


def load_official_modules(vendor_root: Path, vendor_deps: Path) -> tuple[Any, Any]:
    for path in (vendor_root, vendor_deps):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return (
        load_module("native_rtdl", vendor_root / "rtdl_revisiting_models.py"),
        load_module("native_tabm", vendor_root / "tabm_reference.py"),
    )


def set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def inner_train_mask(compounds: np.ndarray) -> np.ndarray:
    """Reuse the compact track's fixed compound-disjoint 80/20 epoch split."""
    result = np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(
                    f"{COMPACT_SELECTION_VERSION}|inner-epoch-selection|{compound}".encode("utf-8")
                ).digest()[:8],
                "big",
            )
            % 5
            != 0
            for compound in compounds
        ],
        dtype=bool,
    )
    if not result.any() or result.all():
        raise RuntimeError("empty native internal compound split")
    return result


def model_specs() -> dict[str, dict[str, Any]]:
    return {
        "ResNet-wide512": {
            "rationale": "Expanded residual-width sensitivity; RTDL has no single native default.",
            "constructor": {"n_blocks": 3, "d_block": 512, "d_hidden_multiplier": 2.0, "dropout1": 0.10, "dropout2": 0.0},
            "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-5},
            "micro_batch_size": 256,
        },
        "FT-Transformer-5block-default": {
            "rationale": "Official RTDL default family at n_blocks=5; all other constructor values come from get_default_kwargs.",
            "constructor": {"default_kwargs": "get_default_kwargs(n_blocks=5)"},
            "optimizer": {"name": "official FTTransformer parameter groups + AdamW", "lr": 1e-4, "weight_decay": 1e-5},
            "micro_batch_size": 16,
        },
        "TabM-native-k32": {
            "rationale": "Official TabM documented default depth/width/k and default AdamW; no numerical embeddings in this base TabM arm.",
            "constructor": {"backbone": {"type": "MLP", "n_blocks": 3, "d_block": 512, "dropout": 0.10}, "arch_type": "tabm", "k": 32},
            "optimizer": {"name": "official TabM default AdamW", "lr": 2e-3, "weight_decay": 3e-4},
            "micro_batch_size": 64,
        },
    }


def build_model(name: str, rtdl: Any, tabm: Any, device: Any) -> Any:
    if name == "ResNet-wide512":
        model = rtdl.ResNet(d_in=CP_DIM, d_out=CP_DIM, n_blocks=3, d_block=512, d_hidden=None, d_hidden_multiplier=2.0, dropout1=0.10, dropout2=0.0)
    elif name == "FT-Transformer-5block-default":
        model = rtdl.FTTransformer(n_cont_features=CP_DIM, cat_cardinalities=[], d_out=CP_DIM, **rtdl.FTTransformer.get_default_kwargs(n_blocks=5))
    elif name == "TabM-native-k32":
        model = tabm.Model(n_num_features=CP_DIM, cat_cardinalities=[], n_classes=CP_DIM, backbone={"type": "MLP", "n_blocks": 3, "d_block": 512, "dropout": 0.10}, bins=None, arch_type="tabm", k=32)
    else:
        raise KeyError(name)
    return model.to(device)


def training_output(name: str, model: Any, x: Any) -> Any:
    if name in {"FT-Transformer-5block-default", "TabM-native-k32"}:
        return model(x, None)
    return model(x)


def prediction(name: str, model: Any, x: Any) -> Any:
    output = training_output(name, model, x)
    return output.mean(dim=1) if name == "TabM-native-k32" else output


def supervised_mse(name: str, output: Any, target: Any) -> Any:
    return (output - target.unsqueeze(1)).square().mean() if name == "TabM-native-k32" else (output - target).square().mean()


def make_optimizer(name: str, model: Any, tabm: Any, torch: Any) -> Any:
    if name == "FT-Transformer-5block-default":
        return torch.optim.AdamW(model.make_parameter_groups(), lr=1e-4, weight_decay=1e-5)
    if name == "TabM-native-k32":
        return torch.optim.AdamW(tabm.make_parameter_groups(model), lr=2e-3, weight_decay=3e-4)
    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)


def model_audit(name: str, rtdl: Any, tabm: Any, torch: Any, device: Any) -> dict[str, Any]:
    set_seed(torch, SEEDS[0])
    model = build_model(name, rtdl, tabm, device)
    parameters = int(sum(value.numel() for value in model.parameters()))
    with torch.no_grad():
        shape = tuple(prediction(name, model, torch.zeros((2, CP_DIM), device=device)).shape)
    if shape != (2, CP_DIM) or not 3_000_000 <= parameters <= 15_000_000:
        raise RuntimeError(f"native size/shape gate failed for {name}: {parameters}, {shape}")
    spec = model_specs()[name]
    return {"parameters": parameters, "prediction_shape": list(shape), "capacity_gate": "PASS", "effective_batch_size": EFFECTIVE_BATCH_SIZE, "micro_batch_size": spec["micro_batch_size"], "gradient_accumulation_steps": EFFECTIVE_BATCH_SIZE // spec["micro_batch_size"], "specification": spec}


def source_audit(vendor_root: Path, protocol: Path) -> dict[str, Any]:
    return {
        "protocol_sha256": sha256_file(protocol),
        "official_sources": {
            "rtdl_revisiting_models": {"commit": RTDL_COMMIT, "sha256": sha256_file(vendor_root / "rtdl_revisiting_models.py")},
            "tabm_reference": {"commit": TABM_COMMIT, "sha256": sha256_file(vendor_root / "tabm_reference.py")},
        },
        "rtdl_num_embeddings": "0.0.11",
    }


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
