#!/usr/bin/env python3
"""Shared frozen utilities for the BBBC047 tabular-backbone stress test."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "BBBC047-modern-tabular-backbone-stress-test-v1-2026-09-20"
CP_DIM = 775
SEEDS = (3407, 42, 2025)
BUDGETS = (1, 2, 3)
MAX_EPOCHS = 80
EFFECTIVE_BATCH_SIZE = 256
EXPECTED_ROLE_SHA256 = "d72229c2cdb060306dffe7f90d8cde9c66f402a91fc1c5250781bf1e5467ace9"
RTDL_COMMIT = "e3ed46cac38568785289d8fa16b8cfa585bde27e"
TABM_COMMIT = "28e47ae301c92ec37787dde1ce923a0793f405b4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_int(label: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{VERSION}|{label}".encode("utf-8")).digest()[:8], "big"
    )


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def torch_modules() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required") from exc
    return torch, nn


def load_official_modules(vendor_root: Path, vendor_deps: Path) -> tuple[Any, Any]:
    for path in (vendor_root, vendor_deps):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    rtdl_path = vendor_root / "rtdl_revisiting_models.py"
    tabm_path = vendor_root / "tabm_reference.py"
    if not rtdl_path.is_file() or not tabm_path.is_file():
        raise FileNotFoundError("pinned official source file is absent")
    return load_module("arch_rtdl", rtdl_path), load_module("arch_tabm", tabm_path)


def set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def inner_train_mask(compounds: np.ndarray) -> np.ndarray:
    """Fixed compound-disjoint 80/20 split independent of profile values."""
    mask = np.asarray(
        [stable_int(f"inner-epoch-selection|{compound}") % 5 != 0 for compound in compounds],
        dtype=bool,
    )
    if not mask.any() or mask.all():
        raise RuntimeError("empty train-internal compound split")
    return mask


def model_specs() -> dict[str, dict[str, Any]]:
    return {
        "MLP-control": {
            "source": "rtdl-revisiting-models",
            "constructor": {"n_blocks": 3, "d_block": 768, "dropout": 0.10},
            "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-5},
            "micro_batch_size": 256,
        },
        "ResNet": {
            "source": "rtdl-revisiting-models",
            "constructor": {
                "n_blocks": 3,
                "d_block": 384,
                "d_hidden_multiplier": 2.0,
                "dropout1": 0.10,
                "dropout2": 0.0,
            },
            "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-5},
            "micro_batch_size": 256,
        },
        "FT-Transformer": {
            "source": "rtdl-revisiting-models",
            "constructor": {"default_kwargs": "get_default_kwargs(n_blocks=4)"},
            "optimizer": {
                "name": "official FTTransformer parameter groups + AdamW",
                "lr": 1e-4,
                "weight_decay": 1e-5,
            },
            "micro_batch_size": 16,
        },
        "TabM": {
            "source": "tabm",
            "constructor": {
                "backbone": {"type": "MLP", "n_blocks": 3, "d_block": 512, "dropout": 0.10},
                "arch_type": "tabm",
                "k": 4,
            },
            "optimizer": {"name": "official TabM parameter groups + AdamW", "lr": 3e-4, "weight_decay": 1e-5},
            "micro_batch_size": 256,
        },
    }


def build_model(name: str, rtdl: Any, tabm: Any, device: Any) -> Any:
    if name == "MLP-control":
        model = rtdl.MLP(d_in=CP_DIM, d_out=CP_DIM, n_blocks=3, d_block=768, dropout=0.10)
    elif name == "ResNet":
        model = rtdl.ResNet(
            d_in=CP_DIM,
            d_out=CP_DIM,
            n_blocks=3,
            d_block=384,
            d_hidden=None,
            d_hidden_multiplier=2.0,
            dropout1=0.10,
            dropout2=0.0,
        )
    elif name == "FT-Transformer":
        model = rtdl.FTTransformer(
            n_cont_features=CP_DIM,
            cat_cardinalities=[],
            d_out=CP_DIM,
            **rtdl.FTTransformer.get_default_kwargs(n_blocks=4),
        )
    elif name == "TabM":
        model = tabm.Model(
            n_num_features=CP_DIM,
            cat_cardinalities=[],
            n_classes=CP_DIM,
            backbone={"type": "MLP", "n_blocks": 3, "d_block": 512, "dropout": 0.10},
            bins=None,
            arch_type="tabm",
            k=4,
        )
    else:
        raise KeyError(name)
    return model.to(device)


def forward_prediction(name: str, model: Any, x: Any) -> Any:
    if name == "FT-Transformer":
        return model(x, None)
    if name == "TabM":
        return model(x, None).mean(dim=1)
    return model(x)


def forward_training_output(name: str, model: Any, x: Any) -> Any:
    """Return the native head output used by the supervised training loss."""
    if name in {"FT-Transformer", "TabM"}:
        return model(x, None)
    return model(x)


def supervised_mse(name: str, output: Any, target: Any) -> Any:
    """Official-style per-member MSE for TabM; ordinary MSE for other arms."""
    if name == "TabM":
        return (output - target.unsqueeze(1)).square().mean()
    return (output - target).square().mean()


def make_optimizer(name: str, model: Any, tabm: Any, torch: Any) -> Any:
    if name == "FT-Transformer":
        return torch.optim.AdamW(model.make_parameter_groups(), lr=1e-4, weight_decay=1e-5)
    if name == "TabM":
        return torch.optim.AdamW(tabm.make_parameter_groups(model), lr=3e-4, weight_decay=1e-5)
    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)


def model_audit(name: str, rtdl: Any, tabm: Any, torch: Any, device: Any) -> dict[str, Any]:
    set_seed(torch, SEEDS[0])
    model = build_model(name, rtdl, tabm, device)
    parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    with torch.no_grad():
        output = forward_prediction(name, model, torch.zeros((2, CP_DIM), device=device))
    if tuple(output.shape) != (2, CP_DIM):
        raise RuntimeError(f"{name} output shape is not 775D: {tuple(output.shape)}")
    if not 1_000_000 <= parameters <= 3_000_000:
        raise RuntimeError(f"{name} parameter count is outside frozen band: {parameters}")
    return {
        "parameters": parameters,
        "prediction_shape": list(output.shape),
        "capacity_gate": "PASS",
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "micro_batch_size": model_specs()[name]["micro_batch_size"],
        "gradient_accumulation_steps": EFFECTIVE_BATCH_SIZE // model_specs()[name]["micro_batch_size"],
        "specification": model_specs()[name],
    }


def source_audit(vendor_root: Path, protocol: Path) -> dict[str, Any]:
    rtdl_path = vendor_root / "rtdl_revisiting_models.py"
    tabm_path = vendor_root / "tabm_reference.py"
    return {
        "protocol_sha256": sha256_file(protocol),
        "official_sources": {
            "rtdl_revisiting_models": {"commit": RTDL_COMMIT, "sha256": sha256_file(rtdl_path)},
            "tabm_reference": {"commit": TABM_COMMIT, "sha256": sha256_file(tabm_path)},
        },
        "rtdl_num_embeddings": "0.0.11",
    }


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
