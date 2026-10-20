# Copyright (c) 2026 Awomo-WAM Team. Apache-2.0.
"""Load Awomo-0.5I checkpoint parameters."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import torch


def validate_checkpoint_payload(payload: dict) -> None:
    """Validate the public tensor-file schema, independent of artifact kind."""
    if not isinstance(payload, dict) or set(payload) != {"dit", "dit_config", "qwen_lora"}:
        raise ValueError("model.pt must contain exactly dit, dit_config and qwen_lora")
    if not all(isinstance(payload[name], dict) for name in payload):
        raise ValueError("Checkpoint model, architecture and adapter entries must be dictionaries")
    if not payload["dit"] or not payload["dit_config"] or not payload["qwen_lora"]:
        raise ValueError("Checkpoint entries must not be empty")


def canonical_lora(name: str) -> str:
    match = re.search(r"language_model\.(layers\.\d+\..+)\.lora_([AB])(?:\.default)?\.weight$", name)
    if not match:
        raise ValueError(f"Unsupported VLM adapter key: {name}")
    return match[1] + ".lora_" + match[2]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checkpoint_config(directory: str | Path, *, verify_checksums: bool = True,
                           allow_base_only: bool = False) -> dict:
    directory = Path(directory)
    base_only = allow_base_only and not (directory / "awomo_config.json").exists()
    required = {"model.pt"} if base_only else {"model.pt", "awomo_config.json", "dataset_stats.json"}
    checksums = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or name != Path(name).name or name in checksums:
            raise ValueError("Invalid checkpoint checksum manifest")
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or indirect checkpoint file: {name}")
        if verify_checksums and file_sha256(path) != digest:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
        checksums[name] = digest
    if not required.issubset(checksums):
        raise ValueError("Incomplete prepared checkpoint")
    if base_only:
        if set(checksums) != {"model.pt"}:
            raise ValueError("Base SHA256SUMS must contain exactly model.pt")
        # Only SFT initialization opts into this contract. Its encoder configuration
        # supplies prefix_v2; inference still requires explicit preprocessing metadata.
        return {"artifact_kind": "base", "vl_text_layout": "prefix_v2"}
    config = json.loads((directory / "awomo_config.json").read_text())
    if config.get("model_type") != "awomo-0.5i-robodojo-sft" or config.get("format_version") != 1:
        raise ValueError("Unsupported prepared checkpoint format")
    if config.get("artifact_kind") not in {"base", "sft"} or config.get("parameter_dtype") != "float32":
        raise ValueError("Expected prepared FP32 base or SFT weights")
    return config


def load_prepared_checkpoint(model, vl_encoder, directory: str | Path, *, verify_checksums: bool = True,
                             allow_base_only: bool = False) -> dict:
    """Strictly load stored FP32 parameters; optimizer state is not restored."""
    directory = Path(directory)
    config = read_checkpoint_config(directory, verify_checksums=verify_checksums, allow_base_only=allow_base_only)
    payload = torch.load(directory / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    validate_checkpoint_payload(payload)
    if config["vl_text_layout"] != vl_encoder.text_layout:
        raise ValueError("VLM text-layout mismatch")
    saved = payload["dit"]
    current = model.state_dict()
    if set(saved) != set(current):
        raise ValueError("Prepared model keys do not match the constructed model")
    for name, tensor in saved.items():
        if tensor.dtype != torch.float32 or tensor.shape != current[name].shape or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid prepared model tensor: {name}")
        if current[name].dtype != torch.float32:
            raise ValueError(f"Construct FP32 master parameters before loading: {name}")
    model.load_state_dict(saved, strict=True)
    parameters = {name: p for name, p in vl_encoder.named_parameters() if "lora_" in name}
    mapped = {canonical_lora(name): name for name in parameters}
    adapters = payload["qwen_lora"]
    if len(mapped) != len(parameters) or set(mapped) != set(adapters):
        raise ValueError("Prepared VLM adapter keys do not match the constructed encoder")
    with torch.no_grad():
        for canonical, name in mapped.items():
            value, parameter = adapters[canonical], parameters[name]
            if value.dtype != parameter.dtype or value.dtype != torch.float32 or value.shape != parameter.shape:
                raise ValueError(f"Invalid prepared adapter shape/dtype: {canonical}")
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite prepared adapter: {canonical}")
            parameter.copy_(value.to(parameter.device))
    return {"path": str(directory), "model_keys": len(saved), "lora_ema_keys": len(adapters),
            "artifact_kind": config["artifact_kind"], "training_state_reset": True}
