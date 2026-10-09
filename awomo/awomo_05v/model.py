# Copyright (c) 2026 Awomo-WAM Team. Licensed under the Apache License, Version 2.0.
"""Awomo-0.5V model: load a released checkpoint; encode text and video; warm up the KV cache; one DiT evaluation.

Sampling (schedulers, step selection, guidance, inactive action slots) belongs to each benchmark adapter.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from awomo.awomo_05v.t5 import WanTextEncoder
from awomo.awomo_05v.wan_dit import CausalWanModel
from awomo.awomo_05v.wan_vae import WanVideoVAE38


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(directory: Path) -> None:
    """Check every file listed in SHA256SUMS."""
    failures = []
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            expected, name = line.split(maxsplit=1)
            path = directory / name.strip().lstrip("*")
            if not path.is_file() or _sha256(path) != expected:
                failures.append(name)
    if failures:
        raise ValueError(f"checkpoint verification failed: {failures}")


def resolve_base_model(spec: dict[str, Any], local_path: str | None, env_var: str) -> Path:
    """Local path > environment variable > Hugging Face download (only the files listed in the spec)."""
    local = local_path or os.environ.get(env_var)
    if local:
        return Path(local)
    from huggingface_hub import snapshot_download

    files = spec["files"]
    patterns = [files["vae"], files["text_encoder"], f"{files['tokenizer']}/*"]
    return Path(snapshot_download(spec["repo_id"], revision=spec["revision"], allow_patterns=patterns))


class Awomo05v:
    def __init__(self, checkpoint: str | Path, device: str = "cuda", wan_path: str | None = None) -> None:
        checkpoint = Path(checkpoint)
        verify_checkpoint(checkpoint)
        config = json.loads((checkpoint / "awomo_config.json").read_text())
        if config.get("model_type") != "awomo-0.5v":
            raise ValueError(f"{checkpoint} is not an Awomo-0.5V checkpoint")
        self.config = config
        self.device = torch.device(device)
        self.action_dim = int(config["action_dim"])
        self.state_dim = int(config["state_dim"])
        self.video_size = tuple(int(v) for v in config["video_size"])
        self.text_len = int(config["text_len"])
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        spec = config["base_models"]["wan2_2_ti2v_5b"]
        wan_dir = resolve_base_model(spec, wan_path, "AWOMO_WAN22_TI2V_5B")
        files = spec["files"]
        self.tokenizer = AutoTokenizer.from_pretrained(str(wan_dir / files["tokenizer"]))

        # model.pt holds the DiT constructor arguments ("dit_config") and its state dict ("dit").
        weights = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True, mmap=True)
        dit = CausalWanModel(**weights["dit_config"])
        dit.load_state_dict(weights.pop("dit"), strict=True)
        self.dit = dit.to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
        self.num_frame_per_block = int(dit.num_frame_per_block)

        text_encoder = WanTextEncoder()
        text_encoder.load_state_dict(torch.load(wan_dir / files["text_encoder"], map_location="cpu", weights_only=True))
        self.text_encoder = text_encoder.eval().requires_grad_(False).to(device=self.device, dtype=torch.bfloat16)

        vae = WanVideoVAE38()
        vae.model.load_state_dict(torch.load(wan_dir / files["vae"], map_location="cpu", weights_only=True))
        self.vae = vae.eval().requires_grad_(False).to(device=self.device, dtype=torch.bfloat16)

    @torch.no_grad()
    def encode_text(self, prompt: str) -> torch.Tensor:
        """Prompt -> [1, text_len, 4096] bf16 embeddings, zero after the last token."""
        tokens = self.tokenizer([prompt], padding="max_length", truncation=True, max_length=self.text_len,
                                return_tensors="pt")
        ids = tokens["input_ids"].to(self.device)
        mask = tokens["attention_mask"].to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        embeddings = self.text_encoder(ids, mask).clone().to(dtype=torch.bfloat16)
        for i, length in enumerate(seq_lens):
            embeddings[i, length:] = 0
        return embeddings

    @torch.no_grad()
    def encode_video(self, video: np.ndarray) -> torch.Tensor:
        """[T, H, W, 3] float32 in [0, 1] (T = 4k + 1) -> latents [1, 48, k + 1, H / 16, W / 16]."""
        videos = torch.from_numpy(np.ascontiguousarray(video))[None].to(self.device).float()
        videos = (videos.permute(0, 4, 1, 2, 3) * 2.0 - 1.0).to(dtype=torch.bfloat16)
        return self.vae.encode(videos)

    def tokens_per_frame(self, latents: torch.Tensor) -> int:
        return (int(latents.shape[-2]) // 2) * (int(latents.shape[-1]) // 2)

    @torch.no_grad()
    def warm_up(self, latents: torch.Tensor, prompt_embs: torch.Tensor) -> list[torch.Tensor]:
        """Build the KV cache of clean observation latents (timestep 0) from an empty cache."""
        dit = self.dit
        head_dim = dit.dim // dit.num_heads
        kv_cache = [torch.zeros(2, 1, 0, dit.num_heads, head_dim, dtype=torch.bfloat16, device=self.device)
                    for _ in range(dit.num_layers)]
        batch, _, frames = latents.shape[:3]
        t_zero = torch.zeros(batch, frames, device=self.device, dtype=torch.bfloat16)
        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            _, _, updated = dit(x=latents, timestep=t_zero, clip_feature=None, y=None, context=prompt_embs,
                                seq_len=frames * self.tokens_per_frame(latents), state=None, action=None,
                                timestep_action=None, kv_cache=kv_cache, crossattn_cache=None, current_start_frame=0)
        return [cache.clone() for cache in updated]

    @torch.no_grad()
    def velocity(self, x_video: torch.Tensor, x_action: torch.Tensor, video_t: torch.Tensor, action_t: torch.Tensor,
                 state: torch.Tensor, prompt_embs: torch.Tensor, kv_cache: list[torch.Tensor],
                 start_frame: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One DiT evaluation of the next block (video latents + action chunk) after `start_frame` cached frames."""
        frames = self.num_frame_per_block
        timestep = video_t.to(self.device).view(1, 1).expand(1, frames)
        timestep_action = action_t.to(self.device).view(1, 1).expand(1, int(x_action.shape[1]))
        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            video_pred, action_pred, _ = self.dit(
                context=prompt_embs, kv_cache=kv_cache, x=x_video, timestep=timestep, clip_feature=None, y=None,
                seq_len=frames * self.tokens_per_frame(x_video), state=state, action=x_action,
                timestep_action=timestep_action, crossattn_cache=None, current_start_frame=start_frame)
        return video_pred, action_pred
