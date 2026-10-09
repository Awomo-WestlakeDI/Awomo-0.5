# Copyright (c) 2026 Awomo-WAM Team. Licensed under the Apache License, Version 2.0.
"""Awomo-0.5I model: load a released checkpoint, encode one observation, evaluate the DiT once."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel
from torch import nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

CONDITION_LAYERS = (9, 18, 27)
SUMMARY_LAYERS = (18, 27, 36)
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
POOL = (4, 4)


# ----------------------------------------------------------------------------------------------- checkpoint


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
    """Local path > environment variable > Hugging Face download."""
    local = local_path or os.environ.get(env_var)
    if local:
        return Path(local)
    from huggingface_hub import snapshot_download

    subfolder = spec.get("subfolder")
    root = Path(snapshot_download(spec["repo_id"], revision=spec.get("revision"),
                                  allow_patterns=[f"{subfolder}/*"] if subfolder else None))
    return root / subfolder if subfolder else root


# ----------------------------------------------------------------------------------------------- networks


class ActionPatchModel(nn.Module):
    def __init__(self, transformer: Flux2Transformer2DModel, action_dim: int, state_dim: int, action_scale: float):
        super().__init__()
        self.transformer = transformer
        self.proprio_encoder = nn.Linear(state_dim, transformer.config.joint_attention_dim,
                                         dtype=next(transformer.parameters()).dtype)
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.repeats = 128 // action_dim

    def encode_action(self, action):
        repeated = action.repeat_interleave(self.repeats, dim=-1) * self.action_scale
        return F.pad(repeated, (0, 128 - repeated.shape[-1]))

    def decode_action(self, tokens):
        used = tokens[..., : self.action_dim * self.repeats]
        return used.unflatten(-1, (self.action_dim, self.repeats)).mean(-1) / self.action_scale

    def pack_state(self, context, text_mask, state):
        batch, length, dim = context.shape
        lengths = text_mask.sum(-1)
        positions = torch.arange(length, device=context.device)
        packed = context.new_zeros(batch, length + 1, dim)
        destinations = positions[None] + (~text_mask).long()
        packed = packed.scatter(1, destinations[..., None].expand(batch, length, dim), context)
        indices = lengths[:, None, None].expand(batch, 1, dim)
        packed = packed.scatter(1, indices, self.proprio_encoder(state)[:, None])
        mask = torch.arange(length + 1, device=context.device)[None] <= lengths[:, None]
        return packed, mask

    @staticmethod
    def attention_mask(text_mask, condition_length, noisy_length, condition_image_mask):
        batch, text_length = text_mask.shape
        prefix = text_length + condition_length
        total = prefix + noisy_length
        mask = torch.ones(batch, 1, total, total, dtype=torch.bool, device=text_mask.device)
        mask[:, :, :prefix, prefix:] = False
        mask[:, :, :, :text_length] &= text_mask[:, None, None, :]
        mask[:, :, :, text_length:prefix] &= condition_image_mask[:, None, None, :]
        return mask

    @staticmethod
    def image_ids(height, width, time, device):
        ids = torch.zeros(height, width, 4, device=device)
        ids[..., 0] = time
        ids[..., 1] = torch.arange(height, device=device)[:, None]
        ids[..., 2] = torch.arange(width, device=device)[None, :]
        return ids.reshape(-1, 4)

    def forward(self, reference, noisy_image, noisy_action, context, text_mask, state, timestep, image_grid,
                history, history_ids, history_valid):
        batch, image_length, _ = reference.shape
        context, text_mask = self.pack_state(context, text_mask, state)
        horizon = noisy_action.shape[1]
        image_ids = torch.cat([self.image_ids(*image_grid, 10.0, reference.device),
                               self.image_ids(*image_grid, 0.0, reference.device),
                               torch.zeros(horizon, 4, device=reference.device)])
        image_ids[-horizon:, 0] = 20
        image_ids[-horizon:, 3] = torch.arange(horizon, device=reference.device)
        text_ids = torch.zeros(context.shape[1], 4, device=reference.device)
        text_ids[:, 3] = torch.arange(context.shape[1], device=reference.device)
        clean = torch.cat([history, reference], dim=1)
        image_ids = torch.cat([history_ids.to(image_ids), image_ids])
        reference_valid = torch.ones(batch, image_length, dtype=torch.bool, device=reference.device)
        condition_image_mask = torch.cat([history_valid.to(torch.bool), reference_valid], dim=1)
        clean_length = clean.shape[1]
        mask = self.attention_mask(text_mask, clean_length, image_length + horizon, condition_image_mask)
        prediction = self.transformer(
            hidden_states=torch.cat([clean, noisy_image, self.encode_action(noisy_action)], dim=1),
            encoder_hidden_states=context, timestep=timestep / 1000, img_ids=image_ids, txt_ids=text_ids,
            joint_attention_kwargs={"attention_mask": mask}, return_dict=False)[0]
        end = clean_length + image_length
        return prediction[:, clean_length:end], self.decode_action(prediction[:, end:])


class VideoVAE(nn.Module):
    def __init__(self, vae: AutoencoderKLFlux2):
        super().__init__()
        self.vae = vae
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def encode(self, video):
        """[B, 3, T, H, W] in [-1, 1] -> [B, 128, T, H/16, W/16]."""
        batch, _, frames, height, width = video.shape
        images = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, 3, height, width)
        images = images.to(device=self.vae.device, dtype=self.vae.dtype)
        latent = Flux2KleinPipeline._patchify_latents(self.vae.encode(images).latent_dist.mode())
        latent = F.batch_norm(latent, self.vae.bn.running_mean, self.vae.bn.running_var, training=False,
                              eps=self.vae.config.batch_norm_eps)
        return latent.unflatten(0, (batch, frames)).permute(0, 2, 1, 3, 4)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scaling = float(alpha) / float(rank)
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features, dtype=torch.float32, device=base.weight.device),
                                   requires_grad=False)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=base.weight.device),
                                   requires_grad=False)

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        update = F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return self.base(x) + update * self.scaling


class VisionLanguageEncoder(nn.Module):
    def __init__(self, qwen, lora_rank: int, lora_alpha: float):
        super().__init__()
        self.qwen = qwen.requires_grad_(False)
        self.lora: dict[str, LoRALinear] = {}
        for index, layer in enumerate(qwen.model.language_model.layers):
            for parent_name, parent in list(layer.named_modules()):
                for child_name, child in list(parent.named_children()):
                    if child_name in LORA_TARGETS and isinstance(child, nn.Linear):
                        wrapped = LoRALinear(child, lora_rank, lora_alpha)
                        setattr(parent, child_name, wrapped)
                        self.lora[f"layers.{index}.{parent_name}.{child_name}"] = wrapped

    def load_lora(self, state: dict[str, torch.Tensor]) -> None:
        expected = {}
        for name, module in sorted(self.lora.items()):
            expected[f"{name}.lora_A"] = module.lora_A
            expected[f"{name}.lora_B"] = module.lora_B
        if set(state) != set(expected):
            raise ValueError("LoRA weights do not match the model")
        with torch.no_grad():
            for name, param in expected.items():
                param.copy_(state[name].to(param.device, param.dtype))

    @torch.no_grad()
    def forward(self, vb):
        outputs = self.qwen.model(input_ids=vb["input_ids"], attention_mask=vb["attention_mask"],
                                  pixel_values=vb["pixel_values"], image_grid_thw=vb["image_grid_thw"],
                                  mm_token_type_ids=vb["mm_token_type_ids"], output_hidden_states=True,
                                  use_cache=False)
        hidden = outputs.hidden_states
        condition = torch.cat([hidden[layer] for layer in CONDITION_LAYERS], dim=-1)
        summary = torch.cat([hidden[layer] for layer in SUMMARY_LAYERS], dim=-1)
        condition_pos, summary_pos = vb["condition_pos"], vb["summary_pos"]
        batch, dim = condition_pos.shape[0], condition.shape[-1]
        text_valid = condition_pos >= 0
        text = torch.gather(condition, 1, condition_pos.clamp(min=0)[..., None].expand(-1, -1, dim))
        rows = torch.arange(batch, device=summary.device)
        summary_tokens = summary[rows, summary_pos.to(summary.device)]
        lengths = text_valid.sum(dim=1)
        width = int(condition_pos.shape[1]) + 1
        context = condition.new_zeros(batch, width, dim)
        context[:, : condition_pos.shape[1]] = text * text_valid[..., None].to(text.dtype)
        context[rows, lengths] = summary_tokens.to(context.dtype)
        context_mask = torch.arange(width, device=context.device)[None] <= lengths[:, None]
        return context, context_mask


# ----------------------------------------------------------------------------------------------- model


def _history_ids(slots: int, device) -> torch.Tensor:
    ids = torch.zeros(slots, POOL[0], POOL[1], 4, device=device)
    ids[..., 0] = -(slots - torch.arange(slots, device=device, dtype=torch.float32))[:, None, None]
    ids[..., 1] = torch.arange(POOL[0], device=device, dtype=torch.float32)[None, :, None]
    ids[..., 2] = torch.arange(POOL[1], device=device, dtype=torch.float32)[None, None, :]
    return ids.reshape(slots * POOL[0] * POOL[1], 4)


class Awomo05i:
    def __init__(self, checkpoint: str | Path, device: str = "cuda", qwen_path: str | None = None,
                 vae_path: str | None = None) -> None:
        checkpoint = Path(checkpoint)
        verify_checkpoint(checkpoint)
        config = json.loads((checkpoint / "awomo_config.json").read_text())
        if config.get("model_type") != "awomo-0.5i":
            raise ValueError(f"{checkpoint} is not an Awomo-0.5I checkpoint")
        self.config = config
        self.device = torch.device(device)
        self.action_dim = int(config["action_dim"])
        self.action_horizon = int(config["action_horizon"])
        self.history_slots = int(config["history"]["slots"])
        self.history_period = int(config["history"]["period_steps"])
        self.video_size = tuple(int(v) for v in config["video_size"])
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        qwen_dir = resolve_base_model(config["base_models"]["qwen3_vl"], qwen_path, "AWOMO_QWEN3_VL")
        vae_dir = resolve_base_model(config["base_models"]["flux2_vae"], vae_path, "AWOMO_FLUX2_VAE")
        self.processor = AutoProcessor.from_pretrained(str(qwen_dir))

        # model.pt holds the DiT config and two state dicts: "dit" and "qwen_lora".
        weights = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True, mmap=True)
        transformer = Flux2Transformer2DModel.from_config(weights["dit_config"]).to(torch.float32)
        dit = ActionPatchModel(transformer, self.action_dim, int(config["state_dim"]), float(config["action_scale"]))
        dit = dit.to(dtype=torch.bfloat16)
        dit.load_state_dict(weights.pop("dit"), strict=True)
        self.dit = dit.to(device=self.device).eval().requires_grad_(False)

        vae = AutoencoderKLFlux2.from_pretrained(str(vae_dir), torch_dtype=torch.bfloat16)
        self.vae = VideoVAE(vae).to(device=self.device, dtype=torch.bfloat16)

        qwen = Qwen3VLForConditionalGeneration.from_pretrained(str(qwen_dir), torch_dtype=torch.bfloat16)
        lora = config["vl_conditioner"]
        encoder = VisionLanguageEncoder(qwen, int(lora["lora_rank"]), float(lora["lora_alpha"]))
        encoder.load_lora(weights.pop("qwen_lora"))
        self.encoder = encoder.to(device=self.device).eval()

    @torch.no_grad()
    def encode(self, sample: dict[str, Any]) -> dict[str, Any]:
        device = self.device
        vb = {key: value.to(device, non_blocking=True) for key, value in sample["vl"].items()}
        context, context_mask = self.encoder(vb)
        videos = torch.from_numpy(sample["video"])[None].to(device=device, dtype=torch.bfloat16)
        videos = videos.permute(0, 4, 1, 2, 3) * 2.0 - 1.0
        latents = self.vae.encode(videos[:, :, [0, -1]])
        history, history_valid = self._encode_history(sample["history"], sample["history_valid"])
        return {
            "reference": latents[:, :, 0].flatten(2).transpose(1, 2),
            "target": latents[:, :, 1].flatten(2).transpose(1, 2),
            "image_grid": (int(latents.shape[-2]), int(latents.shape[-1])),
            "context": context.to(torch.bfloat16),
            "text_mask": context_mask,
            "history": history,
            "history_valid": history_valid,
            "history_ids": _history_ids(self.history_slots, device),
            "state": torch.from_numpy(sample["state"])[None].to(device, torch.bfloat16)[:, 0],
        }

    @torch.no_grad()
    def _encode_history(self, history: np.ndarray, valid: np.ndarray):
        slots, per_frame = self.history_slots, POOL[0] * POOL[1]
        history_t = torch.from_numpy(history)[None]
        slot_valid = torch.from_numpy(np.asarray(valid, dtype=bool))[None].to(self.device)
        pooled = torch.zeros(slots, per_frame, 128, device=self.device, dtype=torch.bfloat16)
        flat_valid = slot_valid.reshape(-1)
        if bool(flat_valid.any()):
            frames = history_t.reshape(slots, *history_t.shape[2:])[flat_valid.cpu()]
            video = frames.to(self.device, torch.bfloat16).permute(0, 3, 1, 2)[:, :, None] * 2.0 - 1.0
            latent = self.vae.encode(video)[:, :, 0]
            tokens = latent.flatten(2).transpose(1, 2)
            grid = (int(latent.shape[-2]), int(latent.shape[-1]))
            spatial = tokens.reshape(tokens.shape[0], grid[0], grid[1], 128).permute(0, 3, 1, 2)
            out = F.adaptive_avg_pool2d(spatial.float(), POOL).to(tokens.dtype)
            pooled[flat_valid] = out.permute(0, 2, 3, 1).reshape(-1, per_frame, 128).to(torch.bfloat16)
        token_valid = slot_valid.repeat_interleave(per_frame, dim=1)
        return pooled.reshape(1, slots * per_frame, 128), token_valid

    @torch.no_grad()
    def velocity(self, encoded: dict[str, Any], noisy_image: torch.Tensor, noisy_action: torch.Tensor,
                 timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One DiT evaluation: velocities of the target image latent and the action chunk at `timestep`."""
        return self.dit(
            encoded["reference"], noisy_image, noisy_action, encoded["context"], encoded["text_mask"],
            encoded["state"], timestep.to(dtype=encoded["target"].dtype).expand(1), encoded["image_grid"],
            encoded["history"], encoded["history_ids"], encoded["history_valid"])
