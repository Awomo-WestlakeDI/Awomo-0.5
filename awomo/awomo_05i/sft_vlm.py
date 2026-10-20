# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen Qwen3-VL instruction/image/history features for Awomo-0.5I SFT.

The DiT receives instruction hidden states plus a summary token from the prompt
prefix. Historical image features are pooled before entering the language tower.
"""

from __future__ import annotations

from logging import getLogger as get_logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from awomo.data.robodojo.prompt import TEXT_LAYOUT, history_offsets_frames

logger = get_logger(__name__)

# Language tower only: the ViT stays frozen so autograd keeps no vision activations.
DEFAULT_LORA_TARGET_MODULES = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


class VLPromptEncoder(nn.Module):
    """Qwen3-VL as a (history + image + text) feature extractor for the FLUX.2 text stream."""

    def __init__(
        self,
        vl_path,
        qwen_layers=(9, 18, 27),
        summary_layers=(18, 27, 36),
        max_prompt_len=640,
        device="cuda",
        torch_dtype=torch.bfloat16,
        lora=None,
        gradient_checkpointing=True,
        attn_implementation="sdpa",
        image_size=(448, 448),
        history_image_size=(448, 448),
        history_pool=(4, 4),
        history_num_slots=20,
        history_period_s=1.0,
        history_fps=25.0,
    ):
        super().__init__()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(vl_path)
        self.tok = self.processor.tokenizer
        self.tok.padding_side = "left"
        self.merge_size = int(getattr(self.processor.image_processor, "merge_size", 2))
        full = Qwen3VLForConditionalGeneration.from_pretrained(
            vl_path, torch_dtype=torch_dtype, attn_implementation=attn_implementation
        )
        full.requires_grad_(False)  # freeze BEFORE injecting LoRA

        lora = dict(lora or {})
        self.lora_enabled = bool(lora.pop("enabled", False))
        self.lora_lr = float(lora.pop("lr", 2e-5))
        if self.lora_enabled:
            from peft import LoraConfig, get_peft_model

            if gradient_checkpointing:
                # LoRA saves optimizer state, not activations: without this the language tower's
                # activations for the whole prompt are kept for the backward pass.
                full.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                full.enable_input_require_grads()
            cfg = LoraConfig(
                r=int(lora.pop("r", 64)),
                lora_alpha=int(lora.pop("alpha", 128)),
                lora_dropout=float(lora.pop("dropout", 0.05)),
                bias="none",
                target_modules=lora.pop("target_modules", DEFAULT_LORA_TARGET_MODULES),
            )
            if lora:
                raise ValueError(f"Unknown lora options: {sorted(lora)}")
            full = get_peft_model(full, cfg)
        self.vl_model = full.to(device)  # peft-wrapped when LoRA is on
        self.image_token_id = int(self.base_model.config.image_token_id)
        self.qwen_layers = tuple(int(layer) for layer in qwen_layers)
        self.summary_layers = tuple(int(layer) for layer in summary_layers)
        if len(self.summary_layers) != len(self.qwen_layers):
            raise ValueError(
                f"summary_layers={self.summary_layers} must have the same count as "
                f"qwen_layers={self.qwen_layers}: both must fill the DiT's txt_in width."
            )
        n_layers = int(self.vl.config.text_config.num_hidden_layers)
        for layer in self.qwen_layers + self.summary_layers:
            if not 0 <= layer <= n_layers:
                raise ValueError(
                    f"layer index {layer} outside [0, {n_layers}] (hidden_states has {n_layers + 1} entries)"
                )
        self.max_prompt_len = int(max_prompt_len)
        hidden = int(self.vl.config.text_config.hidden_size)  # 2560
        self.dim = hidden * len(self.qwen_layers)  # 7680
        self.image_size = tuple(int(v) for v in image_size)  # = data.train.vlm.image_size
        self.history_image_size = tuple(int(v) for v in history_image_size)  # what the ViT sees
        self.history_pool = (
            int(history_pool[0]),
            int(history_pool[1]),
        )  # tokens grid after pooling
        self.history_num_slots = int(history_num_slots)
        self.history_period_s = float(history_period_s)
        self.history_fps = float(history_fps)
        self.text_layout = TEXT_LAYOUT
        self.loaded_text_layout = None  # set from the checkpoint payload
        self._codec = None
        self.device = device
        self.torch_dtype = torch_dtype
        self.to(device)
        if not self.lora_enabled:
            self.eval()
        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "VLPromptEncoder(Qwen3-VL, layers=%s, summary_layers=%s, dim=%d, lora=%s, "
            "trainable=%.1fM, lora_lr=%g, text_layout=%s, image=%s, history=%dx%s pooled %dx%d @%.2fs)",
            self.qwen_layers,
            self.summary_layers,
            self.dim,
            "on" if self.lora_enabled else "off (frozen feature extractor)",
            n_tr / 1e6,
            self.lora_lr,
            self.text_layout,
            self.image_size,
            self.history_num_slots,
            self.history_image_size,
            self.history_pool[0],
            self.history_pool[1],
            self.history_period_s,
        )

    # ------------------------------------------------------------------ views
    @property
    def base_model(self):
        """Qwen3VLForConditionalGeneration, unwrapped from peft if needed."""
        m = self.vl_model
        return m.get_base_model() if hasattr(m, "get_base_model") else m

    @property
    def vl(self):
        """Qwen3VLModel (visual + language_model); LoRA is injected in place."""
        return self.base_model.model

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def lora_parameters(self):
        return [p for n, p in self.named_parameters() if p.requires_grad and "lora_" in n]

    @property
    def codec(self):
        """Tokenizer-side codec, built from this encoder's own processor so inference cannot
        drift from the prompt layout the dataloader used in training."""
        if self._codec is None:
            from awomo.data.robodojo.prompt import VLMPromptCodec

            self._codec = VLMPromptCodec(
                self.processor,
                self.image_size,
                history_image_size=self.history_image_size,
                history_pool=self.history_pool,
                history_period_s=self.history_period_s,
                max_history=self.history_num_slots,
            )
        return self._codec

    def history_offsets(self) -> list[int]:
        """Frame offsets (oldest -> newest) an eval harness must sample the VL history at."""
        return history_offsets_frames(self.history_num_slots, self.history_period_s * self.history_fps)

    def build_infer_batch(self, instructions, images, histories=None):
        """(instructions, current PIL frames, per-row lists of history PIL frames) -> collated `vb`.

        Goes through the SAME codec and collate as the dataloader, so the token layout at
        inference is identical to training by construction.
        """
        from awomo.data.robodojo.prompt import collate_vlm

        if histories is None:
            histories = [[] for _ in instructions]
        if not (len(instructions) == len(images) == len(histories)):
            raise ValueError(
                f"#instructions={len(instructions)} #images={len(images)} #histories={len(histories)} must agree"
            )
        items = [
            self.codec.encode_sample(str(i).strip(), img, list(h))
            for i, img, h in zip(instructions, images, histories)
        ]
        vb = collate_vlm(items)
        dev = next(self.vl.parameters()).device
        return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in vb.items()}

    # ------------------------------------------------------------- the gather
    def _gather_text(self, hs, vb):
        """The one place the DiT text segment is built (training AND inference).

        -> (text [B, Li+1, 7680], valid [B, Li+1] bool):
          rows 0..Li-1  hidden states of `qwen_layers` at the instruction positions (pad -> 0);
          row  Li       the SUMMARY token: hidden states of `summary_layers` at `prompt_len-1`,
                        the last prompt token -- always valid.
        """
        pos = vb["instr_pos"]  # [B,Li], -1 = pad
        valid = pos >= 0
        feats = torch.cat([hs[k] for k in self.qwen_layers], dim=-1)  # [B,L,7680]
        text = torch.gather(feats, 1, pos.clamp(min=0).unsqueeze(-1).expand(-1, -1, feats.shape[-1]))
        text = text * valid.unsqueeze(-1).to(text.dtype)
        batch = int(pos.shape[0])
        last = (vb["prompt_len"] - 1).clamp(min=0).to(pos.device)  # [B]
        rows = torch.arange(batch, device=pos.device)
        summary = torch.cat([hs[k][rows, last] for k in self.summary_layers], dim=-1)  # [B,7680]
        text = torch.cat([text, summary.unsqueeze(1).to(text.dtype)], dim=1)
        valid = torch.cat([valid, torch.ones(batch, 1, dtype=torch.bool, device=pos.device)], dim=1)
        return text, valid

    # ------------------------------------------------------------- checkpoint
    def lora_state_dict(self):
        from peft import get_peft_model_state_dict

        return {k: v.detach().cpu() for k, v in get_peft_model_state_dict(self.vl_model).items()}

    def load_lora_state_dict(self, state_dict):
        from peft import set_peft_model_state_dict

        return set_peft_model_state_dict(self.vl_model, state_dict)

    # ------------------------------------------------- history pooling + forward
    def _image_token_counts(self, grid):
        return ((grid[:, 0] * grid[:, 1] * grid[:, 2]) // (self.merge_size**2)).tolist()

    def _pool_history(self, feats, vb):
        """Pool every HISTORY image's merged-token grid down to the language-tower grid.

        `feats` [N_total, D] holds the ViT output for every image of the batch in prompt order
        (each image = t*H*W merged tokens, raster order). History frames (256x256 -> 8x8) are
        adaptive-avg-pooled to `image_grid_thw_llm` (8/merge x 8/merge = 4x4 = 16 tokens), the
        current frames pass through. Also applied to each deepstack layer (same layout)."""
        m = self.merge_size
        chunks = list(feats.split(self._image_token_counts(vb["image_grid_thw"]), dim=0))
        is_hist = vb["image_is_hist"].tolist()
        hist_idx = [i for i, h in enumerate(is_hist) if h]
        if not hist_idx:
            return feats
        g_act = vb["image_grid_thw"].tolist()
        g_llm = vb["image_grid_thw_llm"].tolist()
        ref_act, ref_llm = g_act[hist_idx[0]], g_llm[hist_idx[0]]
        for i in hist_idx:
            if g_act[i] != ref_act or g_llm[i] != ref_llm:
                raise ValueError(f"history frames must share one grid: {g_act[i]}/{g_llm[i]} vs {ref_act}/{ref_llm}")
        t, H, W = ref_act[0], ref_act[1] // m, ref_act[2] // m
        th, tw = ref_llm[1] // m, ref_llm[2] // m
        x = torch.stack([chunks[i] for i in hist_idx], dim=0)  # [n, t*H*W, D]
        n, _, D = x.shape
        x = x.reshape(n * t, H, W, D).permute(0, 3, 1, 2)  # [n*t, D, H, W]
        x = F.adaptive_avg_pool2d(x.float(), (th, tw)).to(feats.dtype)  # [n*t, D, th, tw]
        x = x.permute(0, 2, 3, 1).reshape(n, t * th * tw, D)
        for j, i in enumerate(hist_idx):
            chunks[i] = x[j]
        return torch.cat(chunks, dim=0)

    def _forward(self, vb, use_cache=False, output_hidden_states=True, past_key_values=None):
        """Custom Qwen3-VL forward: ViT features -> history pooling -> placeholder scatter ->
        language tower. `image_grid_thw` (ViT) splits the features, `image_grid_thw_llm`
        (language tower) drives the placeholder count and M-RoPE positions.
        -> (language_model output, position_ids [3,B,L], rope_deltas [B,1])."""
        vl = self.vl
        input_ids, attention_mask = vb["input_ids"], vb["attention_mask"]
        inputs_embeds = vl.get_input_embeddings()(input_ids)
        # The ViT is frozen and pixel_values carry no grad, so this branch builds no graph.
        with torch.no_grad():
            vision = vl.get_image_features(vb["pixel_values"].to(vl.dtype), vb["image_grid_thw"])
            if hasattr(vision, "pooler_output"):
                image_embeds, deepstack = (
                    vision.pooler_output,
                    vision.deepstack_features,
                )
            else:
                image_embeds, deepstack = vision[0], vision[1]
            if isinstance(image_embeds, (list, tuple)):
                image_embeds = torch.cat(list(image_embeds), dim=0)
            image_embeds = self._pool_history(image_embeds, vb)
            if deepstack is not None:
                deepstack = [
                    self._pool_history(torch.cat(list(d), 0) if isinstance(d, (list, tuple)) else d, vb)
                    for d in deepstack
                ]
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = vl.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        visual_pos_masks = image_mask[..., 0]
        # transformers 5.5 signature: (input_ids, mm_token_type_ids, image_grid_thw=, video_grid_thw=,
        # attention_mask=). The LANGUAGE-TOWER grid goes in here: it must agree with the placeholder
        # count, which is the pooled one for history frames.
        position_ids, rope_deltas = vl.get_rope_index(
            input_ids,
            mm_token_type_ids=vb["mm_token_type_ids"],
            image_grid_thw=vb["image_grid_thw_llm"],
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
        cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
        out = vl.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack,
            output_hidden_states=output_hidden_states,
        )
        return out, position_ids, rope_deltas

    def _run(self, vb):
        return self._forward(vb)[0].hidden_states

    def forward_vlm(self, vb):
        """Return instruction/history-conditioned features and validity mask."""
        hs = self._run(vb)
        text, valid = self._gather_text(hs, vb)
        return text.to(self.torch_dtype), valid

    @torch.no_grad()
    def infer_vlm(self, vb):
        """Use the same feature extraction for inference."""
        return self.forward_vlm(vb)
