# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 Awomo-WAM Team.
# Licensed under the Apache License, Version 2.0.
"""Awomo-0.5V backbone: causal Wan2.2 DiT with action and state tokens (inference path with a request-local KV cache)."""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        return torch.cuda.get_device_capability()[0] == 9

    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False


def _gpu_supports_flash_attention():
    """FlashAttention requires Ampere (compute capability 8.0) or newer."""
    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        return False
    try:
        if not torch.cuda.is_available():
            return False
        cap = torch.cuda.get_device_capability()
        return cap[0] >= 8
    except Exception:
        return False


def _sdpa_attention_fallback(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    dtype=torch.bfloat16,
):
    """PyTorch SDPA fallback for GPUs that don't support FlashAttention (e.g. pre-Ampere)."""
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            "Padding mask is disabled when using scaled_dot_product_attention on this GPU. "
            "It can have a slight impact on quality."
        )
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)
    if q_scale is not None:
        q = q * q_scale
    if softmax_scale is not None:
        q = q * softmax_scale
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=causal, dropout_p=dropout_p
    )
    return out.transpose(1, 2).contiguous()


def _flash_attention_varlen_impl(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    deterministic,
    version,
):
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        return flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )
    assert FLASH_ATTN_2_AVAILABLE
    return flash_attn.flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
    )


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    if window_size is None:
        window_size = (-1, -1)

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    # Use PyTorch SDPA on pre-Ampere GPUs (FlashAttention requires Ampere or newer)
    if not _gpu_supports_flash_attention():
        return _sdpa_attention_fallback(
            q,
            k,
            v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            dtype=dtype,
        )

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn("Flash attention 3 is not available, use flash attention 2 instead.")

    cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
    cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)
    x = _flash_attention_varlen_impl(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=lq,
        max_seqlen_k=lk,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        version=version,
    ).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)



class AttentionModule(nn.Module):
    """FlashAttention-2 when available (Ampere or newer), otherwise PyTorch SDPA."""

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()

    def forward(self, q, k, v):
        return flash_attention(q=q, k=k, v=v, dtype=torch.bfloat16, version=2)

def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half, dtype=position.dtype, device=position.device).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def rope_params(max_seq_len: int, dim: int, theta: float = 10000) -> torch.Tensor:
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)


class WanT2VCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens=None, crossattn_cache=None):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        x = flash_attention(q, k, v, k_lens=context_lens)

        x = x.flatten(2)
        x = self.o(x)
        return x

def swish(x):
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Produces a sinusoidal encoding of shape (B, T, w)
    given timesteps of shape (B, T).
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        timesteps = timesteps.float()

        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)

        return enc


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B, T)
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        a_emb = self.W1(actions, cat_ids)

        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        x = self.W3(x, cat_ids)
        return x


def causal_rope_action_apply(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int,
    num_state_per_block: int,
    action_state_index: int,
):
    B, seq_len, n, _ = x.shape

    x = torch.view_as_complex(x.to(torch.float64).reshape(B, seq_len, n, -1, 2))

    if action_register_length is not None:
        assert action_register_length == (num_action_per_block + num_state_per_block)
        freqs_action = freqs_action[
            action_state_index * num_action_per_block : (action_state_index + 1) * num_action_per_block
        ]
        freqs_state = freqs_state[
            action_state_index * num_state_per_block : (action_state_index + 1) * num_state_per_block
        ]
        freqs_1d = torch.cat([freqs_action, freqs_state], dim=0).view(action_register_length, 1, -1)
        freqs = torch.cat([freqs, freqs_1d], dim=0)

    freqs = freqs.unsqueeze(0)
    x = torch.view_as_real(x * freqs).flatten(3)

    return x


class CausalWanSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        frame_seqlen,
        local_attn_size=-1,
        sink_size=0,
        num_frame_per_block=1,
        qk_norm=True,
        eps=1e-6,
        num_action_per_block=32,
        num_state_per_block=1,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.num_frame_per_block = num_frame_per_block
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 21 * frame_seqlen if local_attn_size == -1 else local_attn_size * frame_seqlen
        self.frame_seqlen = frame_seqlen
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn = AttentionModule(num_heads=self.num_heads, head_dim=self.head_dim)

    def project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project x [B, L, C] to per-head Q, K and V [B, L, heads, head_dim]."""
        batch_size, sequence_length = x.shape[:2]
        q = self.norm_q(self.q(x)).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        k = self.norm_k(self.k(x)).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        v = self.v(x).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        return q, k, v

    def _update_video_kv_cache(
        self,
        kv_cache: torch.Tensor,
        video_k: torch.Tensor,
        video_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append the new video K/V; past the window keep the first frame and the most recent frames."""
        new_k = torch.cat([kv_cache[0], video_k], dim=1)
        new_v = torch.cat([kv_cache[1], video_v], dim=1)
        max_cache_size = self.frame_seqlen + self.max_attention_size
        if new_k.shape[1] > max_cache_size:
            new_k = torch.cat([new_k[:, : self.frame_seqlen], new_k[:, -self.max_attention_size :]], dim=1)
            new_v = torch.cat([new_v[:, : self.frame_seqlen], new_v[:, -self.max_attention_size :]], dim=1)
        return new_k, new_v

    def attend_cached(
        self,
        video_q: torch.Tensor,
        video_k: torch.Tensor,
        video_v: torch.Tensor,
        action_q: torch.Tensor,
        action_k: torch.Tensor,
        action_v: torch.Tensor,
        state_q: torch.Tensor,
        state_k: torch.Tensor,
        state_v: torch.Tensor,
        kv_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cached attention: video and action tokens attend to the cached + new video, action and state tokens;
        state tokens attend to themselves."""
        new_k, new_v = self._update_video_kv_cache(kv_cache, video_k, video_v)
        context_k = torch.cat([new_k, action_k, state_k], dim=1)
        context_v = torch.cat([new_v, action_v, state_v], dim=1)
        video_output = self.attn(video_q, context_k, context_v)
        action_output = self.attn(action_q, context_k, context_v)

        state_output = self.attn(state_q, state_k, state_v)
        updated_kv_cache = torch.stack([new_k, new_v], dim=0)
        return video_output, action_output, state_output, updated_kv_cache

    def forward(self, x, freqs, freqs_action, freqs_state, action_register_length, kv_cache, current_start_frame=0):
        q, k, v = self.project_qkv(x)
        updated_kv_cache = None
        action_state_index = (current_start_frame - 1) // self.num_frame_per_block

        roped_query = causal_rope_action_apply(
            x=q,
            freqs=freqs,
            freqs_action=freqs_action,
            freqs_state=freqs_state,
            action_register_length=action_register_length,
            num_action_per_block=self.num_action_per_block,
            num_state_per_block=self.num_state_per_block,
            action_state_index=action_state_index,
        ).type_as(v)
        roped_key = causal_rope_action_apply(
            x=k,
            freqs=freqs,
            freqs_action=freqs_action,
            freqs_state=freqs_state,
            action_register_length=action_register_length,
            num_action_per_block=self.num_action_per_block,
            num_state_per_block=self.num_state_per_block,
            action_state_index=action_state_index,
        ).type_as(v)

        if action_register_length is not None:
            assert action_register_length == self.num_action_per_block + self.num_state_per_block
            video_length = roped_query.shape[1] - action_register_length
            video_q = roped_query[:, :video_length]
            video_k = roped_key[:, :video_length]
            video_v = v[:, :video_length]
            action_q, state_q = roped_query[:, video_length:].split(
                [self.num_action_per_block, self.num_state_per_block],
                dim=1,
            )
            action_k, state_k = roped_key[:, video_length:].split(
                [self.num_action_per_block, self.num_state_per_block],
                dim=1,
            )
            action_v, state_v = v[:, video_length:].split(
                [self.num_action_per_block, self.num_state_per_block],
                dim=1,
            )
            video_output, action_output, state_output, updated_kv_cache = self.attend_cached(
                video_q=video_q,
                video_k=video_k,
                video_v=video_v,
                action_q=action_q,
                action_k=action_k,
                action_v=action_v,
                state_q=state_q,
                state_k=state_k,
                state_v=state_v,
                kv_cache=kv_cache,
            )
            x = torch.cat([video_output, action_output, state_output], dim=1)
        else:
            new_k, new_v = self._update_video_kv_cache(kv_cache, roped_key, v)
            x = self.attn(roped_query, new_k, new_v)
            updated_kv_cache = torch.stack([new_k, new_v], dim=0)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x, updated_kv_cache


class CausalWanAttentionBlock(nn.Module):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        frame_seqlen,
        local_attn_size=-1,
        sink_size=0,
        num_frame_per_block=1,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        num_action_per_block=32,
        num_state_per_block=1,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim=dim,
            num_heads=num_heads,
            frame_seqlen=frame_seqlen,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            num_frame_per_block=num_frame_per_block,
            qk_norm=qk_norm,
            eps=eps,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block,
        )
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanT2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, freqs, freqs_action, freqs_state, action_register_length, context, kv_cache,
                current_start_frame=0):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        L = x.shape[1]
        aligned = []
        for part in e:
            L_e = part.shape[1]
            if L_e == L:
                aligned.append(part)
            elif L_e >= L:
                aligned.append(part[:, :L])
            else:
                repeat = (L + L_e - 1) // L_e
                aligned.append(part.repeat_interleave(repeat, dim=1)[:, :L])
        e = tuple(aligned)

        y, updated_kv_cache = self.self_attn(
            x=(self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)),
            freqs=freqs,
            freqs_action=freqs_action,
            freqs_state=freqs_state,
            action_register_length=action_register_length,
            kv_cache=kv_cache,
            current_start_frame=current_start_frame,
        )
        x = x + (y * e[2].squeeze(2))

        def cross_attn_ffn(x, context, e):
            x = x + self.cross_attn(self.norm3(x), context)
            y = self.ffn((self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2)))
            x = x + (y * e[5].squeeze(2))
            return x

        x = cross_attn_ffn(x, context, e)
        return x, updated_kv_cache


class CausalHead(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        L = x.shape[1]
        aligned = []
        for part in e:
            L_e = part.shape[1]
            if L_e == L:
                aligned.append(part)
            elif L_e >= L:
                aligned.append(part[:, :L])
            else:
                repeat = (L + L_e - 1) // L_e
                aligned.append(part.repeat_interleave(repeat, dim=1)[:, :L])
        e = tuple(aligned)
        x = self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
        return x


class CausalWanModel(nn.Module):
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        frame_seqlen=220,
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        max_chunk_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        num_frame_per_block=1,
        action_dim=32,
        num_registers=8,
        max_state_dim=64,
        hidden_size=1024,
        num_action_per_block=32,
        num_state_per_block=1,
        concat_first_frame_latent=True,
        cross_attn_type=None,
    ):

        super().__init__()

        assert model_type in ["t2v", "i2v", "ti2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.frame_seqlen = frame_seqlen
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = max_chunk_size * num_frame_per_block + 1 if max_chunk_size != -1 else -1
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.num_frame_per_block = num_frame_per_block
        self.action_dim = action_dim
        self.num_registers = num_registers
        self.max_state_dim = max_state_dim
        self.hidden_size = hidden_size
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block
        self.concat_first_frame_latent = concat_first_frame_latent
        if cross_attn_type is None:
            cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        assert cross_attn_type == "t2v_cross_attn"
        self.cross_attn_type = cross_attn_type

        self.state_encoder = CategorySpecificMLP(
            num_categories=1,
            input_dim=max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=action_dim,
            hidden_size=self.dim,
            num_embodiments=1,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=1,
            input_dim=dim,
            hidden_dim=self.hidden_size,
            output_dim=action_dim,
        )

        # embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList(
            [
                CausalWanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    frame_seqlen,
                    self.local_attn_size,
                    sink_size,
                    num_frame_per_block,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    num_action_per_block,
                    num_state_per_block,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads

        self.freqs_action = rope_params(1024 * 10, d)
        self.freqs_state = rope_params(1024, d)
        self.freqs = [
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ]

    def _forward_blocks(
        self,
        x: torch.Tensor,
        seq_len: int,
        freqs: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None,
        action: torch.Tensor | None,
        timestep_action: torch.Tensor | None,
        state: torch.Tensor | None,
        kv_cache: list[torch.Tensor],
        current_start_frame: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        x = x.flatten(start_dim=2).transpose(1, 2)

        B = x.shape[0]
        F = timestep.shape[1]

        if action is not None:
            control_embodiment = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            action_features = self.action_encoder(action, timestep_action, control_embodiment)
            state_features = self.state_encoder(state, control_embodiment)
            action_register = torch.cat([action_features, state_features], dim=1)
            action_length = action_features.shape[1]
            action_register_length = action_register.shape[1]
            x = torch.cat([x, action_register], dim=1)
        else:
            action_features = None
            state_features = None
            action_length = 0
            action_register_length = None

        # time embeddings
        if F <= seq_len:
            repeat = (seq_len + F - 1) // F
            timestep = timestep.repeat_interleave(repeat, dim=1)[:, :seq_len]
        else:
            indices = torch.linspace(0, F - 1, seq_len, device=timestep.device, dtype=torch.long)
            timestep = timestep[:, indices]

        if action is not None:
            assert timestep_action is not None
            assert state_features is not None
            stride = timestep_action.shape[1] // state_features.shape[1]
            timestep_state = timestep_action[:, ::stride]
            timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x))
        e = e.unflatten(dim=0, sizes=(B, -1))
        e0 = self.time_projection(e)
        e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

        # context
        context = self.text_embedding(context)

        if clip_feature is not None:
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        updated_kv_caches: list[torch.Tensor] = []
        for block_index, block in enumerate(self.blocks):
            x, updated_kv_cache = block(
                x=x,
                e=e0,
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                context=context,
                action_register_length=action_register_length,
                kv_cache=kv_cache[block_index],
                current_start_frame=current_start_frame,
            )
            updated_kv_caches.append(updated_kv_cache)

        if action is not None:
            action_noise_pred = x[:, seq_len : seq_len + action_length]
            action_noise_pred = self.action_decoder(action_noise_pred, control_embodiment)
        else:
            action_noise_pred = None

        x_video = x[:, :seq_len]
        e_video = e[:, :seq_len]

        x_video = self.head(x_video, e_video.unsqueeze(2))

        return x_video, action_noise_pred, updated_kv_caches

    def forward(
        self,
        x,
        timestep,
        context,
        seq_len,
        kv_cache: list[torch.Tensor],
        crossattn_cache: list[torch.Tensor],
        current_start_frame: int,
        y=None,
        clip_feature=None,
        action=None,
        timestep_action=None,
        state=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        if self.model_type == "i2v":
            assert clip_feature is not None and y is not None
        if self.cross_attn_type == "t2v_cross_attn":
            assert clip_feature is None and y is None
        assert context.shape[1] == self.text_len

        if y is not None and self.concat_first_frame_latent:
            x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

        x = self.patch_embedding(x)
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)

        freqs = self._create_freqs(
            grid_size=grid_size,
            start_frame=current_start_frame,
        )

        x_video, action_noise_pred, updated_kv_caches = self._forward_blocks(
            x=x,
            seq_len=seq_len,
            freqs=freqs,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            action=action,
            timestep_action=timestep_action,
            state=state,
            kv_cache=kv_cache,
            current_start_frame=current_start_frame,
        )

        x_video = x_video.clone()
        if action_noise_pred is not None:
            action_noise_pred = action_noise_pred.clone()

        video_noise_pred = self.unpatchify(x_video, grid_size)

        return video_noise_pred, action_noise_pred, updated_kv_caches

    def unpatchify(self, x, grid_size):
        B = x.shape[0]
        c = self.out_dim
        grid_size = grid_size.tolist()
        assert x.shape[1] == math.prod(grid_size)
        x = x.view(B, *grid_size, *self.patch_size, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        x = x.reshape(B, c, *[i * j for i, j in zip(grid_size, self.patch_size)])
        return x

    def _create_freqs(
        self,
        grid_size: torch.Tensor,
        start_frame: int,
    ):
        device = self.patch_embedding.weight.device
        if any(freq.device != device for freq in self.freqs):
            self.freqs = [freq.to(device) for freq in self.freqs]
        if self.freqs_action.device != device:
            self.freqs_action = self.freqs_action.to(device)
        if self.freqs_state.device != device:
            self.freqs_state = self.freqs_state.to(device)

        f, h, w = grid_size.tolist()
        freqs = torch.cat(
            [
                self.freqs[0][start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)

        return freqs
