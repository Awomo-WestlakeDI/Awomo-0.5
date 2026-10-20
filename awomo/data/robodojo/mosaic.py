# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hero-top camera mosaic composition and whole-image resizing."""

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def compose_robodojo_rgb_mosaic(frames: Sequence[np.ndarray], *, color_order: str = "rgb") -> np.ndarray:
    """Compose RGB camera views with the head view above the two wrist views.

    Resize with at most1.05 aspect adjustment, LANCZOS interpolation and symmetric center cropping.

    Args:
        frames: Head, left-wrist and right-wrist uint8 RGB arrays, in that order.
        color_order: Channel order; BGR swaps red/blue before composition and augmentation.

    Returns:
        uint8 mosaic with shape [1120,960,3].
    """
    if color_order not in ("rgb", "bgr"):
        raise ValueError(f"Unsupported mosaic color_order={color_order!r}")
    if len(frames) != 3:
        raise ValueError("RoboDojo hero-top mosaic requires head/left-wrist/right-wrist views")
    output = np.zeros((1120, 960, 3), dtype=np.uint8)
    tiles = ((0, 0, 960, 736), (0, 736, 480, 384), (480, 736, 480, 384))
    for frame, (x, y, width, height) in zip(frames, tiles, strict=True):
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Each mosaic view must be a uint8 RGB array [H,W,3]")
        # Apply the configured channel order once for current/history/VLM images.
        if color_order == "bgr":
            frame = np.ascontiguousarray(frame[..., ::-1])
        source_height, source_width = frame.shape[:2]
        if min(source_height, source_width) <= 0:
            raise ValueError("Empty camera frame cannot be used for a mosaic")
        source_aspect = source_width / source_height
        target_aspect = width / height
        anisotropy = min(max(target_aspect / source_aspect, 1.0 / 1.05), 1.05)
        adjusted_aspect = source_aspect * anisotropy
        if adjusted_aspect >= target_aspect:
            resized_height = height
            resized_width = max(width, int(np.ceil(height * adjusted_aspect)))
        else:
            resized_width = width
            resized_height = max(height, int(np.ceil(width / adjusted_aspect)))
        crop_left = (resized_width - width) // 2
        crop_top = (resized_height - height) // 2
        resized = np.asarray(Image.fromarray(frame).resize((resized_width, resized_height), Image.Resampling.LANCZOS))
        output[y : y + height, x : x + width] = resized[crop_top : crop_top + height, crop_left : crop_left + width]
    return output


def resize_mosaic_for_vae(mosaic: np.ndarray, size_hw: tuple[int, int]) -> torch.Tensor:
    """Resize the full mosaic to float32 [3,H,W] in [0,1].

    Args:
        mosaic: uint8 mosaic from compose_robodojo_rgb_mosaic.
        size_hw: Target height and width.

    Returns:
        float32 image before augmentation and scaling to [-1,1].
    """
    if mosaic.shape != (1120, 960, 3) or mosaic.dtype != np.uint8:
        raise ValueError("Expected the full uint8 RGB [1120,960,3] mosaic before resizing")
    if len(size_hw) != 2 or min(size_hw) <= 0:
        raise ValueError("Mosaic output height/width must be positive")
    tensor = torch.from_numpy(mosaic.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return F.interpolate(tensor, size=size_hw, mode="bilinear", align_corners=False, antialias=True)[0]
