# Copyright (c) 2026 Awomo-WAM Team. Licensed under the Apache License, Version 2.0.
"""RoboCasa365 policy for Awomo-0.5I (runs inside deploy/server.py): raw observations in, 16 actions (20 Hz) out."""

from __future__ import annotations

import random
import subprocess
import time
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transforms_F

from eval_robocasa365.common import (
    ACTION_MASK,
    CAMERA_NAMES,
    ENGINE_ORDER,
    REPLAN_STEPS,
    action12,
    check_timeline,
    parse_request,
    state_vector,
    to_engine_order,
)

LAYOUT_SENTENCE = (
    "Each image is one camera frame split into 3 views: the top view shows the robot's wrist camera "
    "(`robot0_eye_in_hand`), the bottom-left view shows the primary external camera (`robot0_agentview_left`), "
    "and the bottom-right view shows the secondary external camera (`robot0_agentview_right`)."
)


def h264_roundtrip(frames: np.ndarray) -> np.ndarray:
    """[T, C, H, W, 3] uint8: encode + decode each camera stream with libx264 CRF 23."""
    import imageio_ffmpeg

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    count, cameras, height, width, _ = frames.shape
    out = np.empty_like(frames)
    for camera in range(cameras):
        stream = np.ascontiguousarray(frames[:, camera])
        encoded = subprocess.run(
            [exe, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-r", "5", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
             "-f", "h264", "-"], input=stream.tobytes(), capture_output=True, check=True).stdout
        decoded = subprocess.run(
            [exe, "-hide_banner", "-loglevel", "error", "-f", "h264", "-i", "-", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            input=encoded, capture_output=True, check=True).stdout
        out[:, camera] = np.frombuffer(decoded, dtype=np.uint8).reshape(count, height, width, 3)
    return out


def compose_canvas(frames: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """[T, 3, 256, 256, 3] uint8 -> [T, h, w, 3] float32 in [0, 1] (wrist on top, agentviews below)."""
    canvas = np.zeros((frames.shape[0], 384, 256, 3), dtype=np.uint8)
    canvas[:, 0:256, 0:256] = frames[:, 2]
    for index in range(frames.shape[0]):
        canvas[index, 256:384, 0:128] = np.asarray(Image.fromarray(frames[index, 0]).resize((128, 128), Image.Resampling.LANCZOS))
        canvas[index, 256:384, 128:256] = np.asarray(Image.fromarray(frames[index, 1]).resize((128, 128), Image.Resampling.LANCZOS))
    video = torch.from_numpy(canvas).permute(0, 3, 1, 2).to(torch.float32) / 255.0
    video = transforms_F.resize(video, size=[int(size_hw[0]), int(size_hw[1])], interpolation=InterpolationMode.BILINEAR,
                                antialias=True).clamp(0.0, 1.0)
    return video.permute(0, 2, 3, 1).contiguous().numpy()


def action_rows(raw: np.ndarray) -> np.ndarray:
    """Model output [16, 80] -> RoboCasa actions [16, 12] in convert_action order."""
    a = np.asarray(raw, dtype=np.float32).copy()
    a[:, ~ACTION_MASK] = 0.0
    return to_engine_order(action12(a))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def sample_actions(model, encoded: dict[str, Any], seed: int) -> np.ndarray:
    """Flow-matching sampling of one action chunk; inactive action slots follow the noise path. -> [16, 80]."""
    target = encoded["target"]
    seed_everything(int(seed))
    scheduler = model.config["scheduler"]
    steps, shift, scale = int(scheduler["num_inference_steps"]), float(scheduler["shift"]), int(scheduler["timestep_scale"])
    sigma_min = 0.003 / 1.002
    sigmas = torch.linspace(sigma_min + (1.0 - sigma_min), sigma_min, steps)
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    timesteps = sigmas * scale
    noisy_image = torch.randn_like(target)
    noisy_action = torch.randn((1, model.action_horizon, model.action_dim), device=model.device, dtype=target.dtype)
    unused = torch.from_numpy(~ACTION_MASK).to(model.device)
    initial_noise = noisy_action.clone()
    for index, timestep in enumerate(timesteps.to(model.device)):
        image_velocity, action_velocity = model.velocity(encoded, noisy_image, noisy_action, timestep)
        action_velocity = torch.where(unused, initial_noise.to(action_velocity.dtype), action_velocity)
        step = int(torch.argmin((timesteps - timestep.cpu()).abs()))
        delta = (0 if index == steps - 1 else sigmas[step + 1]) - sigmas[step]
        noisy_image = noisy_image + image_velocity * delta
        noisy_action = noisy_action + action_velocity * delta
    return noisy_action[0].float().cpu().numpy()


class RoboCasaPolicy:
    """One episode at a time; a request with replan_index 0 starts a new episode."""

    def __init__(self, model, compress_inputs: bool = True) -> None:
        from awomo.awomo_05i.prompt import PromptEncoder

        self.model = model
        self.prompt = PromptEncoder(model.processor, LAYOUT_SENTENCE)
        self.compress_inputs = compress_inputs
        self.slots, self.period = model.history_slots, model.history_period
        self.episode: dict[str, Any] | None = None
        self.metadata = {"model": "awomo-0.5i", "benchmark": "robocasa365", "action_horizon": REPLAN_STEPS,
                         "camera_names": list(CAMERA_NAMES)}

    def _update(self, request: dict[str, Any], frames: np.ndarray, steps: list[int]) -> dict[str, Any]:
        check_timeline(request, steps, self.episode)
        replan = int(request["replan_index"])
        if replan == 0:
            self.episode = {"id": str(request["session_id"]), "instruction": str(request["instruction"]),
                            "replan": 0, "frames": {0: frames[0].copy()}, "step": 0}
            return self.episode
        ep = self.episode
        for index, step in enumerate(steps):
            ep["frames"][step] = frames[index].copy()
        current = steps[-1]
        for step in [s for s in ep["frames"] if s < current - self.slots * self.period]:
            del ep["frames"][step]
        ep["step"], ep["replan"] = current, replan
        return ep

    def _sample(self, ep: dict[str, Any], state16: np.ndarray) -> dict[str, Any]:
        current = int(ep["step"])
        indices = current - self.period * np.arange(self.slots, 0, -1, dtype=np.int64)
        valid = indices >= 0
        steps = [int(s) for s in indices[valid]] + [current]
        frames = compose_canvas(np.stack([ep["frames"][s] for s in steps]), self.model.video_size)
        video = frames[-1:]
        history = np.zeros((self.slots, *video.shape[1:]), dtype=np.float32)
        history[valid] = frames[:-1]
        return {"video": np.ascontiguousarray(video), "history": history, "history_valid": valid,
                "state": state_vector(state16),
                "vl": self.prompt.encode(str(ep["instruction"]).strip(), video[0], list(frames[:-1]))}

    def predict(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        steps, frames, state16 = parse_request(request)
        if self.compress_inputs:
            frames = h264_roundtrip(frames)
        ep = self._update(request, frames, steps)
        raw = sample_actions(self.model, self.model.encode(self._sample(ep, state16)), int(request["seed"]))
        return {"actions": action_rows(raw), "raw": raw, "latency_seconds": time.perf_counter() - started}

    def act(self, request: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        result = self.predict(request)
        return result["actions"], {"latency_seconds": result["latency_seconds"]}


def load_policy(checkpoint: str, qwen_path: str | None = None, vae_path: str | None = None, device: str = "cuda"):
    """Entry point used by deploy/server.py."""
    from awomo.awomo_05i.model import Awomo05i

    return RoboCasaPolicy(Awomo05i(checkpoint, device=device, qwen_path=qwen_path, vae_path=vae_path))
