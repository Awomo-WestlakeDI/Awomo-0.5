# Copyright (c) 2026 Awomo-WAM Team. Licensed under the Apache License, Version 2.0.
"""RoboCasa365 policy for Awomo-0.5V (runs inside deploy/server.py): raw observations in, 16 actions (20 Hz) out.

Per request: H.264 round-trip of the new camera frames -> three-view canvas -> the latest 25 canvases (5 Hz) are
encoded and cached as clean context -> one block (2 latent frames + 24 actions at 15 Hz) is sampled -> the actions
are resampled to 20 Hz and the first 16 are returned in robocasa.utils.env_utils.convert_action order.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import time
from collections import deque
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transforms_F

from awomo.awomo_05v.scheduler import FlowUniPCMultistepScheduler
from eval_robocasa365.common import (
    ACTION_DIM,
    ACTION_MASK,
    CAMERA_NAMES,
    ENGINE_ORDER,
    REPLAN_STEPS,
    STATE_MASK,
    action12,
    check_timeline,
    parse_request,
    state_vector,
    to_engine_order,
)

HISTORY_BUFFER_FRAMES = 33
HISTORY_FRAMES = 25
CANVAS_HEIGHT, CANVAS_WIDTH = 384, 320
MAX_ANISOTROPY = 1.05
ACTIONS_PER_CHUNK = 24  # 15 Hz
RAW_STEPS_PER_CHUNK = 32  # 20 Hz
CANONICAL_DIM = ACTION_DIM
# 16 scheduler steps; the DiT runs on the marked ones and the other steps reuse its last prediction.
DIT_STEP_MASK = tuple(flag == "T" for flag in "TTTFFFTFFFTFFTTT")
H264_ENCODE_ARGS = ("-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p")
PROMPT_SEED = 42
PROMPT_TEMPLATES = (
    "{instruction}.",
    "Task: {instruction}.",
    "Instruction: {instruction}.",
    "Robot task: {instruction}.",
    "Task instruction: {instruction}.",
    "Requested task: {instruction}.",
    "Assigned task: {instruction}.",
    "Target task: {instruction}.",
    "Manipulation task: {instruction}.",
    "Robot instruction: {instruction}.",
    "Goal: {instruction}.",
    "Objective: {instruction}.",
    "Task goal: {instruction}.",
    "Robot goal: {instruction}.",
    "Target behavior: {instruction}.",
    "Requested behavior: {instruction}.",
    "Required behavior: {instruction}.",
    "Desired behavior: {instruction}.",
    "Intended behavior: {instruction}.",
    "Demonstrated task: {instruction}.",
    "The task is: {instruction}.",
    "The instruction is: {instruction}.",
    "The robot task is: {instruction}.",
    "The requested task is: {instruction}.",
    "The assigned task is: {instruction}.",
    "The target task is: {instruction}.",
    "The required task is: {instruction}.",
    "The intended task is: {instruction}.",
    "The manipulation task is: {instruction}.",
    "The task objective is: {instruction}.",
    "The robot's task is: {instruction}.",
    "The robot's goal is: {instruction}.",
    "The robot's objective is: {instruction}.",
    "The goal for the robot is: {instruction}.",
    "The objective for the robot is: {instruction}.",
    "The requested behavior is: {instruction}.",
    "The required behavior is: {instruction}.",
    "The desired behavior is: {instruction}.",
    "The intended behavior is: {instruction}.",
    "The target behavior is: {instruction}.",
    "Perform this task: {instruction}.",
    "Execute this task: {instruction}.",
    "Complete this task: {instruction}.",
    "Carry out this task: {instruction}.",
    "Follow this task instruction: {instruction}.",
    "Perform the following task: {instruction}.",
    "Execute the following task: {instruction}.",
    "Complete the following task: {instruction}.",
    "Carry out the following task: {instruction}.",
    "Follow the following instruction: {instruction}.",
    "Use this instruction: {instruction}.",
    "Act on this instruction: {instruction}.",
    "Proceed with this task: {instruction}.",
    "Work on this task: {instruction}.",
    "Fulfill this task: {instruction}.",
    "The robot should perform this task: {instruction}.",
    "The robot should execute this task: {instruction}.",
    "The robot should complete this task: {instruction}.",
    "The robot should carry out this task: {instruction}.",
    "The robot should follow this instruction: {instruction}.",
    "The robot must perform this task: {instruction}.",
    "The robot must execute this task: {instruction}.",
    "The robot must complete this task: {instruction}.",
    "The robot must carry out this task: {instruction}.",
    "The robot must follow this instruction: {instruction}.",
    "The robot needs to perform this task: {instruction}.",
    "The robot needs to execute this task: {instruction}.",
    "The robot needs to complete this task: {instruction}.",
    "The robot needs to carry out this task: {instruction}.",
    "The robot needs to follow this instruction: {instruction}.",
    "The robot is asked to perform: {instruction}.",
    "The robot is asked to execute: {instruction}.",
    "The robot is asked to complete: {instruction}.",
    "The robot is asked to carry out: {instruction}.",
    "The robot is asked to follow this instruction: {instruction}.",
    "The robot is instructed to perform: {instruction}.",
    "The robot is instructed to execute: {instruction}.",
    "The robot is instructed to complete: {instruction}.",
    "The robot is instructed to carry out: {instruction}.",
    "The robot is instructed as follows: {instruction}.",
    "This robot task is: {instruction}.",
    "This task asks the robot to: {instruction}.",
    "This instruction asks the robot to: {instruction}.",
    "This demonstration performs the task: {instruction}.",
    "This sequence demonstrates the task: {instruction}.",
    "The demonstrated behavior is: {instruction}.",
    "The demonstrated instruction is: {instruction}.",
    "The demonstrated objective is: {instruction}.",
    "The behavior to perform is: {instruction}.",
    "The action to perform is: {instruction}.",
    "The task to perform is: {instruction}.",
    "The task to execute is: {instruction}.",
    "The task to complete is: {instruction}.",
    "The instruction to follow is: {instruction}.",
    "The objective to achieve is: {instruction}.",
    "The goal to achieve is: {instruction}.",
    "For this sample, the task is: {instruction}.",
    "For this sequence, the task is: {instruction}.",
    "For this demonstration, the task is: {instruction}.",
    "For the robot, the task is: {instruction}.",
)


# ----------------------------------------------------------------------------------------------- observations


def ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:  # noqa: BLE001 - fall back to PATH
        path = shutil.which("ffmpeg")
        if path is None:
            raise RuntimeError("H.264 round-trip needs ffmpeg (imageio-ffmpeg or PATH)") from None
        return path


def h264_roundtrip(frames: np.ndarray, executable: str) -> np.ndarray:
    """One camera's new frames [T, 256, 256, 3] uint8 -> one 5 Hz libx264 CRF 23 clip -> decoded frames."""
    frames = np.ascontiguousarray(frames)
    height, width = frames.shape[1:3]
    encoded = subprocess.run(
        [executable, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", "5", "-i", "-", *H264_ENCODE_ARGS, "-f", "h264", "-"],
        input=frames.tobytes(), capture_output=True, check=True).stdout
    decoded = subprocess.run(
        [executable, "-hide_banner", "-loglevel", "error", "-f", "h264", "-i", "-", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        input=encoded, capture_output=True, check=True).stdout
    out = np.frombuffer(decoded, dtype=np.uint8)
    if out.size != frames.size:
        raise RuntimeError("H.264 round-trip returned a different number of frames")
    return out.reshape(frames.shape).copy()


def resize_crop_light_stretch(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bounded stretch (<= 1.05) + LANCZOS resize + centered crop."""
    source_height, source_width = frame.shape[:2]
    source_aspect = source_width / source_height
    target_aspect = width / height
    anisotropy = min(max(target_aspect / source_aspect, 1.0 / MAX_ANISOTROPY), MAX_ANISOTROPY)
    adjusted_aspect = source_aspect * anisotropy
    if adjusted_aspect >= target_aspect:
        resized_height = height
        resized_width = max(width, int(np.ceil(height * adjusted_aspect)))
    else:
        resized_width = width
        resized_height = max(height, int(np.ceil(width / adjusted_aspect)))
    resized = np.asarray(Image.fromarray(frame).resize((resized_width, resized_height), Image.Resampling.LANCZOS))
    crop_left = (resized_width - width) // 2
    crop_top = (resized_height - height) // 2
    return np.ascontiguousarray(resized[crop_top:crop_top + height, crop_left:crop_left + width])


def compose_canvas(left: np.ndarray, right: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """Wrist camera on top (320 x 256), left and right external cameras below (160 x 128 each) -> [384, 320, 3]."""
    canvas = np.empty((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
    canvas[:256] = resize_crop_light_stretch(wrist, width=320, height=256)
    canvas[256:, :160] = resize_crop_light_stretch(left, width=160, height=128)
    canvas[256:, 160:] = resize_crop_light_stretch(right, width=160, height=128)
    return canvas


def canvases_to_video(canvases: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """[T, 384, 320, 3] uint8 -> [T, H, W, 3] float32 in [0, 1]."""
    video = torch.from_numpy(canvases).permute(0, 3, 1, 2).to(torch.float32) / 255.0
    video = transforms_F.resize(video, size=[int(size_hw[0]), int(size_hw[1])], interpolation=InterpolationMode.BILINEAR,
                                antialias=True).clamp(0.0, 1.0)
    return video.permute(0, 2, 3, 1).contiguous().numpy()


def build_prompt(instruction: str, seed: int) -> str:
    """One of 100 fixed templates, chosen by the instruction and the request seed."""
    digest = hashlib.sha256(f"{PROMPT_SEED}:{instruction}:{seed}".encode()).digest()
    template = PROMPT_TEMPLATES[int.from_bytes(digest[:8], byteorder="big") % len(PROMPT_TEMPLATES)]
    return template.format(instruction=instruction.strip().rstrip("."))


# ----------------------------------------------------------------------------------------------- actions


def _resample_table() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """20 Hz step i lies at 15 Hz index w_i = 3(i + 1) / 4 - 1; step 0 precedes the first sample and takes it."""
    i = np.arange(RAW_STEPS_PER_CHUNK, dtype=np.int64)
    numerator = 3 * (i + 1) - 4
    k0 = np.where(numerator < 0, 0, numerator // 4)
    mu = np.where(numerator < 0, 0.0, (numerator % 4).astype(np.float64) / 4.0)
    k1 = np.minimum(k0 + 1, ACTIONS_PER_CHUNK - 1)
    nearest = np.where(mu < 0.5, k0, k1)
    return k0, k1, mu, nearest


def action_rows(chunk: np.ndarray) -> np.ndarray:
    """Model output [24, 80] at 15 Hz -> [32, 12] at 20 Hz [base(3), torso, mode, eef_pos(3), eef_rot(3), gripper]."""
    a = np.array(chunk, dtype=np.float64, copy=True)
    if np.any(a[:, ~ACTION_MASK] != 0):
        raise ValueError("inactive action slots must end at exactly zero")
    a[:, ~ACTION_MASK] = 0.0
    k0, k1, mu, nearest = _resample_table()
    interpolated = (1.0 - mu[:, None]) * a[k0] + mu[:, None] * a[k1]
    return action12(interpolated, discrete=a[nearest])


@torch.no_grad()
def sample_chunk(model, latents: torch.Tensor, prompt_embs: torch.Tensor, kv_cache: list[torch.Tensor],
                 state80: np.ndarray, seed: int) -> np.ndarray:
    """Jointly denoise the next video block and action chunk (16 UniPC steps, shift 5, guidance scale 1).

    Inactive action slots follow the noise path: their input stays at sigma * eps and their velocity is eps.
    """
    device = model.device
    scheduler = model.config["scheduler"]
    steps, shift = int(scheduler["num_inference_steps"]), float(scheduler["shift"])
    kwargs = {"num_train_timesteps": int(scheduler["num_train_timesteps"]), "shift": 1, "use_dynamic_shifting": False}
    video_scheduler = FlowUniPCMultistepScheduler(**kwargs)
    action_scheduler = FlowUniPCMultistepScheduler(**kwargs)
    video_scheduler.set_timesteps(steps, device=device, shift=shift)
    action_scheduler.set_timesteps(steps, device=device, shift=shift)

    _, channels, latent_frames, height, width = latents.shape
    video_generator = torch.Generator(device=device).manual_seed(int(seed))
    action_generator = torch.Generator(device=device).manual_seed(int(seed) + 1)
    x_video = torch.randn(1, channels, model.num_frame_per_block, height, width, device=device,
                          dtype=torch.bfloat16, generator=video_generator)
    x_action = torch.randn(1, ACTIONS_PER_CHUNK, model.action_dim, device=device, dtype=torch.bfloat16,
                           generator=action_generator)
    ones = torch.ones(CANONICAL_DIM, device=device, dtype=torch.bfloat16).view(1, 1, CANONICAL_DIM)
    inactive = torch.from_numpy(~ACTION_MASK).to(device)
    x_action = x_action * ones
    eps = x_action[..., inactive].clone()
    state = torch.from_numpy(state80).to(device=device, dtype=torch.bfloat16).view(1, 1, CANONICAL_DIM)

    video_pred = action_pred = None
    for index, (video_t, action_t) in enumerate(zip(video_scheduler.timesteps, action_scheduler.timesteps)):
        if DIT_STEP_MASK[index]:
            video_pred, action_pred = model.velocity(x_video, x_action, video_t, action_t, state, prompt_embs,
                                                     kv_cache, int(latent_frames))
            action_pred = action_pred * ones
        x_video = video_scheduler.step(video_pred, video_t, x_video, step_index=index, return_dict=False)[0]
        velocity = action_pred.clone()
        velocity[..., inactive] = eps.to(velocity.dtype)
        x_next = action_scheduler.step(velocity, action_t, x_action, step_index=index, return_dict=False)[0]
        sigma_next = 0.0 if index + 1 >= len(action_scheduler.timesteps) else float(action_scheduler.sigmas[index + 1])
        x_next[..., inactive] = (sigma_next * eps.float()).to(x_next.dtype)
        x_action = x_next * ones
    return x_action.float().cpu().numpy()[0]


# ----------------------------------------------------------------------------------------------- policy


class RoboCasaPolicy:
    """One episode at a time; a request with replan_index 0 starts a new episode."""

    def __init__(self, model, compress_inputs: bool = True) -> None:
        steps = int(model.config["scheduler"]["num_inference_steps"])
        if steps != len(DIT_STEP_MASK):
            raise ValueError(f"DIT_STEP_MASK covers {len(DIT_STEP_MASK)} scheduler steps, the checkpoint uses {steps}")
        self.model = model
        self.compress_inputs = compress_inputs
        self.ffmpeg = ffmpeg_executable() if compress_inputs else None
        self.episode: dict[str, Any] | None = None
        self.metadata = {"model": "awomo-0.5v", "benchmark": "robocasa365", "action_horizon": REPLAN_STEPS,
                         "camera_names": list(CAMERA_NAMES)}

    def _canvases(self, frames: np.ndarray) -> np.ndarray:
        views = [frames[:, camera] for camera in range(3)]
        if self.compress_inputs:
            views = [h264_roundtrip(view, self.ffmpeg) for view in views]
        return np.stack([compose_canvas(views[0][i], views[1][i], views[2][i]) for i in range(frames.shape[0])])

    def _update(self, request: dict[str, Any], canvases: np.ndarray, steps: list[int]) -> dict[str, Any]:
        check_timeline(request, steps, self.episode)
        replan = int(request["replan_index"])
        instruction = str(request["instruction"])
        if replan == 0:
            self.episode = {"id": str(request["session_id"]), "instruction": instruction, "replan": 0,
                            "canvases": deque([canvases[0].copy()], maxlen=HISTORY_BUFFER_FRAMES), "step": 0}
            return self.episode
        ep = self.episode
        if instruction != ep["instruction"]:
            raise ValueError("instruction changed within an episode")
        for canvas in canvases:
            ep["canvases"].append(canvas.copy())
        ep["step"], ep["replan"] = steps[-1], replan
        return ep

    def predict(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        steps, frames, state16 = parse_request(request)
        seed = int(request["seed"])
        ep = self._update(request, self._canvases(frames), steps)
        window = np.stack(list(ep["canvases"])[-HISTORY_FRAMES:])
        model = self.model
        prompt_embs = model.encode_text(build_prompt(ep["instruction"], seed))
        latents = model.encode_video(canvases_to_video(window, model.video_size))
        kv_cache = model.warm_up(latents, prompt_embs)
        chunk = sample_chunk(model, latents, prompt_embs, kv_cache, state_vector(state16), seed)
        del kv_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        native = action_rows(chunk)
        actions = to_engine_order(native[:REPLAN_STEPS])
        return {"actions": actions, "raw": chunk, "native": native, "latency_seconds": time.perf_counter() - started}

    def act(self, request: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        result = self.predict(request)
        return result["actions"], {"latency_seconds": result["latency_seconds"]}


def load_policy(checkpoint: str, wan_path: str | None = None, device: str = "cuda"):
    """Entry point used by deploy/server.py."""
    from awomo.awomo_05v.model import Awomo05v

    return RoboCasaPolicy(Awomo05v(checkpoint, device=device, wan_path=wan_path))
