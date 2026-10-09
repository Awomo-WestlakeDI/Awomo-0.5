# Copyright (c) 2026 Awomo-WAM Team. Licensed under the Apache License, Version 2.0.
"""RoboCasa365 interface shared by the model adapters (server side).

Requests: 3 cameras at 256x256, frames at 5 Hz (every 4 control steps), a new chunk every 16 steps. Replan 0 sends the
step-0 frame; every later replan sends the 4 frames since the previous one.
State and actions use an 80-dim layout: eef position 7-9, eef rotation (6D state, 3D action delta) 10-15 / 10-12,
gripper closedness 16, base xy 58-59, base yaw 63, control mode 76 (action only).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

CAMERA_NAMES = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
FRAME_SHAPE = (3, 256, 256, 3)
OBSERVATION_STRIDE = 4  # 20 Hz control, 5 Hz frames
REPLAN_STEPS = 16
STATE_DIM = ACTION_DIM = 80
STATE_SLOTS = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 58, 59, 63)
ACTION_SLOTS = (7, 8, 9, 10, 11, 12, 16, 58, 59, 63, 76)
STATE_MASK = np.zeros(STATE_DIM, dtype=bool)
STATE_MASK[list(STATE_SLOTS)] = True
ACTION_MASK = np.zeros(ACTION_DIM, dtype=bool)
ACTION_MASK[list(ACTION_SLOTS)] = True
ENGINE_ORDER = (5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4)  # -> robocasa.utils.env_utils.convert_action order
GRIPPER_MAX_APERTURE_M = 0.08


# ----------------------------------------------------------------------------------------------- requests


def parse_request(request: dict[str, Any]) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Validate one request -> (observation steps, camera frames [T, 3, 256, 256, 3] uint8, state [16] float32)."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", str(request["session_id"])):
        raise ValueError("invalid session id")
    steps = [int(s) for s in request["observation_steps"]]
    frames = np.asarray(request["camera_frames"])
    if frames.dtype != np.uint8 or frames.shape[1:] != FRAME_SHAPE or frames.shape[0] != len(steps):
        raise ValueError(f"camera_frames must be uint8 [T, {FRAME_SHAPE}]")
    state16 = np.asarray(request["state"], dtype=np.float32)
    if state16.shape != (16,) or not np.isfinite(state16).all():
        raise ValueError("state must be 16 finite values")
    return steps, frames, state16


def check_timeline(request: dict[str, Any], steps: list[int], episode: dict[str, Any] | None) -> None:
    """Replan 0 must send exactly the step-0 frame; later replans continue the same episode without gaps.

    `episode` holds "id", "replan" and "step" (the last observation step) of the episode in progress.
    """
    replan = int(request["replan_index"])
    if replan == 0:
        if steps != [0]:
            raise ValueError("replan 0 must send exactly the step-0 frame")
        return
    if episode is None or episode["id"] != str(request["session_id"]) or replan != episode["replan"] + 1:
        raise ValueError("unexpected session or replan index")
    expected = list(range(episode["step"] + OBSERVATION_STRIDE, episode["step"] + REPLAN_STEPS + 1, OBSERVATION_STRIDE))
    if steps != expected:
        raise ValueError(f"expected observation steps {expected}, got {steps}")


# ----------------------------------------------------------------------------------------------- state / actions


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm <= 1.0e-8):
        raise ValueError("quaternion contains a zero-length vector")
    x, y, z, w = np.moveaxis(q / norm, -1, 0)
    m = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    m[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    m[..., 0, 1] = 2.0 * (x * y - z * w)
    m[..., 0, 2] = 2.0 * (x * z + y * w)
    m[..., 1, 0] = 2.0 * (x * y + z * w)
    m[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    m[..., 1, 2] = 2.0 * (y * z - x * w)
    m[..., 2, 0] = 2.0 * (x * z - y * w)
    m[..., 2, 1] = 2.0 * (y * z + x * w)
    m[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return m


def state_vector(state16: np.ndarray) -> np.ndarray:
    """State16 [base_pos(3), base_quat(4), eef_pos(3), eef_quat(4), gripper_qpos(2)] -> model state [1, 80] float32."""
    s = np.asarray(state16, dtype=np.float64)[None]
    out = np.zeros((1, STATE_DIM), dtype=np.float64)
    out[:, 7:10] = s[:, 7:10]
    r = _quaternion_to_matrix(s[:, 10:14])
    out[:, 10:16] = np.concatenate((r[..., :, 0], r[..., :, 1]), axis=-1)
    out[:, 16] = np.clip(1.0 - np.abs(s[:, 14] - s[:, 15]) / GRIPPER_MAX_APERTURE_M, 0.0, 1.0)
    out[:, 58:60] = s[:, 0:2]
    b = s[:, 3:7]
    norm = np.linalg.norm(b, axis=-1, keepdims=True)
    if np.any(norm <= 1.0e-8):
        raise ValueError("base quaternion contains a zero-length vector")
    x, y, z, w = np.moveaxis(b / norm, -1, 0)
    out[:, 63] = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    out[:, ~STATE_MASK] = 0.0
    return out.astype(np.float32)


def action12(continuous: np.ndarray, discrete: np.ndarray | None = None) -> np.ndarray:
    """80-dim action rows -> [N, 12] float32 [base(3), torso, mode, eef_pos(3), eef_rot(3), gripper].

    Continuous channels are clipped to [-1, 1]; the gripper closes at closedness >= 0.5; mode > 0 selects the base;
    torso stays 0. `discrete` (default: `continuous`) supplies the gripper and mode rows.
    """
    a = np.asarray(continuous, dtype=np.float64)
    d = a if discrete is None else np.asarray(discrete, dtype=np.float64)
    out = np.zeros((a.shape[0], 12), dtype=np.float64)
    out[:, 5:8] = np.clip(a[:, 7:10], -1.0, 1.0)
    out[:, 8:11] = np.clip(a[:, 10:13], -1.0, 1.0)
    out[:, 0:2] = np.clip(a[:, 58:60], -1.0, 1.0)
    out[:, 2] = np.clip(a[:, 63], -1.0, 1.0)
    out[:, 11] = np.where(d[:, 16] >= 0.5, 1.0, -1.0)
    out[:, 4] = np.where(d[:, 76] > 0.0, 1.0, -1.0)
    return out.astype(np.float32)


def to_engine_order(actions12: np.ndarray) -> np.ndarray:
    """[N, 12] -> robocasa.utils.env_utils.convert_action order."""
    return np.ascontiguousarray(np.asarray(actions12)[:, list(ENGINE_ORDER)])
