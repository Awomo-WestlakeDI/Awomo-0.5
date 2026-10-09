# Copyright (c) 2026 Awomo-WAM Team. Licensed under the Apache License, Version 2.0.
"""Run one RoboCasa365 episode against a policy server."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "deploy") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "deploy"))
from client import Client

GROUPS = ("atomic_seen", "composite_seen", "composite_unseen")
CAMERAS = ("video.robot0_agentview_left", "video.robot0_agentview_right", "video.robot0_eye_in_hand")
STATE = ("state.base_position", "state.base_rotation", "state.end_effector_position_relative",
         "state.end_effector_rotation_relative", "state.gripper_qpos")
SEED = 7


def list_jobs(task_set: str, trials: int, tasks: list[str] | None = None) -> list[dict]:
    """Episode seed = 7 + 50 * task index within its group + trial."""
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

    jobs = []
    for group in GROUPS if task_set == "target50" else (task_set,):
        for index, task in enumerate(TASK_SET_REGISTRY[group]):
            if not tasks or task in tasks:
                jobs += [{"group": group, "task": task, "trial": t, "seed": SEED + 50 * index + t} for t in range(trials)]
    return jobs


def run_episode(client: Client, job: dict) -> dict:
    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action

    horizon = get_task_horizon(job["task"])
    env = gym.make(f"robocasa/{job['task']}", split="pretrain", seed=SEED)
    try:
        obs, _ = env.reset(seed=job["seed"])
        instruction = str(obs["annotation.human.task_description"])
        session = f"{job['task']}-{job['trial']}-{job['seed']}"

        def frames(o):
            return np.stack([np.array(o[k], dtype=np.uint8) for k in CAMERAS])

        pending, pending_steps, plan = [frames(obs)], [0], []
        steps = replan = 0
        success = False
        while steps < horizon:
            if not plan:
                state = np.concatenate([np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in STATE])
                plan = list(client({"session_id": session, "replan_index": replan, "seed": job["seed"] * 10000 + replan,
                                    "observation_steps": pending_steps, "camera_frames": np.stack(pending),
                                    "state": state, "instruction": instruction}))
                pending, pending_steps = [], []
                replan += 1
            obs, _, _, _, info = env.step(convert_action(plan.pop(0)))
            steps += 1
            if steps % 4 == 0:  # 5 Hz frames
                pending.append(frames(obs))
                pending_steps.append(steps)
            if info.get("success"):
                success = True
                break
    finally:
        env.close()
    return {**job, "success": success, "steps": steps}
