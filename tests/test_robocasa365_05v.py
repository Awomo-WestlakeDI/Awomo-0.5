"""CPU tests for the Awomo-0.5V RoboCasa365 adapter (no checkpoint or GPU needed)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_robocasa365.awomo_05v import policy as rc


def solid(value):
    return np.full((256, 256, 3), value, dtype=np.uint8)


def test_canvas_layout():
    canvas = rc.compose_canvas(solid(10), solid(20), solid(30))
    assert canvas.shape == (384, 320, 3) and canvas.dtype == np.uint8
    assert (canvas[:256] == 30).all()  # wrist on top
    assert (canvas[256:, :160] == 10).all() and (canvas[256:, 160:] == 20).all()


def test_prompt():
    prompt = rc.build_prompt("open the drawer", 70000)
    assert prompt == rc.build_prompt("open the drawer", 70000)
    assert prompt in {t.format(instruction="open the drawer") for t in rc.PROMPT_TEMPLATES}
    assert all(rc.build_prompt("open the drawer.", seed).endswith("open the drawer.") and
               not rc.build_prompt("open the drawer.", seed).endswith("..") for seed in range(50))
    assert len(rc.PROMPT_TEMPLATES) == len(set(rc.PROMPT_TEMPLATES)) == 100
    assert len({rc.build_prompt("open the drawer", seed) for seed in range(200)}) > 50


def test_state_vector():
    state = np.zeros(16, dtype=np.float32)
    yaw = 0.3
    state[0:2] = [1.0, 2.0]
    state[3:7] = [0, 0, np.sin(yaw / 2), np.cos(yaw / 2)]
    state[7:10] = [0.1, 0.2, 0.3]
    state[10:14] = [0, 0, 0, 1]
    state[14:16] = [0.04, -0.04]
    s = rc.state_vector(state)[0]
    np.testing.assert_allclose(s[7:10], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(s[10:16], [1, 0, 0, 0, 1, 0], atol=1e-6)
    assert abs(s[16]) < 1e-6  # fully open gripper
    np.testing.assert_allclose(s[[58, 59, 63]], [1.0, 2.0, yaw], atol=1e-6)
    assert (s[~rc.STATE_MASK] == 0).all()


def test_action_resampling():
    k0, k1, mu, _ = rc._resample_table()
    position = k0 + mu  # 15 Hz index of each 20 Hz step
    np.testing.assert_allclose(position[1:], 3 * (np.arange(1, 32) + 1) / 4 - 1)
    assert position[0] == 0 and k1.max() == rc.ACTIONS_PER_CHUNK - 1

    chunk = np.zeros((rc.ACTIONS_PER_CHUNK, rc.CANONICAL_DIM), dtype=np.float32)
    chunk[:, 7] = 0.01 * np.arange(rc.ACTIONS_PER_CHUNK)  # eef x ramps linearly
    chunk[:, 16] = np.arange(rc.ACTIONS_PER_CHUNK) >= 12  # gripper closes at sample 12
    chunk[:, 76] = 1.0
    native = rc.action_rows(chunk)
    assert native.shape == (32, 12)
    np.testing.assert_allclose(native[:, 5], 0.01 * position, atol=1e-6)
    assert set(np.unique(native[:, 11])) == {-1.0, 1.0} and (native[:, 4] == 1).all() and (native[:, 3] == 0).all()

    chunk[0, 20] = 1e-3  # an inactive slot must end at exactly zero
    with pytest.raises(ValueError):
        rc.action_rows(chunk)


def test_dit_step_mask():
    assert len(rc.DIT_STEP_MASK) == 16 and sum(rc.DIT_STEP_MASK) == 8 and rc.DIT_STEP_MASK[0]


class FakeModel:
    device = torch.device("cpu")
    action_dim, num_frame_per_block, video_size = 80, 2, (384, 320)
    config = {"scheduler": {"num_train_timesteps": 1000, "shift": 5.0, "num_inference_steps": 16}}

    def __init__(self):
        self.windows, self.prompts, self.calls = [], [], []

    def encode_text(self, prompt):
        self.prompts.append(prompt)
        return torch.zeros(1, 4, 8)

    def encode_video(self, video):
        self.windows.append(video.shape[0])
        return torch.zeros(1, 48, (video.shape[0] - 1) // 4 + 1, 4, 4)

    def warm_up(self, latents, prompt_embs):
        return []

    def velocity(self, x_video, x_action, video_t, action_t, state, prompt_embs, kv_cache, start_frame):
        self.calls.append(start_frame)
        return torch.zeros_like(x_video), torch.ones_like(x_action)


def request(replan, steps, session="ep"):
    state = np.zeros(16, dtype=np.float32)
    state[3:7] = state[10:14] = [0, 0, 0, 1]
    return {"session_id": session, "replan_index": replan, "seed": 70000 + replan, "observation_steps": steps,
            "state": state, "camera_frames": np.zeros((len(steps), 3, 256, 256, 3), dtype=np.uint8),
            "instruction": "open the drawer"}


def test_episode_history_and_outputs():
    model = FakeModel()
    policy = rc.RoboCasaPolicy(model, compress_inputs=False)
    result = policy.predict(request(0, [0]))
    assert result["actions"].shape == (16, 12) and result["raw"].shape == (24, 80)
    np.testing.assert_array_equal(result["actions"], result["native"][:16][:, list(rc.ENGINE_ORDER)])
    assert (result["raw"][:, ~rc.ACTION_MASK] == 0).all()
    assert len(model.calls) == sum(rc.DIT_STEP_MASK)
    for k in range(1, 10):
        policy.predict(request(k, [16 * k - 12, 16 * k - 8, 16 * k - 4, 16 * k]))
    # 1 + 4k frames until the window holds the latest 25
    assert model.windows == [1, 5, 9, 13, 17, 21, 25, 25, 25, 25]
    assert len(policy.episode["canvases"]) == rc.HISTORY_BUFFER_FRAMES
    assert model.prompts[0] == rc.build_prompt("open the drawer", 70000)


def test_request_checks():
    policy = rc.RoboCasaPolicy(FakeModel(), compress_inputs=False)
    with pytest.raises(ValueError):
        policy.predict(request(0, [0, 4]))  # replan 0 sends only the step-0 frame
    policy.predict(request(0, [0]))
    with pytest.raises(ValueError):
        policy.predict(request(1, [4, 8, 12]))  # missing frame
    with pytest.raises(ValueError):
        policy.predict(request(1, [4, 8, 12, 16], session="other"))


def test_checkpoint_step_count_must_match_mask():
    model = FakeModel()
    model.config = {"scheduler": {**FakeModel.config["scheduler"], "num_inference_steps": 20}}
    with pytest.raises(ValueError):
        rc.RoboCasaPolicy(model, compress_inputs=False)


def test_h264_roundtrip():
    try:
        executable = rc.ffmpeg_executable()
    except RuntimeError:
        pytest.skip("ffmpeg not available")
    ramp = np.tile(np.linspace(0, 255, 256, dtype=np.uint8)[None, :, None], (256, 1, 3))
    frames = np.stack([ramp, ramp[::-1], ramp, ramp[::-1]])
    out = rc.h264_roundtrip(frames, executable)
    assert out.shape == frames.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - frames.astype(int)).mean() < 3
