import json
import socket
import struct
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "deploy")]

from client import Client, send
from server import serve

from eval_robocasa365.awomo_05i import policy as rc

sys.path.insert(0, str(ROOT / "eval_robocasa365"))
import dynamic_eval  # noqa: E402


def test_canvas_layout():
    frames = np.zeros((1, 3, 256, 256, 3), dtype=np.uint8)
    frames[:, 0], frames[:, 1], frames[:, 2] = 10, 20, 30
    canvas = rc.compose_canvas(frames, (384, 256))
    np.testing.assert_allclose(canvas[0, :256].mean(), 30 / 255, atol=1e-6)
    np.testing.assert_allclose(canvas[0, 256:, :128].mean(), 10 / 255, atol=1e-6)


def test_state_and_actions():
    state = np.zeros(16, dtype=np.float32)
    state[3:7] = state[10:14] = [0, 0, 0, 1]
    state[14:16] = [0.04, -0.04]
    s = rc.state_vector(state)[0]
    np.testing.assert_allclose(s[10:16], [1, 0, 0, 0, 1, 0])
    assert abs(s[16]) < 1e-6
    raw = np.random.default_rng(0).normal(size=(16, 80)).astype(np.float32)
    a = rc.action_rows(raw)
    assert a.shape == (16, 12) and np.abs(a).max() <= 1.0
    assert set(np.unique(a[:, 6])) <= {-1.0, 1.0} and set(np.unique(a[:, 11])) <= {-1.0, 1.0}
    assert (a[:, 10] == 0).all()  # torso


class FakeModel:
    history_slots, history_period, video_size = 9, 40, (320, 224)
    action_horizon, action_dim, device = 16, 80, "cpu"
    config = {"scheduler": {"num_inference_steps": 16, "shift": 5.0, "timestep_scale": 1000}}
    processor = None

    def __init__(self):
        self.samples = []

    def encode(self, sample):
        self.samples.append(sample)
        return {"target": torch.zeros(1, 4, 8)}

    def velocity(self, encoded, noisy_image, noisy_action, timestep):
        return torch.zeros_like(noisy_image), torch.zeros_like(noisy_action)


@pytest.fixture()
def policy(monkeypatch):
    import awomo.awomo_05i.prompt as prompt

    monkeypatch.setattr(prompt, "PromptEncoder", lambda *a: types.SimpleNamespace(encode=lambda *b: {}))
    return rc.RoboCasaPolicy(FakeModel(), compress_inputs=False)


def request(replan, steps, session="ep"):
    state = np.zeros(16, dtype=np.float32)
    state[3:7] = state[10:14] = [0, 0, 0, 1]
    return {"session_id": session, "replan_index": replan, "seed": 0, "observation_steps": steps, "state": state,
            "camera_frames": np.zeros((len(steps), 3, 256, 256, 3), dtype=np.uint8), "instruction": "open the drawer"}


def test_history_and_timeline(policy):
    policy.act(request(0, [0]))
    for k in range(1, 24):
        s = 16 * k
        policy.act(request(k, [s - 12, s - 8, s - 4, s]))
    assert policy.model.samples[-1]["history_valid"].all()
    with pytest.raises(ValueError):
        policy.act(request(24, [380, 384, 388, 392]))


def test_server_round_trip():
    class Echo:
        def __init__(self):
            self.metadata = {"model": "fake"}

        def act(self, message):
            if message["seed"] < 0:
                raise ValueError("bad seed")
            return np.full((16, 12), message["seed"], dtype=np.float32), {}

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    threading.Thread(target=serve, args=(Echo(), "127.0.0.1", port), daemon=True).start()
    client = Client(port=port, wait_seconds=10)
    actions = client({"seed": 3})
    assert actions.shape == (16, 12) and actions[0, 0] == 3
    with pytest.raises(RuntimeError):
        client({"seed": -1})
    client.close()


def test_server_survives_a_client_reset():
    class Slow:
        metadata = {"model": "fake"}

        def act(self, message):
            time.sleep(0.3)  # the client resets the connection before the reply is sent
            return np.zeros((16, 12), dtype=np.float32), {}

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    threading.Thread(target=serve, args=(Slow(), "127.0.0.1", port), daemon=True).start()
    first = Client(port=port, wait_seconds=10)
    send(first.sock, {"seed": 1})
    first.sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    first.close()  # RST while the server is still computing
    time.sleep(0.6)
    second = Client(port=port, wait_seconds=5)
    assert second({"seed": 2}).shape == (16, 12)
    second.close()


def test_output_dir_belongs_to_one_server(tmp_path, monkeypatch):
    metadata = {"model": "awomo-0.5i", "benchmark": "robocasa365", "checkpoint": "a"}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.metadata = dict(metadata)

        def close(self):
            pass

    monkeypatch.setattr(dynamic_eval.entry, "Client", FakeClient)
    for name in ("pending", "running", "results", "errors"):
        (tmp_path / name).mkdir()
    dynamic_eval.worker(tmp_path, port=0)  # first server: records its identity
    assert json.loads((tmp_path / "identity.json").read_text())["checkpoint"] == "a"
    dynamic_eval.worker(tmp_path, port=0)  # same server again: resumes
    dynamic_eval.write_json(tmp_path / "config.json", {"task_set": "atomic_seen", "trials": 1})
    dynamic_eval.merge(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["identity"]["checkpoint"] == "a" and summary["config"]["task_set"] == "atomic_seen"
    assert not list(tmp_path.rglob("*.tmp"))  # atomic writes leave no temporary files
    metadata.update(model="awomo-0.5v", checkpoint="b")
    with pytest.raises(SystemExit):
        dynamic_eval.worker(tmp_path, port=0)  # another model must not reuse these results
    other = tmp_path / "other"
    (other / "results").mkdir(parents=True)
    (other / "pending").mkdir()
    (other / "results" / "atomic_seen__CloseFridge__000.json").write_text("{}")
    with pytest.raises(SystemExit):
        dynamic_eval.worker(other, port=0)  # results of unknown origin


def test_reconnect_to_another_checkpoint_is_refused(tmp_path, monkeypatch):
    checkpoints = iter(["a", "b"])  # the server comes back with another checkpoint after a disconnect

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.metadata = {"model": "awomo-0.5v", "benchmark": "robocasa365", "checkpoint": next(checkpoints)}

        def close(self):
            pass

    def disconnected(client, job):
        raise ConnectionError("connection closed")

    monkeypatch.setattr(dynamic_eval.entry, "Client", FakeClient)
    monkeypatch.setattr(dynamic_eval.entry, "run_episode", disconnected)
    for name in ("pending", "running", "results", "errors"):
        (tmp_path / name).mkdir()
    job = {"group": "atomic_seen", "task": "CloseFridge", "trial": 0, "seed": 57}
    (tmp_path / "pending" / "atomic_seen__CloseFridge__000.json").write_text(json.dumps(job))
    with pytest.raises(SystemExit):
        dynamic_eval.worker(tmp_path, port=0)
    assert not list((tmp_path / "results").iterdir())  # nothing recorded under the new checkpoint
    assert (tmp_path / "pending" / "atomic_seen__CloseFridge__000.json").exists()  # the episode stays queued


def test_changed_evaluation_config_is_refused(tmp_path, monkeypatch):
    def jobs(task_set, trials, tasks=None):
        return [{"group": "atomic_seen", "task": "CloseFridge", "trial": t, "seed": 7 + t} for t in range(trials)]

    monkeypatch.setattr(dynamic_eval.entry, "list_jobs", jobs)
    dynamic_eval.init(tmp_path, "atomic_seen", 30, None)
    dynamic_eval.init(tmp_path, "atomic_seen", 30, None)  # same config: resume
    with pytest.raises(SystemExit):
        dynamic_eval.init(tmp_path, "atomic_seen", 1, None)  # fewer trials must not reuse the 30-trial results
    assert json.loads((tmp_path / "config.json").read_text())["trials"] == 30
    legacy = tmp_path / "legacy"
    (legacy / "results").mkdir(parents=True)
    (legacy / "results" / "atomic_seen__CloseFridge__000.json").write_text("{}")
    with pytest.raises(SystemExit):
        dynamic_eval.init(legacy, "atomic_seen", 30, None)  # results without a recorded config


def test_failing_episode_is_retried_then_recorded(tmp_path, monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.metadata = {"model": "fake"}

        def close(self):
            pass

    calls = []

    def failing_episode(client, job):
        calls.append(job["seed"])
        raise RuntimeError("server error")

    monkeypatch.setattr(dynamic_eval.entry, "Client", FakeClient)
    monkeypatch.setattr(dynamic_eval.entry, "run_episode", failing_episode)
    for name in ("pending", "running", "results", "errors"):
        (tmp_path / name).mkdir()
    job = {"group": "atomic_seen", "task": "CloseFridge", "trial": 0, "seed": 57}
    (tmp_path / "pending" / "atomic_seen__CloseFridge__000.json").write_text(json.dumps(job))
    dynamic_eval.worker(tmp_path, port=0)
    assert len(calls) == dynamic_eval.MAX_ATTEMPTS
    assert not list((tmp_path / "pending").iterdir()) and not list((tmp_path / "running").iterdir())
    recorded = json.loads((tmp_path / "errors" / "atomic_seen__CloseFridge__000.json").read_text())
    assert recorded["attempts"] == dynamic_eval.MAX_ATTEMPTS and "server error" in recorded["error"]
    assert dynamic_eval.merge(tmp_path) == 1  # nonzero exit: an episode failed
    assert json.loads((tmp_path / "summary.json").read_text())["errors"] == ["atomic_seen__CloseFridge__000"]


def test_both_adapters_share_the_robocasa_interface():
    from eval_robocasa365 import common
    from eval_robocasa365.awomo_05v import policy as rc05v

    for name in ("state_vector", "ACTION_MASK", "ENGINE_ORDER", "REPLAN_STEPS"):
        assert getattr(rc, name) is getattr(common, name) and getattr(rc05v, name) is getattr(common, name)
    with pytest.raises(ValueError):
        common.check_timeline({"replan_index": 1, "session_id": "ep"}, [4, 8, 12, 16], None)
