# Copyright (c) 2026 Awomo-WAM Team. Licensed under the Apache License, Version 2.0.
"""CPU checks for the exact released SFT math, sampler and dependency boundary."""

import hashlib
import io
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers import Flux2Transformer2DModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training/robodojo"))

from awomo.data.robodojo.dataset import (
    RoboDojoHDF5Dataset,
    convert_normalized_gripper_semantics,
    scatter_action_dims,
)
from awomo.data.robodojo.fixed_budget_dense import FixedBudgetDenseBatchSampler
from awomo.awomo_05i.model import ActionPatchModel
from awomo.awomo_05i.scheduler import FlowMatchScheduler
from awomo.awomo_05i.checkpoint import load_prepared_checkpoint, file_sha256, validate_checkpoint_payload
from awomo.training.ema import EMAManager
from awomo.training.frozen_state_ema import CompleteFrozenStateEMA
from awomo.training.gripper_loss import weighted_action_per_sample
from awomo.training.robodojo import RoboDojoTrainer
from awomo.training.trainer import Trainer
from awomo.data.robodojo.prompt import VLMPromptCodec, collate_vlm
from awomo.awomo_05i.sft_vlm import VLPromptEncoder

ACTIVE = [0, 1, 2, 3, 4, 5, 16, 29, 30, 31, 32, 33, 34, 45]


class SFTReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_native_import_locations(self):
        for name, module in list(sys.modules.items()):
            if name.startswith("awomo") and getattr(module, "__file__", None):
                self.assertTrue(Path(module.__file__).is_relative_to(ROOT), name)

    def test_tracking_hooks_without_optional_tracker(self):
        trainer = RoboDojoTrainer.__new__(RoboDojoTrainer)
        trainer.dist_env = SimpleNamespace(is_main=True)
        trainer.cfg = {}
        trainer.global_batch_size = 256
        trainer._setup_tracking()
        trainer._on_training_finish()
        log = {"train/mfu": float("nan")}
        trainer._extend_train_log_dict(log, 1., 1., 2., float("nan"))
        self.assertEqual(log, {"train/samples_per_second": 128.})
        self.assertIn("_build_models", RoboDojoTrainer.__dict__)

    @staticmethod
    def make_hdf5_fixture(root, name, value):
        import h5py
        import numpy as np
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle["instruction"] = "Move the object"
            handle["additional_info/frequency"] = 25
            for group, delta in (("action", 0), ("state", 100)):
                for field, width in (("left_arm_joint_states", 6), ("left_ee_joint_states", 1),
                                     ("right_arm_joint_states", 6), ("right_ee_joint_states", 1)):
                    handle[f"{group}/{field}"] = np.full((3, width), value + delta, dtype=np.float64)
            for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
                handle[f"vision/{camera}/colors"] = np.zeros((3, 1), dtype=np.uint8)
        return path

    def test_prepare_index_preserves_order_values_and_relative_paths(self):
        from prepare_data import build_index
        from awomo.data.robodojo.dataset import _load_index
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = ["b/arx_x5/data/episode_2.hdf5", "a/arx_x5/data/episode_1.hdf5"]
            for i, entry in enumerate(entries):
                self.make_hdf5_fixture(root, entry, i + 1)
            manifest = root / "episodes.txt"
            manifest.write_text("\n".join(entries) + "\n")
            output = root / "prepared/index.pt"
            summary = build_index(root, manifest, output)
            index = torch.load(output, weights_only=True)
            self.assertEqual([e["task"] for e in index["episodes"]], ["b", "a"])
            self.assertEqual([e["offset"] for e in index["episodes"]], [0, 3])
            self.assertEqual(summary["trajectory_shape"], [6, 14])
            self.assertEqual(summary["windows_stride1"], 4)
            self.assertTrue(all(not Path(e["path"]).is_absolute() for e in index["episodes"]))
            self.assertEqual(set(index["episodes"][0]), {"path", "task", "instruction", "n_frames", "offset"})
            torch.testing.assert_close(index["actions"][:3], torch.ones(3, 14), rtol=0, atol=0)
            torch.testing.assert_close(index["states"][3:], torch.full((3, 14), 102.), rtol=0, atol=0)
            with mock.patch("os.getcwd", return_value="/"):
                loaded = _load_index(str(output))
            self.assertEqual(loaded["episodes"][0]["path"], str(root / entries[0]))
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                build_index(root, manifest, output)
            self.assertEqual(output.read_bytes(), before)

    def test_prepare_index_rejects_invalid_manifest_and_raw_data(self):
        import h5py
        from prepare_data import build_index
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = "task/arx_x5/data/episode_0.hdf5"
            manifest = root / "episodes.txt"
            for content in ("", "../outside.hdf5", "/absolute/path.hdf5", f"{entry}\n{entry}"):
                manifest.write_text(content)
                with self.assertRaises(ValueError):
                    build_index(root, manifest, root / "index.pt")
            manifest.write_text(entry + "\n")
            cases = (
                ("additional_info/frequency", 30),
                ("instruction", ""),
                ("action/left_ee_joint_states", [[float("nan")]] * 3),
                ("state/right_arm_joint_states", [[0.] * 5] * 3),
                ("vision/cam_head/colors", [[0]]),
            )
            for key, value in cases:
                path = self.make_hdf5_fixture(root, entry, 1)
                with h5py.File(path, "a") as handle:
                    del handle[key]
                    handle[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    build_index(root, manifest, root / "index.pt")
                self.assertFalse((root / "index.pt").exists())

    def test_bundled_episode_manifest_is_relative_unique_and_ordered(self):
        entries = (ROOT / "training/robodojo/episodes.txt").read_text().splitlines()
        self.assertEqual(len(entries), 3200)
        self.assertEqual(len(set(entries)), 3200)
        self.assertEqual(len({Path(e).parts[0] for e in entries}), 32)
        self.assertTrue(all(not Path(e).is_absolute() and ".." not in Path(e).parts for e in entries))

    def test_source_revision_without_git_or_with_archive_revision(self):
        from awomo.training.builders import get_git_commit
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ", {}, clear=True):
            directory = Path(tmp)
            self.assertIsNone(get_git_commit(directory))
            with mock.patch.dict("os.environ", {"GIT_COMMIT": "working-tree-sha256:" + "a" * 64}):
                self.assertIsNone(get_git_commit(directory))
            with mock.patch.dict("os.environ", {"GIT_COMMIT": "b" * 40}):
                self.assertEqual(get_git_commit(directory), "b" * 40)
            (directory / ".git").mkdir()
            with mock.patch("subprocess.run", return_value=SimpleNamespace(stdout="c" * 40 + "\n")):
                self.assertEqual(get_git_commit(directory), "c" * 40)

    def test_bundled_normalization_loads_without_repository(self):
        from awomo.data.robodojo.normalizer import load_dataset_stats_from_json
        source = ROOT / "training/robodojo/dataset_stats.json"
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "stats.json"
            copy.write_bytes(source.read_bytes())
            stats = load_dataset_stats_from_json(str(copy))
            self.assertEqual(set(stats), {"action", "state"})
            self.assertTrue(all(torch.isfinite(v).all() for fields in stats.values()
                                for values in fields.values() for v in values.values()))

    def test_trainer_ownership_is_two_layers(self):
        self.assertEqual(RoboDojoTrainer.__bases__, (Trainer,))
        self.assertEqual(Trainer.__bases__, (object,))
        self.assertIs(RoboDojoTrainer.load_checkpoint, Trainer.load_checkpoint)
        self.assertEqual(RoboDojoTrainer._setup_fsdp.__module__, "awomo.training.distributed.fsdp")

    def test_prompt_batch_contains_only_conditioning(self):
        codec = VLMPromptCodec.__new__(VLMPromptCodec)
        codec.image_token_id = 99
        codec.pad_id = 0
        codec.merge_size = 2
        prefix = torch.tensor([1, 99, 2, 3])
        codec.prompt = lambda instruction, count: (prefix, torch.tensor([2]))
        codec.image_inputs = lambda image, history: (
            torch.zeros(4, 3), torch.tensor([[1, 2, 2]]),
            torch.tensor([[1, 2, 2]]), torch.tensor([False]))
        item = codec.encode_sample("Move the object", object())
        torch.testing.assert_close(item["input_ids"], prefix, rtol=0, atol=0)
        expected = {"input_ids", "prompt_len", "instr_pos", "pixel_values", "image_grid_thw",
            "image_grid_thw_llm", "image_is_hist", "n_images", "mm_token_type_ids", "pad_id"}
        self.assertEqual(set(item), expected)
        batch = collate_vlm([item, item])
        self.assertNotIn("labels", batch)
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 4))

    def test_vlm_conditioning_skips_vocabulary_projection(self):
        encoder = VLPromptEncoder.__new__(VLPromptEncoder)
        torch.nn.Module.__init__(encoder)
        encoder.qwen_layers = (0, 1)
        encoder.summary_layers = (1, 2)
        encoder.torch_dtype = torch.float32
        hidden = tuple(torch.arange(40, dtype=torch.float32).reshape(2, 5, 4) + i for i in range(3))
        encoder._run = lambda batch: hidden
        batch = {"instr_pos": torch.tensor([[1, 3], [2, -1]]), "prompt_len": torch.tensor([5, 4])}
        with mock.patch("torch.nn.functional.cross_entropy", side_effect=AssertionError("Unexpected text loss")):
            text, mask = encoder.forward_vlm(batch)
            inferred, inferred_mask = encoder.infer_vlm(batch)
        self.assertEqual(tuple(text.shape), (2, 3, 8))
        self.assertEqual(mask.tolist(), [[True, True, True], [True, False, True]])
        torch.testing.assert_close(text, inferred, rtol=0, atol=0)
        torch.testing.assert_close(mask, inferred_mask, rtol=0, atol=0)

    def test_dataset_needs_no_auxiliary_files_or_source_tag(self):
        import h5py
        import numpy as np
        from PIL import Image
        class Codec:
            def __init__(self, *args, **kwargs): pass
            def encode_sample(self, instruction, image, history):
                return {"input_ids": torch.tensor([1, 2]), "instruction": instruction}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            episode = directory / "episode.hdf5"
            buffer = io.BytesIO()
            Image.new("RGB", (32, 32), (10, 20, 30)).save(buffer, format="JPEG")
            raw = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
            with h5py.File(episode, "w") as handle:
                for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
                    images = handle.create_dataset(f"vision/{camera}/colors", (3,), dtype=h5py.vlen_dtype(np.uint8))
                    for i in range(3): images[i] = raw
            trajectory = torch.zeros(3, 14)
            torch.save({"episodes": [{"path": str(episode), "task": "fixture", "instruction": "Move the object",
                "n_frames": 3, "offset": 0, "source": "arbitrary"}], "actions": trajectory,
                "states": trajectory}, directory / "index.pt")
            stats = {k: {"default": {"global_mean": [0.] * 14, "global_std": [1.] * 14}}
                     for k in ("action", "state")}
            (directory / "stats.json").write_text(json.dumps(stats))
            with mock.patch("transformers.AutoProcessor.from_pretrained", return_value=object()), \
                 mock.patch("awomo.data.robodojo.prompt.VLMPromptCodec", Codec), \
                 mock.patch("glob.glob", side_effect=AssertionError("Unexpected annotation scan")):
                dataset = RoboDojoHDF5Dataset(str(directory / "index.pt"), str(directory / "stats.json"),
                    num_frames=2, video_size=(16, 16), val_set_proportion=0,
                    vlm={"enabled": True, "model_path": "unused", "image_size": [32, 32],
                         "history": {"enabled": False}})
                item = dataset[0]
                self.assertEqual(len(dataset), 2)
                self.assertNotIn("source", item)
                self.assertEqual(item["vlm"]["instruction"], "Move the object")

    def test_gripper_gradient_multiplier_and_denominator(self):
        x = torch.ones(2, 16, 80, requires_grad=True)
        dims = torch.zeros(2, 80)
        dims[:, ACTIVE] = 1
        valid_steps = torch.ones(2, 16)
        first = weighted_action_per_sample(x, dims, valid_steps, 1.0)
        second = weighted_action_per_sample(x, dims, valid_steps, 2.0)
        torch.testing.assert_close(first, torch.ones(2))
        torch.testing.assert_close(second, torch.full((2,), 16 / 14))
        grad1 = torch.autograd.grad(first.sum(), x, retain_graph=True)[0]
        grad2 = torch.autograd.grad(second.sum(), x)[0]
        expected = grad1.clone()
        expected[:, :, [16, 45]] *= 2
        torch.testing.assert_close(grad2, expected, rtol=0, atol=0)

    def test_dense_checkpoint_target_modules_without_expert_dependencies(self):
        from awomo.training.checkpoint.addons import _extract_target_modules
        dense = torch.nn.Module()
        dense.proj = torch.nn.Module()
        dense.proj.lora_A = torch.nn.Linear(3, 2, bias=False)
        self.assertEqual(_extract_target_modules(dense), ["proj"])
        self.assertFalse(any(name.startswith("awomo.training.moe") for name in sys.modules))
        dense.state_dict_adapter = object()
        with self.assertRaisesRegex(NotImplementedError, "dense FLUX"):
            _extract_target_modules(dense)

    def test_prepared_base_direct_load_is_exact_and_idempotent(self):
        model = torch.nn.Linear(3, 2)
        parameter = torch.nn.Parameter(torch.randn(2, 3))
        encoder = SimpleNamespace(text_layout="prefix_v2", named_parameters=lambda: [
            ("model.language_model.layers.0.q_proj.lora_A.default.weight", parameter)])
        stored = {name: value.detach().clone() for name, value in model.state_dict().items()}
        adapter = parameter.detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            torch.save({"dit": stored, "dit_config": {"in_features":3,"out_features":2},
                "qwen_lora": {"layers.0.q_proj.lora_A": adapter}}, directory / "model.pt")
            config = {"model_type":"awomo-0.5i-robodojo-sft", "format_version":1,
                "artifact_kind":"base", "parameter_dtype":"float32", "vl_text_layout":"prefix_v2"}
            (directory / "awomo_config.json").write_text(json.dumps(config))
            (directory / "dataset_stats.json").write_text("{}")
            names = ["model.pt", "awomo_config.json", "dataset_stats.json"]
            (directory / "SHA256SUMS").write_text("".join(f"{file_sha256(directory / n)}  {n}\n" for n in names))
            with torch.no_grad():
                model.weight.zero_()
                parameter.zero_()
            for _ in range(2):
                load_prepared_checkpoint(model, encoder, directory)
                for name, actual in model.state_dict().items():
                    torch.testing.assert_close(actual, stored[name], rtol=0, atol=0)
                torch.testing.assert_close(parameter, adapter, rtol=0, atol=0)
            encoder.text_layout = "incompatible"
            with self.assertRaisesRegex(ValueError, "text-layout mismatch"):
                load_prepared_checkpoint(model, encoder, directory)
            encoder.text_layout = "prefix_v2"
            (directory / "dataset_stats.json").write_text('{"tampered":true}')
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_prepared_checkpoint(model, encoder, directory)

    def test_public_checkpoint_schema_is_exact(self):
        payload = {"dit":{"transformer.weight":torch.zeros(2,2)},
            "dit_config":{"dim":2},"qwen_lora":{"adapter":torch.zeros(2,2)}}
        validate_checkpoint_payload(payload)
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_checkpoint_payload({**payload,"vl_text_layout":"prefix_v2"})
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_checkpoint_payload({k:v for k,v in payload.items() if k != "dit_config"})

    def test_complete_32_rank_sampling_sequence(self):
        expected = torch.randperm(1724338, generator=torch.Generator().manual_seed(42))
        expected = torch.cat([expected, expected[:4942]]).reshape(6755, 32, 8)
        for rank in range(32):
            sampler = FixedBudgetDenseBatchSampler(1724338, 42, 8, rank, 32, 6755)
            actual = torch.tensor(list(sampler))
            torch.testing.assert_close(actual, expected[:, rank], rtol=0, atol=0)
            sampler.set_data_step(6754)
            self.assertEqual(list(sampler), expected[6754:, rank].tolist())
        self.assertEqual(expected.numel(), 1729280)

    def test_gripper_and_canonical80_round_trip(self):
        compact = torch.randn(2, 16, 14)
        closed = convert_normalized_gripper_semantics(compact, gripper_indices=[6, 13],
                                                      source_semantics="open_is_one", target_semantics="closed_is_one")
        dense, mask = scatter_action_dims(closed, output_dim=80, output_indices=ACTIVE)
        self.assertEqual(int(mask.sum()), 66)
        self.assertEqual(int(torch.count_nonzero(dense[..., mask])), 0)
        restored = convert_normalized_gripper_semantics(dense[..., ACTIVE], gripper_indices=[6, 13],
                                                        source_semantics="closed_is_one", target_semantics="open_is_one")
        torch.testing.assert_close(restored, compact, rtol=0, atol=0)

    def test_separate_and_packed_conditions_are_equivalent(self):
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(91)
            transformer = Flux2Transformer2DModel(in_channels=128, out_channels=128, num_layers=1,
                num_single_layers=1, attention_head_dim=8, num_attention_heads=1,
                joint_attention_dim=24, axes_dims_rope=(2, 2, 2, 2)).to(dtype)
            model = ActionPatchModel(transformer, 80, 80, 1.0).eval()
            self.assertFalse(hasattr(model, "conditioning_profile"))
            reference = torch.randn(2, 4, 128, dtype=dtype)
            history = torch.randn(2, 4, 128, dtype=dtype)
            history_ids = model.image_ids(2, 2, -1, torch.device("cpu"))
            history_valid = torch.tensor([[True,False,True,False],[True,True,True,True]])
            shared = dict(noisy_image=torch.randn(2,4,128,dtype=dtype),
                noisy_action=torch.randn(2,16,80,dtype=dtype), context=torch.randn(2,3,24,dtype=dtype),
                text_mask=torch.tensor([[True,True,False],[True,True,True]]),
                state=torch.randn(2,80,dtype=dtype), timestep=torch.tensor([300.,800.],dtype=dtype))
            with torch.no_grad():
                separate = model(reference, **shared, image_grid=(2,2), history=history,
                    history_ids=history_ids, history_valid=history_valid)
                packed = model(torch.cat([history,reference],1), **shared,
                    reference_ids=torch.cat([history_ids,model.image_ids(2,2,10,torch.device("cpu"))]),
                    target_ids=model.image_ids(2,2,0,torch.device("cpu")),
                    condition_mask=torch.cat([history_valid,torch.ones(2,4,dtype=torch.bool)],1))
            for left,right in zip(separate,packed):
                torch.testing.assert_close(left,right,rtol=0,atol=0)

    def test_partial_or_mixed_condition_inputs_are_rejected(self):
        transformer = Flux2Transformer2DModel(in_channels=128, out_channels=128, num_layers=1,
            num_single_layers=1, attention_head_dim=8, num_attention_heads=1,
            joint_attention_dim=24, axes_dims_rope=(2,2,2,2))
        model = ActionPatchModel(transformer,80,80,1.0)
        common = dict(reference=torch.zeros(1,4,128), noisy_image=torch.zeros(1,4,128),
            noisy_action=torch.zeros(1,16,80),context=torch.zeros(1,3,24),
            text_mask=torch.ones(1,3,dtype=torch.bool),state=torch.zeros(1,80),timestep=torch.tensor([500.]))
        ids = torch.zeros(4,4)
        with self.assertRaisesRegex(ValueError,"together"):
            model(**common,reference_ids=ids,target_ids=ids)
        with self.assertRaisesRegex(ValueError,"Do not mix"):
            model(**common,reference_ids=ids,target_ids=ids,condition_mask=torch.ones(1,4,dtype=torch.bool),image_grid=(2,2))
        with self.assertRaisesRegex(ValueError,"Separate conditions require"):
            model(**common)

    def test_tiny_transformer_update_full_ema_save_reload(self):
        torch.manual_seed(13)
        transformer = Flux2Transformer2DModel(in_channels=128, out_channels=128, num_layers=1,
                                             num_single_layers=1, attention_head_dim=8, num_attention_heads=1,
                                             joint_attention_dim=24, axes_dims_rope=(2, 2, 2, 2))
        model = ActionPatchModel(transformer, action_dim=80, state_dim=80, action_scale=1.0)
        model.proprio_encoder.requires_grad_(False)
        model.train()
        frozen = {name: value.detach().clone() for name, value in model.proprio_encoder.state_dict().items()}
        ema = CompleteFrozenStateEMA(EMAManager(model, decay=0.999), model)
        scheduler = FlowMatchScheduler(num_train_timesteps=1000, shift=5)
        scheduler.set_timesteps(1000, training=True)
        dims = torch.ones(2, 80, dtype=torch.bool)
        dims[:, ACTIVE] = False
        batch = {"action_dim_is_pad": dims, "action_is_pad": torch.zeros(2, 16, dtype=torch.bool)}
        encoded = {"reference": torch.randn(2, 4, 128), "target": torch.randn(2, 4, 128),
                   "action": torch.randn(2, 16, 80), "context": torch.randn(2, 3, 24),
                   "text_mask": torch.ones(2, 3, dtype=torch.bool), "state": torch.randn(2, 80),
                   "reference_ids": model.image_ids(2, 2, 10, torch.device("cpu")),
                   "target_ids": model.image_ids(2, 2, 0, torch.device("cpu")),
                   "condition_mask": torch.ones(2, 4, dtype=torch.bool)}
        encoded["action"][:, :, dims[0]] = 0
        padded = model.encode_action(encoded["action"])
        self.assertEqual(int(torch.count_nonzero(padded[..., 80:])), 0)
        fixture = SimpleNamespace(_encode_batch=lambda _: encoded, scheduler=scheduler, device=torch.device("cpu"),
                                  model=model, cfg=SimpleNamespace(gripper_experiment=SimpleNamespace(loss_multiplier=2.)),
                                  _model_cfg=SimpleNamespace(bsmntw_enabled=True),
                                  loss_lambda_video=0.5, loss_lambda_action=1.)
        losses = RoboDojoTrainer.train_step(fixture, batch, 0)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        torch.testing.assert_close(losses["total_loss"], losses["dynamics_loss"] + losses["action_loss"], rtol=0, atol=0)
        before = model.transformer.proj_out.weight.detach().clone()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-6,
                                      weight_decay=0.01, betas=(0.9, 0.95))
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        ema.update(model)
        self.assertFalse(torch.equal(before, model.transformer.proj_out.weight))
        for name, value in frozen.items():
            torch.testing.assert_close(model.proprio_encoder.state_dict()[name], value, rtol=0, atol=0)
            torch.testing.assert_close(ema.state_dict()["proprio_encoder." + name], value, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "checkpoint.pt"
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict()}, file)
            restored = torch.load(file, weights_only=True)
            model.load_state_dict(restored["model"], strict=True)
            ema.load_state_dict(restored["ema"], strict=True)
            self.assertEqual(set(restored["model"]), set(restored["ema"]))
            self.assertIsNone(model.transformer.proj_out.bias)


class BaseCheckpointTests(unittest.TestCase):
    def test_two_file_base_loads_exact_weights_without_metadata_or_statistics(self):
        from awomo.awomo_05i.checkpoint import read_checkpoint_config
        model = torch.nn.Linear(3, 2)
        adapter = torch.nn.Parameter(torch.zeros(2, 3))
        encoder = SimpleNamespace(text_layout="prefix_v2", named_parameters=lambda: [
            ("model.language_model.layers.0.q_proj.lora_A.default.weight", adapter)])
        stored = {name: value.detach().clone() for name, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.save({"dit": stored, "dit_config": {"in_features": 3, "out_features": 2},
                        "qwen_lora": {"layers.0.q_proj.lora_A": torch.ones(2, 3)}}, root / "model.pt")
            (root / "SHA256SUMS").write_text(f"{file_sha256(root / 'model.pt')}  model.pt\n")
            self.assertEqual(read_checkpoint_config(root, allow_base_only=True)["artifact_kind"], "base")
            with self.assertRaises(ValueError):
                read_checkpoint_config(root)  # Inference must not silently accept missing metadata.
            with torch.no_grad():
                model.weight.zero_()
            load_prepared_checkpoint(model, encoder, root, allow_base_only=True)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, stored[name], rtol=0, atol=0)
            torch.testing.assert_close(adapter, torch.ones(2, 3), rtol=0, atol=0)

    def test_two_file_base_requires_the_correct_model_only_checksum(self):
        from awomo.awomo_05i.checkpoint import read_checkpoint_config
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.pt").write_bytes(b"fixture")
            with self.assertRaises(FileNotFoundError):
                read_checkpoint_config(root, allow_base_only=True)
            (root / "SHA256SUMS").write_text("0" * 64 + "  model.pt\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                read_checkpoint_config(root, allow_base_only=True)
            (root / "other.json").write_text("{}")
            (root / "SHA256SUMS").write_text(f"{file_sha256(root / 'model.pt')}  model.pt\n"
                f"{file_sha256(root / 'other.json')}  other.json\n")
            with self.assertRaisesRegex(ValueError, "exactly model.pt"):
                read_checkpoint_config(root, allow_base_only=True)


if __name__ == "__main__":
    unittest.main()
