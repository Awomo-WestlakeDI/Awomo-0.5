# Copyright (c) 2026 Awomo-WAM Team. Apache-2.0.
"""Worker random-state and prefetch checkpoint tests."""

import copy
import io
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from awomo.data.robodojo.augmentation import VideoAugmentation
from awomo.data.robodojo.dataset import DenseHistorySampler, VLMHistorySampler, RoboDojoHDF5Dataset
from awomo.data.robodojo.loader import DenseBatchSampler, RoboDojoDataLoader, seed_worker
from awomo.training.robodojo import RoboDojoTrainer
from awomo.training.trainer import Trainer


class RandomWindowDataset(RoboDojoHDF5Dataset):
    """Exercise the same augmentation and history RNGs without HDF5 I/O."""

    def __init__(self):
        self.frames = torch.linspace(0, 1, 2 * 3 * 8 * 8).reshape(1, 2, 3, 8, 8)
        self.augment = VideoAugmentation(augment_types=("color_only",),
                                         color_jitter={"brightness": 0.2, "contrast": 0.2})
        self.history = DenseHistorySampler([(0.1, 0.5), (0.5, 1.0), (1, 2)])
        self.vlm_history = VLMHistorySampler(num_slots=3)

    def __len__(self):
        return 192

    def __getitem__(self, index):
        values = torch.tensor([index, random.random(), np.random.rand(), torch.rand(()).item()], dtype=torch.float64)
        history = torch.tensor([v if v is not None else -1 for v in self.history.sample(200)])
        vlm = torch.tensor(self.vlm_history.sample(200))
        return {"values": values, "image": self.augment(self.frames), "history": history, "vlm": vlm}


def make_loader(workers, persistent=True, stateful=True, rank=0, world_size=1):
    sampler = DenseBatchSampler(192, 42, 2, rank, world_size)
    kwargs = dict(dataset=RandomWindowDataset(), batch_sampler=sampler, num_workers=workers,
                  worker_init_fn=seed_worker)
    if workers:
        kwargs.update(prefetch_factor=2, persistent_workers=persistent)
    loader = RoboDojoDataLoader if stateful else DataLoader
    return loader(**kwargs)


def close(iterator):
    if hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()


def reset_rng():
    random.seed(71)
    np.random.seed(71)
    torch.manual_seed(71)


def collect(iterator, count):
    # Parent draws model noise after each delivered batch.
    return [(next(iterator), torch.randn(5)) for _ in range(count)]


class DataResumeTests(unittest.TestCase):
    def assert_batches_equal(self, left, right):
        self.assertEqual(len(left), len(right))
        for (a, noise_a), (b, noise_b) in zip(left, right):
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            torch.testing.assert_close(noise_a, noise_b, rtol=0, atol=0)

    def test_worker_random_state_round_trip(self):
        self.assertEqual(RoboDojoHDF5Dataset.__bases__, (Dataset,))
        dataset = RandomWindowDataset()
        reset_rng()
        state = dataset.state_dict()
        expected = dataset[5]
        dataset.load_state_dict(state)
        for key, value in dataset[5].items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    def test_fresh_loader_matches_torch_loader(self):
        for workers in (0, 2):
            with self.subTest(workers=workers):
                reset_rng()
                old = iter(make_loader(workers, stateful=False))
                try:
                    expected = collect(old, 6)
                finally:
                    close(old)
                reset_rng()
                new = iter(make_loader(workers))
                try:
                    self.assert_batches_equal(expected, collect(new, 6))
                finally:
                    close(new)

    def test_prefetched_worker_and_parent_rng_resume(self):
        for workers, persistent in ((0, False), (2, True), (2, False), (12, True)):
            with self.subTest(workers=workers, persistent=persistent):
                reset_rng()
                loader = make_loader(workers, persistent, rank=1, world_size=2)
                iterator = iter(loader)
                try:
                    collect(iterator, 5)
                    buffer = io.BytesIO()
                    torch.save(loader.state_dict(), buffer)
                    parent_state = RandomWindowDataset().state_dict()
                    expected = collect(iterator, 8)
                finally:
                    close(iterator)
                buffer.seek(0)
                resumed = make_loader(workers, persistent, rank=1, world_size=2)
                resumed.load_state_dict(torch.load(buffer, weights_only=False))
                RandomWindowDataset().load_state_dict(parent_state)
                iterator = iter(resumed)
                try:
                    self.assert_batches_equal(expected, collect(iterator, 8))
                    state = copy.deepcopy(resumed.state_dict())
                    parent = RandomWindowDataset().state_dict()
                    expected_again = collect(iterator, 4)
                finally:
                    close(iterator)
                again = make_loader(workers, persistent, rank=1, world_size=2)
                again.load_state_dict(state)
                RandomWindowDataset().load_state_dict(parent)
                iterator = iter(again)
                try:
                    self.assert_batches_equal(expected_again, collect(iterator, 4))
                finally:
                    close(iterator)

    def test_changed_worker_count_is_rejected(self):
        loader = make_loader(2)
        iterator = iter(loader)
        try:
            next(iterator)
            state = loader.state_dict()
        finally:
            close(iterator)
        with self.assertRaisesRegex(ValueError, "settings must match"):
            make_loader(0).load_state_dict(state)

    def test_sampler_iterator_restores_prefetch_cursor(self):
        sampler = DenseBatchSampler(192, 42, 2, 1, 2)
        iterator = iter(sampler)
        for _ in range(7):
            next(iterator)
        state = iterator.state_dict()
        expected = list(iterator)
        restored = iter(sampler)
        restored.load_state_dict(state)
        self.assertEqual(expected, list(restored))

    def test_epoch_boundary_restores_worker_stream(self):
        reset_rng()
        loader = make_loader(2)
        iterator = iter(loader)
        try:
            list(iterator)
            loader.batch_sampler.set_epoch(1)
            iterator = iter(loader)
            collect(iterator, 3)
            state = copy.deepcopy(loader.state_dict())
            parent = RandomWindowDataset().state_dict()
            expected = collect(iterator, 5)
        finally:
            close(iterator)
        resumed = make_loader(2)
        resumed.load_state_dict(state)
        RandomWindowDataset().load_state_dict(parent)
        iterator = iter(resumed)
        try:
            self.assert_batches_equal(expected, collect(iterator, 5))
        finally:
            close(iterator)


class TrainerResumeTests(unittest.TestCase):
    def make_trainer(self):
        trainer = object.__new__(Trainer)
        scheduler = SimpleNamespace(epoch=0, step=4, grad_acc_steps=1)
        scheduler.load_state_dict = mock.Mock()
        trainer.__dict__.update(
            step_scheduler=scheduler, sampler=DenseBatchSampler(192, 42, 2, 0, 1),
            dataloader=make_loader(0), checkpointer=SimpleNamespace(load_on_dp_ranks=mock.Mock()),
        )
        return trainer

    def test_old_checkpoint_warns_and_restores_window_position(self):
        trainer = self.make_trainer()
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch("awomo.training.trainer.torch.load", return_value={"epoch": 0, "data_step": 0}):
                with self.assertLogs(level="WARNING") as logs:
                    trainer._restore_step_and_data_state(folder, True, 96)
        self.assertIn("window positions only", " ".join(logs.output))
        self.assertEqual(trainer.sampler.data_step, 4)
        trainer.checkpointer.load_on_dp_ranks.assert_not_called()

    def test_worker_snapshot_does_not_apply_sampler_offset_twice(self):
        trainer = self.make_trainer()
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "dataloader").mkdir()
            with mock.patch("awomo.training.trainer.torch.load", return_value={"epoch": 0, "data_step": 0}):
                trainer._restore_step_and_data_state(folder, True, 96)
        trainer.checkpointer.load_on_dp_ranks.assert_called_once()
        self.assertEqual(trainer.sampler.data_step, 0)

    def test_partial_worker_snapshot_fails_instead_of_falling_back(self):
        trainer = self.make_trainer()
        trainer.checkpointer.load_on_dp_ranks.side_effect = FileNotFoundError("missing rank snapshot")
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "dataloader").mkdir()
            with mock.patch("awomo.training.trainer.torch.load", return_value={"epoch": 0, "data_step": 0}):
                with self.assertRaisesRegex(FileNotFoundError, "missing rank snapshot"):
                    trainer._restore_step_and_data_state(folder, True, 96)

    def test_distributed_gradient_clipping_is_retained(self):
        trainer = object.__new__(RoboDojoTrainer)
        model, mesh = object(), object()
        trainer.__dict__.update(model=model, device_mesh=mesh, clip_grad_max_norm=1.)
        with mock.patch("awomo.training.robodojo.clip_grad_norm", return_value=torch.tensor(2.)) as clip:
            self.assertEqual(trainer._clip_gradients().item(), 2.)
        clip.assert_called_once_with(1., [model], device_mesh=mesh, foreach=True)
