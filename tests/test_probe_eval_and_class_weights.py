"""Deterministic eval loaders, BCE pos_weight / CE class weights, test-split scoring,
z-scored regression targets and per-head masked loss normalization (linear probing)."""
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataloaders import video_clip_dataset, video_dataset
from projects.linear_probing_project import LinearProbingProject
from runners.linear_probing_runner import LinearProbingRunner
from utils.loss.losses import BCEWithLogitsLoss, CrossEntropyLoss, MultiHeadLoss
from utils.loss.target_stats import resolve_class_weights, resolve_target_normalization
from utils.loss.typing import Loss
from utils.video import load_video


class _CapturingDataset(list):
    """Stands in for a dataset class: records its constructor kwargs."""

    calls: list = []

    def __init__(self, **kwargs):
        super().__init__([0, 1])
        type(self).calls.append(kwargs)
        self.split = kwargs.get("split")


def _loader_config(**overrides):
    config = SimpleNamespace(
        data_filename="unused.csv", target_label=["y"], datapoint_loc_label="video", frames=4,
        model_name="x3d_m", rand_augment=True, stride=3, resize=32, batch_size=1, num_workers=0,
        shuffle_videos=True,
    )
    config.__dict__.update(overrides)
    return config


class TestEvalLoadersAreDeterministic(unittest.TestCase):
    def _video_dataset_kwargs(self, split, shuffle_videos=None):
        _CapturingDataset.calls = []
        with patch.object(video_dataset, "VideoDataset", _CapturingDataset):
            video_dataset.get_distributed_video_dataloader(
                config=_loader_config(), split=split, mean=[0.0], std=[1.0], shuffle=False,
                num_replicas=1, rank=0, shuffle_videos=shuffle_videos,
            )
        return _CapturingDataset.calls[-1]

    def test_train_split_keeps_augmentation(self):
        kwargs = self._video_dataset_kwargs("train", shuffle_videos=True)
        self.assertTrue(kwargs["rand_augment"])
        self.assertTrue(kwargs["random_stride"])
        self.assertTrue(kwargs["shuffle_videos"])
        self.assertEqual(kwargs["stride"], 3)

    def test_eval_splits_are_not_augmented(self):
        for split in ("val", "test", "inference"):
            with self.subTest(split=split):
                kwargs = self._video_dataset_kwargs(split, shuffle_videos=True)
                self.assertFalse(kwargs["rand_augment"])
                self.assertFalse(kwargs["random_stride"])
                self.assertFalse(kwargs["shuffle_videos"])

    def test_contrastive_eval_loader_uses_fixed_stride(self):
        for split, expected in (("train", True), ("val", False), ("test", False)):
            with self.subTest(split=split):
                _CapturingDataset.calls = []
                with patch.object(video_clip_dataset, "VideoClipDataset", _CapturingDataset):
                    video_clip_dataset.get_distributed_video_clip_dataloader(
                        config=_loader_config(), split=split, mean=[0.0], std=[1.0],
                        shuffle=False, num_replicas=1, rank=0,
                    )
                kwargs = _CapturingDataset.calls[-1]
                self.assertEqual(kwargs["random_stride"], expected)
                self.assertEqual(kwargs["rand_augment"], expected)

    def test_load_video_fixed_stride_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "v.mp4")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 32))
            rng = np.random.default_rng(0)
            for _ in range(24):
                writer.write(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8))
            writer.release()

            with patch("utils.video.np.random.randint", side_effect=AssertionError("random stride")):
                first = load_video(path, n_frames=4, resize=32, normalize=False, stride=3, random_stride=False)
                second = load_video(path, n_frames=4, resize=32, normalize=False, stride=3, random_stride=False)
            np.testing.assert_array_equal(first, second)
            with patch("utils.video.np.random.randint", return_value=2) as randint:
                load_video(path, n_frames=4, resize=32, normalize=False, stride=3)
            randint.assert_called_once()


class TestClassWeights(unittest.TestCase):
    def test_bce_wrapper_uses_pos_weight(self):
        logits = torch.tensor([0.3, -1.2, 2.0, 0.1])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]))(logits, targets)
        expected = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=torch.tensor([4.0]))
        self.assertTrue(torch.allclose(loss, expected))
        self.assertFalse(torch.allclose(loss, F.binary_cross_entropy_with_logits(logits, targets)))

    def test_multi_head_loss_passes_pos_weight_and_ce_weight(self):
        outputs = {"b": torch.tensor([[0.3], [-1.2], [2.0]]), "m": torch.randn(3, 3)}
        targets = {"b": torch.tensor([1.0, 0.0, 1.0]), "m": torch.tensor([0, 2, 1])}
        weighted = MultiHeadLoss(
            head_structure={"b": 1, "m": 3}, loss_structure={"b": "bce_logit", "m": "ce"},
            pos_weights={"b": torch.tensor([3.0])}, loss_weights={"m": torch.tensor([1.0, 2.0, 5.0])},
        )(outputs, targets)
        self.assertTrue(torch.allclose(weighted["b"], F.binary_cross_entropy_with_logits(
            outputs["b"].squeeze(1), targets["b"], pos_weight=torch.tensor([3.0]))))
        self.assertTrue(torch.allclose(weighted["m"], F.cross_entropy(
            outputs["m"], targets["m"], weight=torch.tensor([1.0, 2.0, 5.0]))))

    def test_default_loss_is_unweighted(self):
        outputs = {"b": torch.tensor([[0.3], [-1.2]])}
        targets = {"b": torch.tensor([1.0, 0.0])}
        for loss_structure in (None, {"b": "bce_logit"}):
            losses = MultiHeadLoss(head_structure={"b": 1}, loss_structure=loss_structure)(outputs, targets)
            self.assertTrue(torch.allclose(
                losses["main"], F.binary_cross_entropy_with_logits(outputs["b"].squeeze(1), targets["b"])))

    def test_resolve_auto_weights(self):
        outcomes = [{"b": 1.0, "m": "x"}, {"b": 1.0, "m": "y"}] + [{"b": 0.0, "m": "x"}] * 8
        outcomes += [{"b": float("nan"), "m": float("nan")}]
        pos, ce = resolve_class_weights(
            "auto", head_structure={"b": 1, "m": 3, "r": 1},
            loss_structure={"b": "bce_logit", "m": "ce", "r": "mse"}, train_outcomes=outcomes,
            labels_map={"m": {"x": 0, "y": 1, "z": 2}},
        )
        self.assertEqual(pos["b"].tolist(), [4.0])  # 8 negatives / 2 positives
        self.assertEqual(ce["m"].tolist(), [1.0, 9.0, 1.0])  # 9/9, 9/1, absent class -> 1
        self.assertNotIn("r", pos)

    def test_resolve_caps_explicit_and_errors(self):
        outcomes = [{"b": 1.0}] + [{"b": 0.0}] * 99
        pos, _ = resolve_class_weights({"b": "auto"}, {"b": 1}, {"b": "bce_logit"}, outcomes)
        self.assertEqual(pos["b"].tolist(), [10.0])
        pos, _ = resolve_class_weights({"b": "auto"}, {"b": 1}, {"b": "bce_logit"}, outcomes, cap=50)
        self.assertEqual(pos["b"].tolist(), [50.0])
        pos, _ = resolve_class_weights({"b": 2.5}, {"b": 1}, None, outcomes)
        self.assertEqual(pos["b"].tolist(), [2.5])
        self.assertEqual(resolve_class_weights(None, {"b": 1}, None, outcomes), ({}, {}))
        with self.assertRaises(ValueError):
            resolve_class_weights({"r": "auto"}, {"r": 1}, {"r": "mse"}, outcomes)
        with self.assertRaises(ValueError):
            resolve_class_weights({"m": [1.0, 2.0]}, {"m": 3}, {"m": "ce"}, outcomes)

    def test_project_builds_weighted_loss_from_train_split(self):
        config = SimpleNamespace(
            head_structure={"b": 1, "r": 1}, loss_structure={"b": "bce_logit", "r": "mse"},
            head_weights={"b": 1.0, "r": 1.0}, head_task={"b": "binary_classification", "r": "regression"},
            class_weights="auto", class_weight_cap=10.0, normalize_regression_targets=True,
            labels_map=None, is_ref_device=False,
        )
        outcomes = [{"b": 1.0, "r": 10.0}, {"b": 0.0, "r": 20.0}, {"b": 0.0, "r": 30.0}]
        train_loader = SimpleNamespace(dataset=SimpleNamespace(outcomes=outcomes))
        loss = LinearProbingProject(config, Mock())._build_loss_fn(train_loader=train_loader).loss_type
        self.assertEqual(loss.pos_weights["b"].tolist(), [2.0])
        self.assertAlmostEqual(loss.target_normalization["r"]["mean"], 20.0)
        self.assertAlmostEqual(loss.target_normalization["r"]["std"], float(np.std([10, 20, 30])))

    def test_project_default_loss_does_not_read_train_split(self):
        config = SimpleNamespace(
            head_structure={"b": 1}, loss_structure={"b": "bce_logit"}, head_weights={"b": 1.0},
        )
        loss = LinearProbingProject(config, Mock())._build_loss_fn(train_loader=object()).loss_type
        self.assertEqual(loss.pos_weights, {})
        self.assertEqual(loss.target_normalization, {})


class TestTargetNormalization(unittest.TestCase):
    def test_loss_uses_zscored_targets_and_outputs_denormalize(self):
        stats = {"r": {"mean": 50.0, "std": 10.0}}
        loss_fn = MultiHeadLoss(head_structure={"r": 1}, loss_structure={"r": "mse"}, target_normalization=stats)
        outputs = {"r": torch.tensor([[0.5], [-1.0]])}
        targets = {"r": torch.tensor([60.0, 40.0])}
        self.assertTrue(torch.allclose(loss_fn(outputs, targets)["r"], torch.tensor((0.5 - 1.0) ** 2 / 2)))
        self.assertTrue(torch.allclose(loss_fn.denormalize_outputs(outputs)["r"], torch.tensor([[55.0], [40.0]])))
        self.assertIs(MultiHeadLoss(head_structure={"r": 1}).denormalize_outputs(outputs), outputs)

    def test_resolve_target_normalization(self):
        outcomes = [{"r": 1.0, "s": 5.0}, {"r": 3.0, "s": float("nan")}, None]
        task = {"r": "regression", "s": "regression", "b": "binary_classification"}
        self.assertEqual(resolve_target_normalization(False, task, outcomes), {})
        self.assertEqual(resolve_target_normalization(["r"], task, outcomes), {"r": {"mean": 2.0, "std": 1.0}})
        with self.assertRaises(ValueError):  # single value -> std 0
            resolve_target_normalization(True, task, outcomes)
        with self.assertRaises(ValueError):
            resolve_target_normalization(["b"], task, outcomes)

    def test_runner_denormalizes_logits(self):
        runner = LinearProbingRunner.__new__(LinearProbingRunner)
        runner.loss_fn = Loss(loss_type=MultiHeadLoss(
            head_structure={"r": 1}, loss_structure={"r": "mse"},
            target_normalization={"r": {"mean": 1.0, "std": 2.0}},
        ))
        self.assertEqual(runner._denormalize_outputs({"r": torch.tensor([1.0])})["r"].item(), 3.0)


class TestMaskedPerHeadNormalization(unittest.TestCase):
    def test_head_loss_is_mean_over_labelled_targets(self):
        loss_fn = MultiHeadLoss(head_structure={"r": 1, "b": 1}, loss_structure={"r": "mae", "b": "bce_logit"})
        outputs = {"r": torch.tensor([[1.0], [2.0], [9.0], [4.0]]), "b": torch.tensor([[0.2], [0.4], [-0.3], [1.0]])}
        nan = float("nan")
        targets = {"r": torch.tensor([0.0, nan, nan, 0.0]), "b": torch.tensor([nan, nan, nan, nan])}
        losses = loss_fn(outputs, targets)
        self.assertTrue(torch.allclose(losses["r"], torch.tensor(2.5)))  # (1 + 4) / 2 labelled
        self.assertNotIn("b", losses)  # no labelled target -> no contribution
        self.assertTrue(torch.allclose(losses["main"], torch.tensor(2.5)))


class TestRunnerTestSplit(unittest.TestCase):
    def _runner(self, test_loader):
        runner = LinearProbingRunner.__new__(LinearProbingRunner)
        runner.test_loader = test_loader
        runner.val_loader = Mock(name="val_loader")
        runner.config = SimpleNamespace(run_mode="test")
        return runner

    def test_test_without_test_loader_raises(self):
        runner = self._runner(None)
        with patch.object(LinearProbingRunner, "_evaluate") as evaluate:
            with self.assertRaises(RuntimeError):
                runner.test()
        evaluate.assert_not_called()

    def test_test_rejects_non_test_split(self):
        runner = self._runner(SimpleNamespace(dataset=SimpleNamespace(split="val")))
        with patch.object(LinearProbingRunner, "_evaluate"), self.assertRaises(RuntimeError):
            runner.test()

    def test_test_scores_test_loader(self):
        loader = SimpleNamespace(dataset=SimpleNamespace(split="test"))
        runner = self._runner(loader)
        with patch.object(LinearProbingRunner, "_evaluate") as evaluate:
            runner.test()
        evaluate.assert_called_once_with(loader=loader, mode="test", desc="Test")

    def test_project_hands_test_loader_to_runner(self):
        project = LinearProbingProject(SimpleNamespace(run_mode="test"), Mock())
        loader = object()
        with patch.object(LinearProbingProject, "_setup_validation_objects",
                          return_value={"val_loader": loader, "loss_fn": None}):
            objects = project._setup_test_objects()
        self.assertIs(objects["test_loader"], loader)
        self.assertIsNone(objects["val_loader"])


if __name__ == "__main__":
    unittest.main()
