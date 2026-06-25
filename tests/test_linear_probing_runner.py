import unittest
from unittest.mock import Mock, MagicMock, patch
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import GradScaler

from runners.linear_probing_runner import LinearProbingRunner
from utils.config.linear_probing_config import LinearProbingConfig
from utils.wandb_wrapper import WandbWrapper
from utils.loss.typing import Loss


class _TinyLinearProbing(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)

    def forward(self, batch_video, video_mask=None, view_ids=None):
        return {"test_head": self.linear(batch_video.reshape(-1, 1).float())}


class _MSELoss:
    def run(self, outputs, targets):
        target = targets["test_head"].reshape(-1, 1).float()
        return {"main": torch.nn.functional.mse_loss(outputs["test_head"], target)}


class _NonFiniteLoss:
    def run(self, outputs, targets):
        return {"main": outputs["test_head"].sum() * float("nan")}


class _SkippingScaler:
    def __init__(self):
        self._scale = 2.0

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        pass

    def update(self):
        self._scale = 1.0

    def get_scale(self):
        return self._scale


class TestLinearProbingRunner(unittest.TestCase):
    """Test cases for LinearProbingRunner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock config
        self.config = Mock(spec=LinearProbingConfig)
        self.config.world_size = 1
        self.config.device = 0
        self.config.is_ref_device = True
        self.config.scheduler_name = "linear_with_warmup"
        self.config.head_task = {"test_head": "classification"}
        self.config.head_structure = {"test_head": 2}
        self.config.gradient_accumulation_steps = 1
        self.config.use_amp = False
        self.config.max_grad_norm = 0.0
        
        # Create mock components
        self.wandb_wrapper = Mock(spec=WandbWrapper)
        self.wandb_wrapper.is_initialized.return_value = False
        
        self.train_loader = Mock(spec=DataLoader)
        self.val_loader = Mock(spec=DataLoader)
        
        self.linear_probing = Mock()
        self.linear_probing.train = Mock()
        
        self.optimizer = Mock(spec=Adam)
        self.scaler = Mock(spec=GradScaler)
        self.lr_scheduler = Mock()
        self.loss_fn = Mock(spec=Loss)
        
        self.output_dir = "/tmp/test_output"
        
    def test_scheduler_is_per_iteration_with_warmup(self):
        """Test _scheduler_is_per_iteration method with warmup scheduler."""
        # Test case where scheduler contains "warmup"
        self.config.scheduler_name = "linear_with_warmup"
        
        runner = LinearProbingRunner(
            config=self.config,
            wandb_wrapper=self.wandb_wrapper,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            linear_probing=self.linear_probing,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            loss_fn=self.loss_fn,
            output_dir=self.output_dir,
        )
        
        # This should return True because "warmup" is in the scheduler name
        self.assertTrue(runner._scheduler_is_per_iteration())
        
    def test_scheduler_is_per_iteration_with_cosine_warmup(self):
        """Test _scheduler_is_per_iteration method with cosine warmup scheduler."""
        # Test case where scheduler contains "with_warmup"
        self.config.scheduler_name = "cosine_with_warmup"
        
        runner = LinearProbingRunner(
            config=self.config,
            wandb_wrapper=self.wandb_wrapper,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            linear_probing=self.linear_probing,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            loss_fn=self.loss_fn,
            output_dir=self.output_dir,
        )
        
        # This should return True because "with_warmup" is in the scheduler name
        self.assertTrue(runner._scheduler_is_per_iteration())
        
    def test_scheduler_is_per_iteration_without_warmup(self):
        """Test _scheduler_is_per_iteration method without warmup scheduler."""
        # Test case where scheduler doesn't contain warmup keywords
        self.config.scheduler_name = "step_lr"
        
        runner = LinearProbingRunner(
            config=self.config,
            wandb_wrapper=self.wandb_wrapper,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            linear_probing=self.linear_probing,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            loss_fn=self.loss_fn,
            output_dir=self.output_dir,
        )
        
        # This should return False because no warmup keywords are present
        self.assertFalse(runner._scheduler_is_per_iteration())
        
    def test_scheduler_is_per_iteration_empty_scheduler_name(self):
        """Test _scheduler_is_per_iteration method with empty scheduler name."""
        # Test case where scheduler name is empty
        self.config.scheduler_name = ""
        
        runner = LinearProbingRunner(
            config=self.config,
            wandb_wrapper=self.wandb_wrapper,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            linear_probing=self.linear_probing,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            loss_fn=self.loss_fn,
            output_dir=self.output_dir,
        )
        
        # This should return False for empty scheduler name
        self.assertFalse(runner._scheduler_is_per_iteration())

    def test_preprocess_inputs_preserves_video_mask(self):
        """Video masks should be moved to device and kept boolean."""
        self.config.device = torch.device("cpu")
        runner = LinearProbingRunner(
            config=self.config,
            wandb_wrapper=self.wandb_wrapper,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            linear_probing=self.linear_probing,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            loss_fn=self.loss_fn,
            output_dir=self.output_dir,
        )
        batch = {
            "videos": torch.zeros((2, 3, 1, 2, 2, 1)),
            "targets": {"test_head": torch.tensor([0, 1])},
            "video_mask": torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.int64),
        }

        processed = runner._preprocess_inputs(batch)

        self.assertEqual(processed["video_mask"].dtype, torch.bool)
        self.assertEqual(processed["video_mask"].device.type, "cpu")
        self.assertTrue(
            torch.equal(
                processed["video_mask"],
                torch.tensor([[True, False, True], [False, True, False]]),
            )
        )

    @patch("runners.linear_probing_runner.DistributedUtils.sync_process_group")
    def test_train_step_steps_optimizer_without_scaler(self, _mock_sync):
        """Non-AMP training should still call optimizer.step()."""
        self.config.device = torch.device("cpu")
        model = _TinyLinearProbing()
        model.linear.weight.data.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.param_groups[0]["name"] = "main"
        runner = LinearProbingRunner(
            config=self.config,
            wandb_wrapper=self.wandb_wrapper,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            linear_probing=model,
            optimizer=optimizer,
            scaler=None,
            lr_scheduler=None,
            loss_fn=_MSELoss(),
            output_dir=self.output_dir,
        )

        runner._train_step(
            batch_video=torch.tensor([[1.0], [2.0]]),
            batch_targets={"test_head": torch.zeros(2)},
        )

        self.assertLess(model.linear.weight.item(), 1.0)
        self.assertEqual(runner.step, 1)

    @patch("runners.linear_probing_runner.DistributedUtils.sync_process_group")
    def test_train_step_skips_nonfinite_loss(self, _mock_sync):
        """A non-finite loss should not update weights or step the scheduler."""
        self.config.device = torch.device("cpu")
        model = _TinyLinearProbing()
        model.linear.weight.data.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.param_groups[0]["name"] = "main"
        scheduler = Mock()
        runner = LinearProbingRunner(
            config=self.config,
            wandb_wrapper=self.wandb_wrapper,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            linear_probing=model,
            optimizer=optimizer,
            scaler=None,
            lr_scheduler=scheduler,
            loss_fn=_NonFiniteLoss(),
            output_dir=self.output_dir,
        )

        runner._train_step(
            batch_video=torch.tensor([[1.0], [2.0]]),
            batch_targets={"test_head": torch.zeros(2)},
        )

        self.assertEqual(model.linear.weight.item(), 1.0)
        scheduler.step.assert_not_called()
        self.assertEqual(runner.step, 1)

    @patch("runners.linear_probing_runner.DistributedUtils.sync_process_group")
    def test_train_step_does_not_advance_scheduler_when_scaler_skips_step(self, _mock_sync):
        """Per-iteration schedulers should only advance after a real optimizer step."""
        self.config.device = torch.device("cpu")
        model = _TinyLinearProbing()
        model.linear.weight.data.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.param_groups[0]["name"] = "main"
        scheduler = Mock()
        runner = LinearProbingRunner(
            config=self.config,
            wandb_wrapper=self.wandb_wrapper,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            linear_probing=model,
            optimizer=optimizer,
            scaler=_SkippingScaler(),
            lr_scheduler=scheduler,
            loss_fn=_MSELoss(),
            output_dir=self.output_dir,
        )

        runner._train_step(
            batch_video=torch.tensor([[1.0], [2.0]]),
            batch_targets={"test_head": torch.zeros(2)},
        )

        self.assertEqual(model.linear.weight.item(), 1.0)
        scheduler.step.assert_not_called()
        self.assertEqual(runner.step, 1)


if __name__ == '__main__':
    unittest.main()
