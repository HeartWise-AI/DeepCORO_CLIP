import unittest
import torch
import torch.nn as nn
import torch.utils.data
from collections import namedtuple
import numpy as np
from utils.schedulers import get_scheduler


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=100):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return torch.randn(10), torch.tensor(1)


class TestSchedulers(unittest.TestCase):
    def setUp(self):
        # Create a simple model for testing
        self.model = nn.Linear(10, 2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        # Create dummy dataset and dataloader
        dataset = DummyDataset()
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        self.num_epochs = 10
        self.gradient_accumulation_steps = 1

    def test_cosine_scheduler(self):
        scheduler = get_scheduler(
            scheduler_name="cosine",
            optimizer=self.optimizer,
            num_epochs=self.num_epochs,
            train_dataloader=self.dataloader,
            gradient_accumulation_steps=self.gradient_accumulation_steps
        )
        
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        
        # Check that T_max is set correctly
        expected_t_max = (len(self.dataloader) * self.num_epochs) // self.gradient_accumulation_steps
        self.assertEqual(scheduler.T_max, expected_t_max)

    def test_step_scheduler(self):
        step_size = 2
        factor = 0.5
        
        scheduler = get_scheduler(
            scheduler_name="step",
            optimizer=self.optimizer,
            num_epochs=self.num_epochs,
            train_dataloader=self.dataloader,
            step_size=step_size,
            factor=factor
        )
        
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.StepLR)
        self.assertEqual(scheduler.step_size, step_size)
        self.assertEqual(scheduler.gamma, factor)

    def test_cosine_warm_restart_scheduler(self):
        num_restarts = 3
        warm_restart_tmult = 2
        
        scheduler = get_scheduler(
            scheduler_name="cosine_warm_restart",
            optimizer=self.optimizer,
            num_epochs=self.num_epochs,
            train_dataloader=self.dataloader,
            warm_restart_tmult=warm_restart_tmult,
            num_restarts=num_restarts
        )
        
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
        
        # Check that T_0 and T_mult are set correctly
        t_total = (len(self.dataloader) * self.num_epochs) // self.gradient_accumulation_steps
        expected_T_0 = max(1, t_total // num_restarts)
        self.assertEqual(scheduler.T_0, expected_T_0)
        self.assertEqual(scheduler.T_mult, warm_restart_tmult)

    def test_linear_warmup_scheduler(self):
        num_warmup_percent = 0.2
        
        scheduler = get_scheduler(
            scheduler_name="linear_warmup",
            optimizer=self.optimizer,
            num_epochs=self.num_epochs,
            train_dataloader=self.dataloader,
            num_warmup_percent=num_warmup_percent
        )
        
        # Check scheduler type - can't directly check instance type as it's a LambdaLR internally
        # But we can check it's not one of our other types
        self.assertNotIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        self.assertNotIsInstance(scheduler, torch.optim.lr_scheduler.StepLR)
        self.assertNotIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)

    def test_cosine_with_warmup_scheduler(self):
        num_warmup_percent = 0.2
        
        scheduler = get_scheduler(
            scheduler_name="cosine_with_warmup",
            optimizer=self.optimizer,
            num_epochs=self.num_epochs,
            train_dataloader=self.dataloader,
            num_warmup_percent=num_warmup_percent
        )
        
        # Similar to linear_warmup, we can't check exact type but can exclude others
        self.assertNotIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        self.assertNotIsInstance(scheduler, torch.optim.lr_scheduler.StepLR)
        self.assertNotIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)

    def test_cosine_with_hard_restarts_with_warmup_scheduler(self):
        num_warmup_percent = 0.2
        num_hard_restarts_cycles = 2.0
        
        scheduler = get_scheduler(
            scheduler_name="cosine_with_hard_restarts_with_warmup",
            optimizer=self.optimizer,
            num_epochs=self.num_epochs,
            train_dataloader=self.dataloader,
            num_warmup_percent=num_warmup_percent,
            num_hard_restarts_cycles=num_hard_restarts_cycles
        )
        
        # Similar to others from transformers package
        self.assertNotIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        self.assertNotIsInstance(scheduler, torch.optim.lr_scheduler.StepLR)
        self.assertNotIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)

    def test_invalid_scheduler(self):
        with self.assertRaises(ValueError):
            get_scheduler(
                scheduler_name="invalid_scheduler",
                optimizer=self.optimizer,
                num_epochs=self.num_epochs,
                train_dataloader=self.dataloader
            )

    def test_lr_progression(self):
        """Test that learning rate actually changes as expected for different schedulers"""
        # Test each scheduler individually with appropriate settings
        
        # 1. Test cosine scheduler - should see change over more iterations
        scheduler = get_scheduler(
            scheduler_name="cosine",
            optimizer=self.optimizer,
            num_epochs=self.num_epochs,
            train_dataloader=self.dataloader
        )
        
        initial_lr = self.optimizer.param_groups[0]['lr']
        # Run for more steps to see a change in cosine scheduler
        for _ in range(30):
            scheduler.step()
        final_lr = self.optimizer.param_groups[0]['lr']
        self.assertNotEqual(initial_lr, final_lr)
        
        # Reset optimizer's LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 0.01
            
        # 2. Test step scheduler with small step size to ensure change
        scheduler = get_scheduler(
            scheduler_name="step",
            optimizer=self.optimizer,
            num_epochs=self.num_epochs,
            train_dataloader=self.dataloader,
            step_size=1,  # Small step size to ensure we see a change
            factor=0.5    # Significant reduction
        )
        
        initial_lr = self.optimizer.param_groups[0]['lr']
        scheduler.step()  # Just one step should be enough
        final_lr = self.optimizer.param_groups[0]['lr']
        self.assertLess(final_lr, initial_lr)  # LR should decrease
        
        # Reset optimizer's LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 0.01
            
        # 3. Test cosine_warm_restart with small T_0
        scheduler = get_scheduler(
            scheduler_name="cosine_warm_restart",
            optimizer=self.optimizer,
            num_epochs=self.num_epochs,
            train_dataloader=self.dataloader,
            num_restarts=10  # More restarts = smaller T_0
        )
        
        initial_lr = self.optimizer.param_groups[0]['lr']
        # Capture LR values over time
        lrs = [initial_lr]
        for _ in range(10):
            scheduler.step()
            lrs.append(self.optimizer.param_groups[0]['lr'])
        
        # Verify LR changes at some point (not necessarily at the end)
        self.assertTrue(any(lr != initial_lr for lr in lrs[1:]), 
                        "Learning rate should change at some point during schedule")


if __name__ == "__main__":
    unittest.main() 