import torch
from transformers import get_linear_schedule_with_warmup

def get_scheduler(
    scheduler_name: str, 
    optimizer: torch.optim.Optimizer, 
    num_epochs: int,
    train_dataloader: torch.utils.data.DataLoader,
    factor: float = 0.3,
    step_size: int = 15,
    gradient_accumulation_steps: int = 1,
    num_warmup_percent: float = 0.1
) -> torch.optim.lr_scheduler.LRScheduler:
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs
        )
    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=step_size, 
            gamma=factor
        )
    elif scheduler_name == 'cosine_warm_restart':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            last_epoch=-1
        )
    elif scheduler_name == 'linear_warmup':
        t_total = len(train_dataloader) // num_epochs * gradient_accumulation_steps
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_epochs * num_warmup_percent),
            num_training_steps=t_total
        )
    else:
        raise ValueError(f"Scheduler {scheduler_name} not found")

