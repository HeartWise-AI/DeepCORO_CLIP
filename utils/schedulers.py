import torch
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

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
        # Compute total training steps: batches per epoch times epochs, adjusted for gradient accumulation.
        t_total = (len(train_dataloader) * num_epochs) // gradient_accumulation_steps
        print(f"t_total: {t_total}")
        # Compute warmup steps as a fraction of the total steps.
        num_warmup_steps = int(t_total * num_warmup_percent)
        print(f"num_warmup_steps: {num_warmup_steps}")
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=t_total
        )
    elif scheduler_name == 'cosine_with_warmup':
        t_total = (len(train_dataloader) * num_epochs) // gradient_accumulation_steps
        num_warmup_steps = int(t_total * num_warmup_percent)
        print(f"num_warmup_steps: {num_warmup_steps}")
        print(f"t_total: {t_total}")
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=t_total
        )
    elif scheduler_name == 'cosine_with_hard_restarts_with_warmup':
        t_total = (len(train_dataloader) * num_epochs) // gradient_accumulation_steps
        num_warmup_steps = int(t_total * num_warmup_percent)
        print(f"num_warmup_steps: {num_warmup_steps}")
        print(f"t_total: {t_total}")
        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=t_total
        )
    else:
        raise ValueError(f"Scheduler {scheduler_name} not found")

