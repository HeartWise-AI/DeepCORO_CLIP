import torch

def get_scheduler(
    scheduler_name: str, 
    optimizer: torch.optim.Optimizer, 
    num_epochs: int,
    factor: float = 0.3,
    step_size: int = 15
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
    else:
        raise ValueError(f"Scheduler {scheduler_name} not found")

