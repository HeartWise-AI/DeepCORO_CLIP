import torch
from tqdm import tqdm
from typing import Tuple
from torch.utils.data import DataLoader

from utils.config.heartwise_config import HeartWiseConfig
from dataloaders.stats_dataset import get_stats_dataloader

def calculate_dataset_statistics_ddp(config: HeartWiseConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate dataset statistics (mean and standard deviation) and broadcast them in distributed environments.

    Args:
        config (HeartWiseConfig): Configuration object

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation tensors
    """
    # Check if pre-computed mean/std are provided in config
    if hasattr(config, 'dataset_mean') and hasattr(config, 'dataset_std') and config.dataset_mean is not None and config.dataset_std is not None:
        mean = torch.tensor(config.dataset_mean)
        std = torch.tensor(config.dataset_std)
        if config.is_ref_device:
            print("\n=== Using Pre-computed Dataset Statistics ===")
            print(f"Mean: {mean.tolist()}")
            print(f"Std:  {std.tolist()}")
            print("===========================\n")
        return mean, std

    mean, std = None, None

    # Calculate statistics only on reference device
    if config.is_ref_device:
        print("\n=== Calculating Dataset Statistics ===")
        
        stats_loader: DataLoader = get_stats_dataloader(config)
        
        print(f"Frame count per video: {config.frames}")
        print(f"Number of videos: {len(stats_loader)}")
        
        assert len(stats_loader) > 0, f"No videos found in the dataset for mode {config.run_mode}"
        
        mean_sum, squared_sum, pixel_count = 0.0, 0.0, 0
        # Only use first batch for quick statistics
        for i, batch in enumerate(tqdm(stats_loader, desc="Calculating statistics")):
            batch = batch.float()
            batch = batch.reshape(-1, batch.shape[-1])
            mean_sum += batch.sum(dim=0)
            squared_sum += (batch**2).sum(dim=0)
            pixel_count += batch.shape[0]
            if i == 0:  # Only process first batch
                print("Using only first batch for quick statistics calculation")
                break
            
        mean: torch.Tensor = mean_sum / pixel_count
        std: torch.Tensor = torch.sqrt((squared_sum / pixel_count) - (mean**2))
        
        print("\nDataset Statistics:")
        print(f"Mean: {mean.tolist()}")
        print(f"Std:  {std.tolist()}")
        print("===========================\n")
    
    # Broadcast statistics if distributed
    if torch.distributed.is_initialized():
        if mean is not None:
            mean = mean.cuda()
            std = std.cuda()
        mean_tensor = torch.zeros(3, device="cuda")
        std_tensor = torch.zeros(3, device="cuda")
        if config.is_ref_device:
            mean_tensor.copy_(mean)
            std_tensor.copy_(std)
        torch.distributed.broadcast(mean_tensor, 0)
        torch.distributed.broadcast(std_tensor, 0)
        mean = mean_tensor.cpu()
        std = std_tensor.cpu()
    
    print(f"Rank: {config.device} - mean: {mean} - std: {std}")
    
    # Return default values if mean and std are None
    if mean is None:
        mean = torch.tensor([0.485, 0.456, 0.406])
    if std is None:
        std = torch.tensor([0.229, 0.224, 0.225])
        
    return mean, std