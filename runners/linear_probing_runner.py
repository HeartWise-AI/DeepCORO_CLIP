
import torch
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from utils.loss.typing import Loss
from utils.registry import RunnerRegistry
from utils.config.heartwise_config import HeartWiseConfig

from models.linear_probing import LinearProbing


@RunnerRegistry.register("DeepCORO_video_linear_probing")
class LinearProbingRunner:
    """
    This class runs a linear probing pipeline using a VideoEncoder and TextEncoder.
    It handles both training and validation loops in a distributed data-parallel setting.
    """

    def __init__(
        self,
        config: HeartWiseConfig,
        linear_probing: LinearProbing,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        scaler: GradScaler,
        full_output_path: str,
        loss_fn: Loss,
    ):
        self.config: HeartWiseConfig = config
        print("LinearProbingRunner initialized")
        
    def train(
        self, 
        start_epoch: int, 
        end_epoch: int
    ):
        print(f"Training from epoch {start_epoch} to {end_epoch}")
        
    def inference(self):
        raise NotImplementedError("Linear probing inference not implemented")