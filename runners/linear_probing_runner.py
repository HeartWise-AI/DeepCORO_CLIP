
from utils.config.heartwise_config import HeartWiseConfig
from utils.registry import RunnerRegistry


@RunnerRegistry.register("DeepCORO_video_linear_probing")
class LinearProbingRunner:
    """
    This class runs a linear probing pipeline using a VideoEncoder and TextEncoder.
    It handles both training and validation loops in a distributed data-parallel setting.
    """

    def __init__(
        self,
        config: HeartWiseConfig
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