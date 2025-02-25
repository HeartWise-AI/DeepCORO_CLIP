
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
        config: HeartWiseConfig,
        device: int,
        world_size: int,
    ):
        print("LinearProbingRunner initialized")