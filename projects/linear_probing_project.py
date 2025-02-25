
from utils.registry import ProjectRegistry
from utils.config.heartwise_config import HeartWiseConfig

@ProjectRegistry.register("DeepCORO_video_linear_probing")
class LinearProbingProject:
    def __init__(self, config: HeartWiseConfig):
        self.config = config

    def run(self):
        print(f"Running linear probing project with config: {self.config}")

