import argparse
from utils.config import HeartWiseConfig

class HeartWiseParser:
    @staticmethod
    def parse_config() -> HeartWiseConfig:
        """Parse command line arguments and load config file."""
        parser = argparse.ArgumentParser(description="Train DeepCORO_CLIP model")
        parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
        args = parser.parse_args()
        
        return HeartWiseConfig.from_yaml(args.config)