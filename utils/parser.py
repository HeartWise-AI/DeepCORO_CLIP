import os
import argparse

from utils.config import load_config

class HeartWiseParser:
    @staticmethod
    def parse_args():
        """Parse command line arguments and optionally load config file."""
        parser = argparse.ArgumentParser(description="Train DeepCORO_CLIP model")

        # Config file argument
        parser.add_argument("--config", type=str, help="Path to YAML config file")

        # Training parameters
        parser.add_argument("--gpu", type=int, default=None, help="GPU index to use")
        parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU")
        parser.add_argument(
            "--num-workers", type=int, default=4, help="Number of data loading workers"
        )
        parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
        parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
        parser.add_argument("--local_rank", "--local-rank", type=int, default=-1, help="Local rank")
        parser.add_argument("--debug", action="store_true", help="Enable debug mode")
        parser.add_argument(
            "--temp", type=float, default=0.1, help="Temperature for contrastive loss"
        )
        parser.add_argument("--use-amp", action="store_true", help="Use AMP training")

        # Data parameters
        parser.add_argument(
            "--data-filename",
            type=str,
            default="processed/reports/reports_sampled_1000.csv",
            help="Data CSV",
        )
        parser.add_argument("--root", type=str, default="data/", help="Root directory")
        parser.add_argument("--target-label", type=str, default="Report", help="Target text column")
        parser.add_argument("--datapoint-loc-label", type=str, default="FileName", help="Path column")
        parser.add_argument("--frames", type=int, default=16, help="Number of frames")
        parser.add_argument("--stride", type=int, default=2, help="Frame sampling stride")
        parser.add_argument("--rand-aug", type=bool, default=False, help="Use random augmentation")
        # Model parameters
        parser.add_argument(
            "--model-name", type=str, default="mvit_v2_s", help="Video backbone model name"
        )
        parser.add_argument("--pretrained", action="store_true", help="Use pretrained backbone")

        # Optimization parameters
        parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer type")
        parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
        parser.add_argument("--scheduler-type", type=str, default="step", help="LR scheduler type")
        parser.add_argument("--lr-step-period", type=int, default=15, help="LR step period")
        parser.add_argument("--factor", type=float, default=0.3, help="Factor for scheduler")

        # Logging parameters
        parser.add_argument("--project", type=str, default="deepcoro_clip", help="W&B project name")
        parser.add_argument("--entity", type=str, default=None, help="W&B entity name")
        parser.add_argument("--tag", type=str, default=None, help="Additional tag")
        parser.add_argument(
            "--output-dir", type=str, default="outputs", help="Directory to save outputs"
        )

        # Loss parameters
        parser.add_argument("--loss-name", type=str, default="contrastive", help="Loss function name")

        args = parser.parse_args()

        # Load config file if provided
        if args.config:
            config = load_config(args.config)
            args_dict = vars(args)
            for key, value in config.items():
                if key in args_dict and args_dict[key] == parser.get_default(key):
                    args_dict[key] = value
            # Explicitly cast known numeric parameters to float
            args.lr = float(args.lr)
            args.weight_decay = float(args.weight_decay)
            args.factor = float(args.factor)

        if args.local_rank == -1:
            args.local_rank = int(os.environ.get("LOCAL_RANK", -1))

        return args