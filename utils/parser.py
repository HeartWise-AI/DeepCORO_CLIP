import os
import argparse
from utils.config import HeartWiseConfig
from utils.parser_typing import (
    str2bool, 
    parse_list, 
    parse_optional_int,
    parse_optional_str
)

class HeartWiseParser:
    @staticmethod
    def set_gpu_info(config: HeartWiseConfig) -> None:
        """Set GPU information from environment variables."""
        config.gpu = int(os.environ["LOCAL_RANK"])
        config.world_size = int(os.environ["WORLD_SIZE"])
        config.is_ref_device = (int(os.environ["LOCAL_RANK"]) == 0)

    @staticmethod
    def parse_config() -> HeartWiseConfig:
        """Parse command line arguments and load config file."""
        parser = argparse.ArgumentParser(description="Train DeepCORO_CLIP model")

        # base config
        base_group = parser.add_argument_group('Base')
        base_group.add_argument('--base_config', type=str, required=True)

        # Training parameters
        train_group = parser.add_argument_group('Training')
        train_group.add_argument('--lr', type=float)
        train_group.add_argument('--batch_size', type=int)
        train_group.add_argument('--epochs', type=int)
        train_group.add_argument('--num_workers', type=int)
        train_group.add_argument('--debug', type=str2bool)
        train_group.add_argument('--temperature', type=float)
        train_group.add_argument('--mode', type=str)
        
        # Optimization parameters
        optim_group = parser.add_argument_group('Optimization')
        optim_group.add_argument('--optimizer', type=str)
        optim_group.add_argument('--scheduler_type', type=str)
        optim_group.add_argument('--lr_step_period', type=int)
        optim_group.add_argument('--factor', type=float)
        optim_group.add_argument('--video_weight_decay', type=float)
        optim_group.add_argument('--text_weight_decay', type=float)

        # Data parameters
        data_group = parser.add_argument_group('Data')
        data_group.add_argument('--data_filename', type=str)
        data_group.add_argument('--root', type=str)
        data_group.add_argument('--target_label', type=str)
        data_group.add_argument('--datapoint_loc_label', type=str)
        data_group.add_argument('--frames', type=int)
        data_group.add_argument('--stride', type=int)
        data_group.add_argument('--multi_video', type=str2bool)
        data_group.add_argument('--num_videos', type=int)
        data_group.add_argument('--groupby_column', type=str)
        data_group.add_argument('--shuffle_videos', type=str2bool)
        
        # Model parameters
        model_group = parser.add_argument_group('Model')
        model_group.add_argument('--model_name', type=str)
        model_group.add_argument('--pretrained', type=str2bool)
        model_group.add_argument('--video_freeze_ratio', type=float)
        model_group.add_argument('--text_freeze_ratio', type=float)
        model_group.add_argument('--dropout', type=float)
        model_group.add_argument('--num_heads', type=int)
        model_group.add_argument('--aggregator_depth', type=int)
        model_group.add_argument('--gradient_accumulation_steps', type=int)
        model_group.add_argument('--num_warmup_percent', type=float)
        
        # System parameters
        system_group = parser.add_argument_group('System')
        system_group.add_argument('--output_dir', type=str)
        system_group.add_argument('--seed', type=parse_optional_int)
        system_group.add_argument('--use_amp', type=str2bool)
        system_group.add_argument('--device', type=str)
        system_group.add_argument('--period', type=int)

        # Loss and metrics parameters
        metrics_group = parser.add_argument_group('Loss and Metrics')
        metrics_group.add_argument('--loss_name', type=str)
        metrics_group.add_argument('--recall_k', type=parse_list)
        metrics_group.add_argument('--ndcg_k', type=parse_list)

        # Data augmentation parameters
        augment_group = parser.add_argument_group('Data Augmentation')
        augment_group.add_argument('--rand_augment', type=str2bool)
        augment_group.add_argument('--resize', type=int)
        augment_group.add_argument('--apply_mask', type=str2bool)
        augment_group.add_argument('--view_count', type=parse_optional_int)

        # Checkpointing parameters
        checkpoint_group = parser.add_argument_group('Checkpointing')
        checkpoint_group.add_argument('--save_best', type=str)
        checkpoint_group.add_argument('--resume_training', type=str2bool)
        checkpoint_group.add_argument('--checkpoint', type=parse_optional_str)

        # Logging parameters
        sweep_group = parser.add_argument_group('Sweep')
        sweep_group.add_argument('--tag', type=str)
        sweep_group.add_argument('--name', type=str)
        sweep_group.add_argument('--project', type=str)
        sweep_group.add_argument('--entity', type=str)

        args = parser.parse_args()

        # Load base config from yaml
        config: HeartWiseConfig = HeartWiseConfig.from_yaml(args.base_config)
        
        # Create sweep config from args
        config: HeartWiseConfig = HeartWiseConfig.update_config_with_args(config, args)
        
        # Set GPU info
        HeartWiseParser.set_gpu_info(config)
        
        return config
