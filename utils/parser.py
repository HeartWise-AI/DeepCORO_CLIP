import sys
import argparse

from utils.registry import ParserRegistry
from utils.parser_typing import (
    str2bool, 
    parse_list, 
    parse_optional_int,
    parse_optional_str
)
from utils.config.heartwise_config import HeartWiseConfig


class BaseParser:
    """Base parser for handling common arguments and config loading."""
    def __init__(self, description: str):
        self.parser = argparse.ArgumentParser(description=description, add_help=False)
        self._add_base_arguments()
        self._add_common_arguments()
        
    def _add_base_arguments(self):
        """Adds the essential base_config argument."""
        base_group = self.parser.add_argument_group('Base')
        base_group.add_argument('--base_config', type=str, required=True, help="Path to the base YAML configuration file.")
        
        # Add the wandb arguments
        wandb_group = self.parser.add_argument_group('Wandb')
        wandb_group.add_argument('--use_wandb', type=str2bool, help="Enable Wandb logging.")
        wandb_group.add_argument('--name', type=str, help="Wandb name.")
        wandb_group.add_argument('--project', type=str, help="Wandb project name.")
        wandb_group.add_argument('--entity', type=str, help="Wandb entity name.")

    def _add_common_arguments(self):
        """Adds the common arguments."""
        common_group = self.parser.add_argument_group('Common')
        common_group.add_argument('--seed', type=int, help="Random seed for reproducibility.")
        common_group.add_argument('--device', type=str, help="Device to use (e.g., 'cuda', 'cpu').")
        common_group.add_argument('--world_size', type=int, help="Number of processes for distributed training.")
    
@ParserRegistry.register("DeepCORO_clip")
@ParserRegistry.register("DeepCORO_clip_test")
class ClipParser(BaseParser):
    """Parser for CLIP-style contrastive pretraining."""
    def __init__(self):
        super().__init__(description="Train DeepCORO_CLIP model (Contrastive Pipeline)")
        self.parser.add_help = True
        self.parser.prog = sys.argv[0]
        self._add_clip_arguments()

    def _add_clip_arguments(self):
        """Adds arguments specific to the CLIP pipeline."""
        clip_train_group = self.parser.add_argument_group('CLIP Training parameters')
        clip_train_group.add_argument('--lr', type=float, help="Learning rate for the CLIP model.")
        clip_train_group.add_argument('--batch_size', type=int, help="Batch size for the CLIP model.")
        clip_train_group.add_argument('--num_workers', type=int, help="Number of workers for the CLIP model.")
        clip_train_group.add_argument('--debug', type=str2bool, help="Enable debug mode for the CLIP model.")
        clip_train_group.add_argument('--temperature', type=float, help="Temperature parameter for CLIP loss.")
        clip_train_group.add_argument('--base_checkpoint_path', type=str, help="Path to the base checkpoint file.")
        
        clip_data_group = self.parser.add_argument_group('CLIP Data parameters')
        clip_data_group.add_argument('--data_filename', type=str, help="Path to the data CSV/manifest file.")
        clip_data_group.add_argument('--root', type=str, help="Root directory for data.")
        clip_data_group.add_argument('--target_label', type=str, help="Column name for the target label (e.g., text description).")
        clip_data_group.add_argument('--datapoint_loc_label', type=str, help="Column name for the video file path or identifier.")
        clip_data_group.add_argument('--frames', type=int, help="Number of frames to sample per video.")
        clip_data_group.add_argument('--stride', type=int, help="Stride between sampled frames.")
        clip_data_group.add_argument('--multi_video', type=str2bool, help="Whether to load multiple videos per sample.")
        clip_data_group.add_argument('--num_videos', type=int, help="Number of videos per sample if multi_video is True.")
        clip_data_group.add_argument('--groupby_column', type=str, help="Column to group data by (e.g., patient ID).")
        clip_data_group.add_argument('--shuffle_videos', type=str2bool, help="Shuffle videos within a group if multi_video is True.")
                
        clip_model_group = self.parser.add_argument_group('CLIP Model parameters')
        clip_model_group.add_argument('--model_name', type=str, help="Name of the video encoder model.")
        clip_model_group.add_argument('--pretrained', type=str2bool, help="Whether to use a pretrained video encoder.")
        clip_model_group.add_argument('--video_freeze_ratio', type=float, help="Ratio of video encoder layers to freeze.")
        clip_model_group.add_argument('--text_freeze_ratio', type=float, help="Ratio of text encoder layers to freeze.")
        clip_model_group.add_argument('--dropout', type=float, help="Dropout rate.")
        clip_model_group.add_argument('--num_heads', type=int, help="Number of output units in the classification head.")
        clip_model_group.add_argument('--aggregator_depth', type=int, help="Depth of the aggregation/classification head.")

        clip_optim_group = self.parser.add_argument_group('CLIP Optimization parameters')
        clip_optim_group.add_argument('--optimizer', type=str, help="Optimizer name (e.g., 'AdamW').")
        clip_optim_group.add_argument('--scheduler_name', type=str, help="Learning rate scheduler name.")
        clip_optim_group.add_argument('--lr_step_period', type=int, help="Period for step LR scheduler.")
        clip_optim_group.add_argument('--factor', type=float, help="Factor for ReduceLROnPlateau scheduler.")
        clip_optim_group.add_argument('--video_weight_decay', type=float, help="Weight decay for video encoder.")
        clip_optim_group.add_argument('--text_weight_decay', type=float, help="Weight decay for text components.")
        clip_optim_group.add_argument('--gradient_accumulation_steps', type=int, help="Number of steps to accumulate gradients.")
        clip_optim_group.add_argument('--num_warmup_percent', type=float, help="Percentage of training steps for warmup.")
        clip_optim_group.add_argument('--num_hard_restarts_cycles', type=float, help="Number of cycles for cosine annealing with restarts.")
        clip_optim_group.add_argument('--warm_restart_tmult', type=int, help="Factor to increase T_i after a restart in cosine annealing.")

        clip_system_group = self.parser.add_argument_group('CLIP System parameters')
        clip_system_group.add_argument('--use_amp', type=str2bool, help="Enable Automatic Mixed Precision (AMP).")
        clip_system_group.add_argument('--period', type=int, help="Logging/checkpointing period.")

        clip_metrics_group = self.parser.add_argument_group('CLIP Metrics parameters')
        clip_metrics_group.add_argument('--loss_name', type=str, help="Name of the loss function.")
        clip_metrics_group.add_argument('--recall_k', type=parse_list, help="Values of k for Recall@k metric.")
        clip_metrics_group.add_argument('--ndcg_k', type=parse_list, help="Values of k for NDCG@k metric.")

        clip_augment_group = self.parser.add_argument_group('CLIP Data Augmentation parameters')
        clip_augment_group.add_argument('--rand_augment', type=str2bool, help="Enable RandAugment.")
        clip_augment_group.add_argument('--resize', type=int, help="Resize dimension for input frames.")
        clip_augment_group.add_argument('--apply_mask', type=str2bool, help="Apply masking augmentation.")
        clip_augment_group.add_argument('--view_count', type=parse_optional_int, help="Number of views for multi-view augmentation.")
        
        clip_checkpoint_group = self.parser.add_argument_group('CLIP Checkpointing parameters')
        clip_checkpoint_group.add_argument('--save_best', type=str, help="Metric to monitor for saving the best checkpoint.")
        clip_checkpoint_group.add_argument('--resume_training', type=str2bool, help="Resume training from a checkpoint.")
        clip_checkpoint_group.add_argument('--checkpoint', type=parse_optional_str, help="Path to a specific checkpoint to load.")
                
    def parse_args_and_update(
        self, 
        config: HeartWiseConfig, 
        args_list = None
    ):
        """Parses all args and updates config."""
        args = self.parser.parse_args(args_list)
        config = HeartWiseConfig.update_config_with_args(config, args)
        return config


@ParserRegistry.register("DeepCORO_video_linear_probing")
class LinearProbingParser(BaseParser):
    """Parser for Linear Probing evaluation."""
    def __init__(self):
        super().__init__(description="Run Linear Probing Evaluation")
        # Don't add help here initially, parse known args first
        self.parser.add_help = False 
        self.parser.prog = sys.argv[0]
        self._add_linear_probing_arguments()
        # Map dot-notation prefixes to (attribute_name, type_conversion_function)
        # This handles arguments like --head_lr.Value=0.001
        self.supported_unknown_args = {
            "head_lr": float,
            "head_weight_decay": float,
            # "head_dropout": float, # Example if needed later
        }
        # Add mappings for simple unknown args if necessary (e.g., --some_flag)
        # self.supported_simple_unknown_args = { ... }

    def _add_linear_probing_arguments(self):
        """Adds arguments specific to the Linear Probing pipeline.
           NOTE: Dictionary parameters intended for sweeping (like head_lr, head_weight_decay)
           should NOT be added here explicitly. They will be handled dynamically
           based on the sweep config's dot notation (e.g., --head_lr.key value) as an unknown argument.
        """
        lp_train_group = self.parser.add_argument_group('Linear Probing Training parameters')
        # Arguments like head_lr, head_weight_decay are NOT defined here.
        lp_train_group.add_argument('--scheduler_name', type=str, help="Learning rate scheduler name.")
        lp_train_group.add_argument('--lr_step_period', type=int, help="Period for step LR scheduler.")
        lp_train_group.add_argument('--factor', type=float, help="Factor for ReduceLROnPlateau scheduler.")
        lp_train_group.add_argument('--optimizer', type=str, help="Optimizer name (e.g., 'AdamW').")
        lp_train_group.add_argument('--video_encoder_weight_decay', type=float, help="Weight decay for the video encoder.")
        lp_train_group.add_argument('--use_amp', type=str2bool, help="Enable Automatic Mixed Precision (AMP).")
        lp_train_group.add_argument('--gradient_accumulation_steps', type=int, help="Number of steps to accumulate gradients.")
        lp_train_group.add_argument('--num_warmup_percent', type=float, help="Percentage of training steps for warmup.")
        lp_train_group.add_argument('--num_hard_restarts_cycles', type=float, help="Number of cycles for cosine annealing with restarts.")
        lp_train_group.add_argument('--warm_restart_tmult', type=int, help="Factor to increase T_i after a restart in cosine annealing.")
        # Removed the flattened head_lr/head_weight_decay arguments

        lp_data_group = self.parser.add_argument_group('Linear Probing Data parameters')
        lp_data_group.add_argument('--data_filename', type=str, help="Path to the data CSV/manifest file.")
        lp_data_group.add_argument('--num_workers', type=int, help="Number of workers for the Linear Probing model.")
        lp_data_group.add_argument('--batch_size', type=int, help="Batch size for the Linear Probing model.")
        lp_data_group.add_argument('--datapoint_loc_label', type=str, help="Column name for the video file path or identifier.")
        lp_data_group.add_argument('--target_label', type=str, help="Column name for the target label (e.g., text description).")
        lp_data_group.add_argument('--rand_augment', type=str2bool, help="Enable RandAugment.")
        lp_data_group.add_argument('--resize', type=int, help="Resize dimension for input frames.")
        lp_data_group.add_argument('--frames', type=int, help="Number of frames to sample per video.")
        lp_data_group.add_argument('--stride', type=int, help="Stride between sampled frames.")
        
        lp_model_group = self.parser.add_argument_group('Linear Probing Video Encoder parameters')
        lp_model_group.add_argument('--model_name', type=str, help="Name of the video encoder model.")
        lp_model_group.add_argument('--aggregator_depth', type=int, help="Depth of the aggregation/classification head.")
        lp_model_group.add_argument('--num_heads', type=int, help="Number of output units in the classification head.")
        lp_model_group.add_argument('--video_freeze_ratio', type=float, help="Ratio of video encoder layers to freeze.")
        lp_model_group.add_argument('--dropout', type=float, help="Dropout rate.")
        lp_model_group.add_argument('--pretrained', type=str2bool, help="Whether to use a pretrained video encoder.")
        lp_model_group.add_argument('--video_encoder_checkpoint_path', type=str, help="Path to the video encoder checkpoint file.")
        lp_model_group.add_argument('--video_encoder_lr', type=float, help="Learning rate for the video encoder.")
        
        lp_linear_probing_group = self.parser.add_argument_group('Linear Probing Linear Probing parameters')
        lp_linear_probing_group.add_argument('--head_structure', type=str, help="Output dimension of each head (e.g., JSON string).") # Example: If passed as string
        lp_linear_probing_group.add_argument('--loss_structure', type=str, help="Loss function for each head (e.g., JSON string).") # Example: If passed as string
        lp_linear_probing_group.add_argument('--head_weights', type=str, help="Weight for each head (e.g., JSON string).") # Example: If passed as string
        # head_dropout might be handled as unknown dict arg if passed like --head_dropout.Value=0.1
        lp_linear_probing_group.add_argument('--head_dropout', type=str, help="Dropout for each head (e.g., JSON string OR handled by dot notation).")
        lp_linear_probing_group.add_argument('--head_task', type=str, help="Task for each head (e.g., JSON string).")

    def parse_args_and_update(
        self,
        config: HeartWiseConfig
    ):
        """Parses known args, loads config, updates with known args,
           then updates dictionary attributes based on unknown (dot-notation) args."""

        # Parse only the arguments defined in _add_*_arguments methods
        # Use parse_known_args to separate known and unknown arguments
        args, unknown = self.parser.parse_known_args()

        # Update config with known arguments provided via command line
        config = HeartWiseConfig.update_config_with_args(config, args)

        print(f"Initial config after known args: {config}")
        print(f"Unknown arguments received: {unknown}")

        # Handle unknown arguments
        unsupported_unknown_args = []
        for arg in unknown:
            if arg.startswith('--'):
                meta, value = arg.split('=', 1)
                key, head = meta.split('.', 1)
                key = key.strip('--')
                if key in self.supported_unknown_args:
                    print(f"Processing unknown argument: --{key}.{head}={value}")
                    if not hasattr(config, key):
                        raise ValueError(f"Config object does not initially have the attribute '{key}' defined in the base config '{args.base_config}'.")
                    
                    # Get the attribute
                    head_obj = getattr(config, key)
                    print(f"Attribute '{key}' current value: {head_obj}")
                    
                    # Check if the attribute is a dictionary
                    if not isinstance(head_obj, dict):
                        raise TypeError(f"Attribute '{key}' must be a dictionary to handle dot notation, but found type {type(head_obj)}.")

                    # Check if the head exists in the dictionary
                    if head not in head_obj:
                        raise ValueError(
                            f"The head '{head}' was not found in the dictionary attribute '{key}' loaded from the base configuration '{args.base_config}'. "
                            f"Make sure base config has the correct attribute '{key}' defined."
                        )
                    
                    # Convert the value to the expected type
                    converted_value = self.supported_unknown_args[key](value)
                    head_obj[head] = converted_value # head_obj is a mutable dictionary reference
                    
                    print(f"Updated attribute '{key}': {head_obj}")
                else:
                    unsupported_unknown_args.append(arg)

        # Raise an error if there are unsupported unknown arguments
        if unsupported_unknown_args:
            raise ValueError(f"Unsupported arguments: {unsupported_unknown_args} for pipeline: {config.pipeline_project}")

        return config

class HeartWiseParser:
    @staticmethod
    def _get_pipeline_parser() -> tuple[BaseParser, HeartWiseConfig]:
        """
        Determines the pipeline from the base config and returns the
        appropriate parser instance and the initial config object.
        """
        # Use a minimal parser just to get the --base_config argument
        initial_parser = argparse.ArgumentParser(add_help=False)
        initial_parser.add_argument('--base_config', type=str, required=True)
        # Parse only known args to avoid errors if other args are present
        known_args, _ = initial_parser.parse_known_args()

        # Load the base config YAML
        yaml_config = HeartWiseConfig.from_yaml(known_args.base_config)

        # Determine the pipeline project name from the config
        if not hasattr(yaml_config, 'pipeline_project'):
            raise ValueError(f"'pipeline_project' key not found in the base configuration file: {known_args.base_config}")

        project_name: str = getattr(yaml_config, 'pipeline_project')

        # Get the appropriate parser class from the registry
        if project_name not in ParserRegistry.list_registered():
            raise ValueError(f"Unknown pipeline_project '{project_name}' specified in config {known_args.base_config}. Registered parsers: {ParserRegistry.list_registered()}")

        parser_cls = ParserRegistry.get(project_name)
        if parser_cls is None: # Should not happen if check above passes, but good practice
            raise ValueError(f"No parser found for pipeline_project '{project_name}' despite being registered.")

        # Instantiate the specific parser (e.g., ClipParser, LinearProbingParser)
        pipeline_parser_instance = parser_cls()

        return pipeline_parser_instance, yaml_config

    @staticmethod
    def parse_config() -> HeartWiseConfig:
        """
        Main entry point. Parses command line arguments using the appropriate
        specialized parser based on the pipeline specified in the base config file.
        Loads the base config, gets the correct parser, lets the parser handle
        known and unknown arguments to update the config.
        """
        # 1. Get the specific parser instance and the base config loaded from YAML
        pipeline_parser, config = HeartWiseParser._get_pipeline_parser()

        # 2. Let the specific parser handle all arguments (known and unknown)
        updated_config = pipeline_parser.parse_args_and_update(config)

        # 3. Perform any final updates or checks (e.g., setting GPU info)
        HeartWiseConfig.set_gpu_info_in_place(updated_config) # Example if needed

        return updated_config