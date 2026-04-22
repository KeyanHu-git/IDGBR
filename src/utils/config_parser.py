import os
import sys
import yaml
import argparse

def _resolve_relative_values(value, config_dir):
    if isinstance(value, dict):
        return {key: _resolve_relative_values(item, config_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_relative_values(item, config_dir) for item in value]
    if isinstance(value, str) and (value.startswith("./") or value.startswith("../")):
        return os.path.normpath(os.path.join(config_dir, value))
    return value


class ConfigLoader:
    @staticmethod
    def load_recursive(file_path):
        file_path = os.path.abspath(file_path)
        with open(file_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        cfg = _resolve_relative_values(cfg, os.path.dirname(file_path))
        if '_base_' in cfg:
            base_path = cfg.pop('_base_')
            if not os.path.isabs(base_path):
                base_path = os.path.join(os.path.dirname(file_path), base_path)
            base_cfg = ConfigLoader.load_recursive(base_path)
            base_cfg.update(cfg)
            return base_cfg
        
        return cfg


def parse_args(input_args=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config_file", type=str, default=None)
    if input_args is not None:
        pre_args, _ = pre_parser.parse_known_args(input_args)
    else:
        pre_args, _ = pre_parser.parse_known_args()
    cfg_defaults = {}
    if pre_args.config_file is not None:
        cfg_defaults = ConfigLoader.load_recursive(pre_args.config_file)

    parser = argparse.ArgumentParser(description="IDGBR Training Script (Original Config)")
    parser.add_argument("--config_file", type=str, default=None, help="Path to yaml config file")

    # ==============================================================================
    # 1. Model Configuration (Base Model, SD, Tokenizer)
    # ==============================================================================
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--sd_model_name_or_path", type=str, default=None, help="Path to SD model")
    parser.add_argument("--revision", type=str, default=None, help="Model revision")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Tokenizer name")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Enable xformers")
    
    # ==============================================================================
    # 2. Project Specific (Custom Components)
    # ==============================================================================
    parser.add_argument("--label_embed_dir", type=str, default=None, help="Label Embed Net directory")
    parser.add_argument("--encoder_path", type=str, default=None, help="Encoder checkpoint path (e.g., DINOv2)")
    parser.add_argument("--acc_steps", type=int, default=200, help="Alignment steps for encoder guidance")
    parser.add_argument("--enable_alignment", dest="enable_alignment", action="store_true", help="Enable representation alignment")
    parser.add_argument("--disable_alignment", dest="enable_alignment", action="store_false", help="Disable representation alignment")
    parser.add_argument("--proj_coeff", type=float, default=0.5, help="Projection loss coefficient")
    parser.add_argument("--metadata_file", type=str, default=None, help="Metadata filename")
    parser.add_argument("--dataset", default=None, help="Dataset behavior configuration")
    parser.add_argument("--conditioning_dropout_prob", type=float, default=0.2, help="Dropout for conditioning")
    parser.add_argument("--time_sample_strategy", action="store_true", help="Whether to use a cubic time step sampling strategy")
    parser.add_argument("--use_rough_guidance", dest="use_rough_guidance", action="store_true", help="Use rough label guidance")
    parser.add_argument("--no_use_rough_guidance", dest="use_rough_guidance", action="store_false", help="Disable rough label guidance")

    # ==============================================================================
    # 3. Data Configuration (Input, Resolution, Batch)
    # ==============================================================================
    parser.add_argument("--train_data_dir", type=str, default=None, help="Training data directory")
    parser.add_argument("--dataset_name", type=str, default=None, help="HuggingFace dataset name")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="Dataset config name")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory")
    parser.add_argument("--image_column", type=str, default="image", help="Image column name")
    parser.add_argument("--conditioning_image_column", type=str, default=None, help="Conditioning image column name")
    parser.add_argument("--caption_column", type=str, default="text", help="Caption column name")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--dataloader_num_workers", type=int, default=6, help="Number of dataloader workers")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Truncate number of training examples")
    parser.add_argument("--proportion_empty_prompts", type=float, default=0, help="Proportion of empty prompts")

    # ==============================================================================
    # 4. Training Loop Configuration (Steps, Epochs, Checkpoints)
    # ==============================================================================
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of epochs")
    parser.add_argument("--max_train_steps", type=int, default=20000, help="Total training steps")
    parser.add_argument("--checkpointing_steps", type=int, default=5000, help="Save checkpoint every X steps")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Max number of checkpoints to keep")
    parser.add_argument("--resume_from_checkpoint", type=str, default="latest", help="Resume from checkpoint")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Steps to accumulate gradient")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision mode")
    parser.add_argument("--allow_tf32", action="store_true", help="Allow TF32 on Ampere GPUs")

    # ==============================================================================
    # 5. Optimizer & Scheduler Configuration
    # ==============================================================================
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale learning rate by batch size")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="Learning rate scheduler")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Scheduler cycles")
    parser.add_argument("--lr_power", type=float, default=1.0, help="Scheduler power")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8-bit Adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--set_grads_to_none", action="store_true", help="Set gradients to None instead of Zero")

    # ==============================================================================
    # 6. Validation, Logging & Hub
    # ==============================================================================
    parser.add_argument("--validation_prompt", type=str, default=None, nargs="+", help="Prompts for validation")
    parser.add_argument("--validation_image", type=str, default=None, nargs="+", help="Images for validation")
    parser.add_argument("--num_validation_images", type=int, default=4, help="Number of images generated per prompt")
    parser.add_argument("--validation_steps", type=int, default=100, help="Run validation every X steps")
    parser.add_argument("--log_steps", type=int, default=100, help="Log metrics every X steps")
    parser.add_argument("--logging_dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Report to wandb or tensorboard")
    parser.add_argument("--tracker_project_name", type=str, default="train_controlnet", help="Tracker project name")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to HuggingFace Hub")
    parser.add_argument("--hub_token", type=str, default=None, help="Hub token")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Hub model ID")

    # Parse args
    parser.set_defaults(use_rough_guidance=True, enable_alignment=True)
    if cfg_defaults:
        parser.set_defaults(**cfg_defaults)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # ==============================================================================
    # 7. Logical Validation (Original Logic)
    # ==============================================================================
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is None:
        raise ValueError("`--train_data_dir` is required for the current local dataset pipeline.")

    if args.dataset is not None and args.train_data_dir is None:
        raise ValueError("`--train_data_dir` must be set when using `--dataset`.")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (args.validation_image is not None and args.validation_prompt is not None and
        len(args.validation_image) != 1 and len(args.validation_prompt) != 1 and
        len(args.validation_image) != len(args.validation_prompt)):
        raise ValueError("Must provide either 1 `--validation_image`, 1 `--validation_prompt`, or matching numbers")

    return args
