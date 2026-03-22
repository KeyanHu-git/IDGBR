import os
import argparse

from src.utils.config_parser import ConfigLoader, _resolve_relative_paths


def _resolve_extra_paths(cfg, config_file_path):
    config_dir = os.path.dirname(os.path.abspath(config_file_path))
    for key in ("init_from", "output_dir"):
        value = cfg.get(key)
        if isinstance(value, str) and (value.startswith("./") or value.startswith("../")):
            cfg[key] = os.path.normpath(os.path.join(config_dir, value))
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
        cfg_defaults = _resolve_relative_paths(cfg_defaults, pre_args.config_file)
        cfg_defaults = _resolve_extra_paths(cfg_defaults, pre_args.config_file)

    parser = argparse.ArgumentParser(description="LabelEmbedNet Training Script")
    parser.add_argument("--config_file", type=str, default=None, help="Path to yaml config file")

    # Core
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for trained label embed net")
    parser.add_argument("--init_from", type=str, default=None, help="Optional pretrained label embed net directory")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of label classes")
    parser.add_argument("--label_emb_inchannels", type=int, default=3, help="Embedding channels for label tokens")
    parser.add_argument("--label_resolution", type=int, default=64, help="Label map resolution for training")
    parser.add_argument("--noise_std", type=float, default=0.15, help="Noise std added to label embeddings")

    # Training
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--max_train_steps", type=int, default=20000, help="Total training steps")
    parser.add_argument("--checkpointing_steps", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Max number of checkpoints to keep")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Optimizer & Scheduler
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale learning rate by batch size")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="Scheduler type")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Scheduler cycles")
    parser.add_argument("--lr_power", type=float, default=1.0, help="Scheduler power")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Adam weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--set_grads_to_none", action="store_true", help="Set gradients to None instead of zero")

    # Precision & logging
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true", help="Allow TF32 on Ampere GPUs")
    parser.add_argument("--resume_from_checkpoint", type=str, default="latest", help="Resume from checkpoint")
    parser.add_argument("--log_steps", type=int, default=100, help="Log every N steps")
    parser.add_argument("--logging_dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Report to tracker")
    parser.add_argument("--tracker_project_name", type=str, default="train_label_embed", help="Tracker project name")

    if cfg_defaults:
        parser.set_defaults(**cfg_defaults)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.output_dir is None:
        raise ValueError("`--output_dir` must be set for label-embed training.")

    return args
