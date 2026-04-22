import argparse
import os

from src.utils.config_parser import ConfigLoader

def _normalize_checkpoint(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    if text.isdigit():
        return f"checkpoint-{int(text)}"
    return text


def _normalize_optional_text(value):
    if value is None:
        return None
    text = str(value)
    if text.strip().lower() in {"none", "null"}:
        return None
    return text


def parse_infer_args(input_args=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config_file", type=str, default=None)
    if input_args is not None:
        pre_args, _ = pre_parser.parse_known_args(input_args)
    else:
        pre_args, _ = pre_parser.parse_known_args()

    cfg_defaults = {}
    if pre_args.config_file:
        cfg_defaults = ConfigLoader.load_recursive(pre_args.config_file)

    parser = argparse.ArgumentParser(description="IDGBR inference")
    parser.add_argument("--config_file", type=str, default=None, help="Path to yaml config file")
    parser.add_argument("--model_path", type=str, default=None, help="Experiment root, checkpoint dir, or exported pipeline dir")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint selector: latest, 30000, or checkpoint-30000")
    parser.add_argument("--base_model_name_or_path", type=str, default=None, help="Fallback SD base model path for checkpoint loading")
    parser.add_argument("--label_embed_dir", type=str, default=None, help="Fallback label embed model path")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset root directory")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset config yaml")
    parser.add_argument("--metadata_file", type=str, default="metadata_i2s_segformer_test.jsonl", help="Metadata file inside data_dir")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save predictions")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--dataloader_num_workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="Classifier-free guidance scale")
    parser.add_argument("--strength", type=float, default=1.0, help="Img2img strength")
    parser.add_argument("--prompt", type=str, default="", help="Fallback prompt when not using dataset text")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt")
    parser.add_argument("--use_dataset_text", dest="use_dataset_text", action="store_true", help="Use text from dataset metadata")
    parser.add_argument("--no_use_dataset_text", dest="use_dataset_text", action="store_false", help="Ignore text from dataset metadata")
    parser.add_argument("--use_rough_guidance", dest="use_rough_guidance", action="store_true", help="Use rough label guidance")
    parser.add_argument("--no_use_rough_guidance", dest="use_rough_guidance", action="store_false", help="Disable rough label guidance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic inference")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample cap for quick runs")
    parser.add_argument("--skip_existing", dest="skip_existing", action="store_true", help="Skip predictions that already exist")
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false", help="Always overwrite predictions")
    parser.set_defaults(use_dataset_text=True, use_rough_guidance=True, skip_existing=True, deterministic=True)

    if cfg_defaults:
        parser.set_defaults(**cfg_defaults)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    args.checkpoint = _normalize_checkpoint(args.checkpoint)
    args.prompt = _normalize_optional_text(args.prompt) or ""
    args.negative_prompt = _normalize_optional_text(args.negative_prompt)

    if args.model_path is None:
        raise ValueError("`--model_path` is required.")
    if args.data_dir is None:
        raise ValueError("`--data_dir` is required.")
    if args.dataset is None:
        raise ValueError("`--dataset` is required.")
    if args.batch_size != 1:
        raise ValueError("Current pipeline output only supports `batch_size=1` reliably.")

    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_path, "predictions")

    return args
