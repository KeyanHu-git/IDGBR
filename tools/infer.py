#!/usr/bin/env python
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from diffusers import AutoencoderKL, DDIMScheduler
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data import build_dataset
from src.models.SuperModel import InteractNet, LabelEmbedNet
from src.models.UnetModel import UNet2DConditionEncodeModel, UNet2DConditionNewModel
from src.models.pipeline_sidmodel_img2img import StableDiffusionSIDModelImg2ImgPipeline
from src.utils.infer_config import parse_infer_args
from src.utils.util import import_model_class_from_model_name_or_path

LOGGER = logging.getLogger("infer")


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _set_determinism(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _is_full_pipeline_dir(path: Path) -> bool:
    required = [
        "model_index.json",
        "tokenizer",
        "text_encoder",
        "vae",
        "scheduler",
        "unet_ori",
        "unet_gen",
        "interact_net",
        "label_embed_net",
    ]
    return path.is_dir() and all((path / name).exists() for name in required)


def _is_checkpoint_dir(path: Path) -> bool:
    return path.is_dir() and (path / "unet_gen").is_dir() and (path / "interact_net").is_dir()


def _list_checkpoints(experiment_dir: Path) -> List[Path]:
    checkpoints = []
    for item in experiment_dir.iterdir():
        if not item.is_dir() or not item.name.startswith("checkpoint-"):
            continue
        try:
            step = int(item.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, item))
    return [path for _, path in sorted(checkpoints, key=lambda pair: pair[0])]


def _resolve_checkpoint_dir(model_path: Path, checkpoint: Optional[str]) -> Tuple[Path, Optional[Path]]:
    if checkpoint is None and _is_full_pipeline_dir(model_path):
        return model_path, None

    if _is_checkpoint_dir(model_path):
        if model_path.name.startswith("checkpoint-"):
            return model_path.parent, model_path
        return model_path, model_path

    if checkpoint is None:
        return model_path, None

    if not model_path.is_dir():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    if checkpoint == "latest":
        checkpoints = _list_checkpoints(model_path)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint-* directories found under {model_path}")
        return model_path, checkpoints[-1]

    checkpoint_dir = model_path / checkpoint
    if not _is_checkpoint_dir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory is invalid: {checkpoint_dir}")
    return model_path, checkpoint_dir


def _resolve_base_source(experiment_root: Path, base_model_name_or_path: Optional[str]) -> Path:
    required = ["tokenizer", "text_encoder", "vae", "scheduler", "unet_ori"]
    if experiment_root.is_dir() and all((experiment_root / name).exists() for name in required):
        return experiment_root
    if not base_model_name_or_path:
        raise ValueError("`--base_model_name_or_path` is required when the experiment root does not contain base components.")
    base_source = Path(base_model_name_or_path).expanduser().resolve()
    if not base_source.exists():
        raise FileNotFoundError(f"Base model path does not exist: {base_source}")
    return base_source


def _resolve_label_embed_source(experiment_root: Path, label_embed_dir: Optional[str]) -> Path:
    local_dir = experiment_root / "label_embed_net"
    if local_dir.is_dir():
        return local_dir
    if not label_embed_dir:
        raise ValueError("`--label_embed_dir` is required when the experiment root does not contain `label_embed_net`.")
    source = Path(label_embed_dir).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Label embed path does not exist: {source}")
    return source


def _load_pipeline_from_exported(model_dir: Path, device: torch.device) -> StableDiffusionSIDModelImg2ImgPipeline:
    LOGGER.info("Loading exported pipeline from %s", model_dir)
    pipeline = StableDiffusionSIDModelImg2ImgPipeline.from_pretrained(
        str(model_dir),
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline.set_progress_bar_config(disable=True)
    return pipeline.to(device)


def _load_pipeline_from_checkpoint(
    experiment_root: Path,
    checkpoint_dir: Path,
    base_source: Path,
    label_embed_source: Path,
    device: torch.device,
) -> StableDiffusionSIDModelImg2ImgPipeline:
    LOGGER.info("Loading checkpoint pipeline from %s", checkpoint_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(base_source), subfolder="tokenizer", use_fast=False)
    text_encoder_cls = import_model_class_from_model_name_or_path(str(base_source), revision=None)
    text_encoder = text_encoder_cls.from_pretrained(str(base_source), subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(str(base_source), subfolder="vae")
    scheduler = DDIMScheduler.from_pretrained(str(base_source), subfolder="scheduler")
    scheduler.config.prediction_type = "epsilon"
    unet_ori = UNet2DConditionEncodeModel.from_pretrained(str(base_source), subfolder="unet_ori" if (base_source / "unet_ori").is_dir() else "unet")
    unet_gen = UNet2DConditionNewModel.from_pretrained(str(checkpoint_dir / "unet_gen"))
    interact_dir = checkpoint_dir / "interact_net"
    if not interact_dir.is_dir():
        interact_dir = experiment_root / "interact_net"
    interact_net = InteractNet.from_pretrained(str(interact_dir))
    label_embed_net = LabelEmbedNet.from_pretrained(str(label_embed_source))

    pipeline = StableDiffusionSIDModelImg2ImgPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet_ori=unet_ori,
        unet_gen=unet_gen,
        interact_net=interact_net,
        label_embed_net=label_embed_net,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipeline.set_progress_bar_config(disable=True)
    return pipeline.to(device)


def _build_dataloader(args) -> Tuple[Any, DataLoader]:
    with open(args.dataset, "r") as handle:
        dataset_cfg = yaml.safe_load(handle) or {}

    dataset = build_dataset(
        dataset_cfg,
        path=args.data_dir,
        metadata_file=args.metadata_file,
        num_samples_to_use=args.max_samples,
    )

    def _seed_worker(worker_id: int):
        if args.seed is None:
            return
        worker_seed = int(args.seed) + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
    )
    return dataset, dataloader


def _unwrap_single_image(images: Any) -> Image.Image:
    if isinstance(images, Image.Image):
        return images
    if isinstance(images, list):
        if len(images) != 1:
            raise ValueError("Current inference entry expects a single image output per batch.")
        return _unwrap_single_image(images[0])
    if isinstance(images, np.ndarray):
        if images.ndim == 3:
            return Image.fromarray(images.astype(np.uint8))
        if images.ndim == 4 and images.shape[0] == 1:
            return Image.fromarray(images[0].astype(np.uint8))
    raise TypeError(f"Unsupported pipeline output type: {type(images)!r}")


def _resolve_prompt(batch: Dict[str, Any], args) -> Any:
    if args.use_dataset_text and "text" in batch:
        return batch["text"]
    return args.prompt


def _serialize_prompt(prompt: Any) -> Any:
    if isinstance(prompt, (list, tuple)) and len(prompt) == 1:
        return prompt[0]
    return prompt


def _write_resolved_config(output_dir: Path, args, resolved: Dict[str, Any]):
    payload = dict(vars(args))
    payload.update(resolved)
    config_path = output_dir / "resolved_infer_config.yaml"
    with open(config_path, "w") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    _setup_logging()
    args = parse_infer_args()

    model_path = Path(args.model_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _select_device()
    generator = None
    if args.seed is not None and args.deterministic:
        _set_determinism(args.seed)
        generator_device = "cuda" if device.type == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(args.seed)
        torch.manual_seed(args.seed)

    experiment_root, checkpoint_dir = _resolve_checkpoint_dir(model_path, args.checkpoint)
    resolved: Dict[str, Any] = {
        "resolved_model_path": str(model_path),
        "resolved_experiment_root": str(experiment_root),
        "resolved_checkpoint_dir": None,
        "load_mode": "exported_pipeline",
        "device": str(device),
    }

    if checkpoint_dir is None:
        if not _is_full_pipeline_dir(model_path):
            raise FileNotFoundError(
                f"`{model_path}` is neither a checkpoint dir nor an exported pipeline dir. "
                "Use `--checkpoint` for experiment roots with checkpoint-* folders."
            )
        pipeline = _load_pipeline_from_exported(model_path, device)
    else:
        base_source = _resolve_base_source(experiment_root, args.base_model_name_or_path)
        label_embed_source = _resolve_label_embed_source(experiment_root, args.label_embed_dir)
        resolved.update(
            {
                "resolved_checkpoint_dir": str(checkpoint_dir),
                "resolved_base_source": str(base_source),
                "resolved_label_embed_source": str(label_embed_source),
                "load_mode": "checkpoint",
            }
        )
        pipeline = _load_pipeline_from_checkpoint(
            experiment_root=experiment_root,
            checkpoint_dir=checkpoint_dir,
            base_source=base_source,
            label_embed_source=label_embed_source,
            device=device,
        )

    _write_resolved_config(output_dir, args, resolved)

    dataset, dataloader = _build_dataloader(args)
    manifest_path = output_dir / "predictions.jsonl"

    LOGGER.info("Running inference on %d samples", len(dataset))
    LOGGER.info("Saving outputs to %s", output_dir)

    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        for index, batch in enumerate(tqdm(dataloader, desc="Inference", total=len(dataloader))):
            raw_sample = dataset.raw_datas[index] if hasattr(dataset, "raw_datas") else {}
            image_name = batch["img_name"][0] if isinstance(batch["img_name"], list) else batch["img_name"]
            output_path = output_dir / image_name
            prompt = _resolve_prompt(batch, args)

            record = {
                "index": index,
                "image": raw_sample.get("image"),
                "rough_label_index": raw_sample.get("rough_label_index"),
                "label_index": raw_sample.get("label_index"),
                "prompt": _serialize_prompt(prompt),
                "negative_prompt": args.negative_prompt,
                "output": str(output_path),
                "skipped": False,
            }

            if args.skip_existing and output_path.exists():
                record["skipped"] = True
                manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            result = pipeline(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                image=batch["image"],
                rough_label=batch["rough_label_index"],
                use_rough_guidance=args.use_rough_guidance,
                strength=args.strength,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )
            output_image = _unwrap_single_image(result.images)
            output_image.save(output_path)
            manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    LOGGER.info("Inference finished. Manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
