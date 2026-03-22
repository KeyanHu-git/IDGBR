import os
import sys
from pathlib import Path

import torch
from torchvision import transforms
try:
    from torchvision.transforms import InterpolationMode
except Exception:
    from torchvision.transforms.functional import InterpolationMode


def mean_flat(x):
    return x.mean(dim=list(range(1, x.ndim)))


def _resolve_dinov2_code_path():
    local_path = Path(__file__).resolve().parents[1] / "models" / "dinov2"
    if local_path.is_dir():
        return str(local_path)
    env_path = os.getenv("DINOv2_CODE_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    return None


def _parse_patch_size(model_config: str) -> int:
    if model_config and len(model_config) > 1 and model_config[1:].isdigit():
        return int(model_config[1:])
    return 16


@torch.no_grad()
def load_encoders_from_file(encoder_path, device):
    if encoder_path is None or str(encoder_path) == "None":
        return None, None, None

    encoder_path = os.path.abspath(encoder_path)
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"encoder_path not found: {encoder_path}")

    code_path = _resolve_dinov2_code_path()
    if code_path is None:
        raise RuntimeError("DINOv2 code not found. Set DINOv2_CODE_PATH or install dinov2-main.")
    if code_path not in sys.path:
        sys.path.append(code_path)

    from dinov2.models.vision_transformer import vit_small, vit_base, vit_large, vit_giant2, DinoVisionTransformer

    filename = Path(encoder_path).stem
    parts = filename.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected encoder filename: {filename}")

    encoder_type = parts[0]
    architecture = parts[1]
    model_config = parts[2]

    if encoder_type != "dinov2":
        raise ValueError(f"Unsupported encoder type: {encoder_type}")

    model_mapping = {
        "s": vit_small,
        "b": vit_base,
        "l": vit_large,
        "g": vit_giant2,
    }
    model_size_key = model_config[0].lower()
    if model_size_key not in model_mapping:
        raise ValueError(f"Unsupported DINOv2 model config: {filename}")

    patch_size = _parse_patch_size(model_config)
    encoder = model_mapping[model_size_key](patch_size=patch_size)

    checkpoint = torch.load(encoder_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]

    if isinstance(checkpoint, dict):
        encoder.load_state_dict(checkpoint, strict=True)
    elif isinstance(checkpoint, DinoVisionTransformer):
        encoder = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format.")

    encoder.head = torch.nn.Identity()
    encoder = encoder.to(device).eval()
    return encoder, encoder_type, architecture


_PREPROCESS_CONFIGS = {
    "dinov2": {
        "resize": (448, 448),
        "interpolation": InterpolationMode.BICUBIC,
    },
}


def preprocess_image(x, encoder_type=None):
    if encoder_type is None:
        return x
    if encoder_type not in _PREPROCESS_CONFIGS:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    preprocess_cfg = _PREPROCESS_CONFIGS[encoder_type]
    if "resize" in preprocess_cfg:
        resize_transform = transforms.Resize(
            size=preprocess_cfg["resize"], interpolation=preprocess_cfg["interpolation"]
        )
        x = resize_transform(x)
    return x
