import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
try:
    from torchvision.transforms import InterpolationMode
except Exception:
    from torchvision.transforms.functional import InterpolationMode


def mean_flat(x):
    return x.mean(dim=list(range(1, x.ndim)))


def _ensure_hw_tuple(size):
    if isinstance(size, int):
        return (int(size), int(size))
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return (int(size[0]), int(size[1]))
    raise ValueError(f"Expected an int or a 2-value size, got: {size!r}")


def _round_down_to_multiple(value: int, multiple: int) -> int:
    value = int(value)
    multiple = int(multiple)
    if value < multiple:
        raise ValueError(f"Value {value} must be >= multiple {multiple}")
    return (value // multiple) * multiple


def _infer_repa_resize_dim(value: int, patch_size: int) -> int:
    # Generalize REPA's 512 -> 448 ratio to arbitrary spatial sizes while keeping
    # the DINO input aligned to the encoder patch size.
    repa_like_dim = max(patch_size, int(value * 7 / 8))
    repa_like_dim = min(int(value), repa_like_dim)
    return _round_down_to_multiple(max(repa_like_dim, patch_size), patch_size)


def _resolve_preprocess_size(image_hw, encoder_type=None, patch_size=None):
    image_h, image_w = _ensure_hw_tuple(image_hw)
    if encoder_type is None:
        return (image_h, image_w)
    if encoder_type == "dinov2":
        patch_multiple = int(patch_size) if patch_size is not None else 14
        return (
            _infer_repa_resize_dim(image_h, patch_multiple),
            _infer_repa_resize_dim(image_w, patch_multiple),
        )
    raise ValueError(f"Unknown encoder type: {encoder_type}")


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


def _parse_encoder_filename(filename: str):
    parts = filename.split("_")
    if len(parts) >= 3 and parts[0] == "dinov2" and parts[1] == "vit":
        return parts[0], parts[1], parts[2]
    if len(parts) >= 2 and parts[0] == "dinov2" and parts[1].startswith("vit"):
        model_config = parts[1][3:]
        if model_config:
            return parts[0], "vit", model_config
    raise ValueError(f"Unexpected encoder filename: {filename}")


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Unsupported checkpoint format.")


@torch.no_grad()
def load_encoders_from_file(encoder_path, device):
    if encoder_path is None or str(encoder_path) == "None":
        return None, None, None, None

    encoder_path = os.path.abspath(encoder_path)
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"encoder_path not found: {encoder_path}")

    code_path = _resolve_dinov2_code_path()
    if code_path is None:
        raise RuntimeError("DINOv2 code not found. Set DINOv2_CODE_PATH or install dinov2-main.")
    if code_path not in sys.path:
        sys.path.append(code_path)

    from dinov2.hub.backbones import dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14

    filename = Path(encoder_path).stem
    encoder_type, architecture, model_config = _parse_encoder_filename(filename)

    if encoder_type != "dinov2":
        raise ValueError(f"Unsupported encoder type: {encoder_type}")

    model_mapping = {
        "s": dinov2_vits14,
        "b": dinov2_vitb14,
        "l": dinov2_vitl14,
        "g": dinov2_vitg14,
    }
    model_size_key = model_config[0].lower()
    if model_size_key not in model_mapping:
        raise ValueError(f"Unsupported DINOv2 model config: {filename}")

    patch_size = _parse_patch_size(model_config)
    encoder = model_mapping[model_size_key](pretrained=False)

    checkpoint = torch.load(encoder_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    encoder.load_state_dict(state_dict, strict=True)

    encoder.head = torch.nn.Identity()
    encoder = encoder.to(device).eval()
    return encoder, encoder_type, architecture, patch_size


_PREPROCESS_CONFIGS = {
    "dinov2": {
        "interpolation": InterpolationMode.BICUBIC,
    },
}


def preprocess_image(x, encoder_type=None, patch_size=None):
    if encoder_type is None:
        return x
    if encoder_type not in _PREPROCESS_CONFIGS:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    preprocess_cfg = _PREPROCESS_CONFIGS[encoder_type]
    target_size = _resolve_preprocess_size(x.shape[-2:], encoder_type=encoder_type, patch_size=patch_size)
    if tuple(int(dim) for dim in x.shape[-2:]) != target_size:
        resize_transform = transforms.Resize(
            size=target_size,
            interpolation=preprocess_cfg["interpolation"],
            antialias=None,
        )
        x = resize_transform(x)
    return x
