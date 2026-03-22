import os
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


def list_images(root_dir: str, exts: Sequence[str]) -> List[str]:
    if not os.path.isdir(root_dir):
        return []
    exts_lower = {e.lower() for e in exts}
    files = []
    for name in os.listdir(root_dir):
        _, ext = os.path.splitext(name)
        if ext.lower() in exts_lower:
            files.append(name)
    files.sort()
    return files


def find_pred_path(pred_dir: str, base_name: str, exts: Sequence[str]) -> Optional[str]:
    for ext in exts:
        candidate = os.path.join(pred_dir, base_name + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def read_image(path: str) -> np.ndarray:
    return np.array(Image.open(path))
