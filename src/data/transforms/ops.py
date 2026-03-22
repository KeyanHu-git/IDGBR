import numpy as np
import torch
from PIL import Image

from .registry import TRANSFORMS, BaseTransform


@TRANSFORMS.register_module()
class ResizeI2S(BaseTransform):
    def __init__(self, size, keys=None):
        self.size = int(size) if isinstance(size, int) else tuple(size)
        self.keys = list(keys) if keys is not None else ["image", "label_index", "rough_label_index"]

    def __call__(self, results: dict) -> dict:
        target_size = (self.size, self.size) if isinstance(self.size, int) else tuple(self.size)
        for key in self.keys:
            if key not in results:
                continue
            array = results[key]
            if not isinstance(array, np.ndarray):
                continue
            if array.dtype == np.int64:
                array = array.astype(np.int32)
            resample = Image.BILINEAR if key == "image" else Image.NEAREST
            results[key] = np.array(Image.fromarray(array).resize(target_size, resample))
        return results


@TRANSFORMS.register_module()
class ToTensorI2S(BaseTransform):
    def __init__(self, image_key="image"):
        self.image_key = image_key

    def __call__(self, results: dict) -> dict:
        if self.image_key not in results:
            return results
        image = results[self.image_key]
        if isinstance(image, np.ndarray):
            image = np.ascontiguousarray(image.transpose(2, 0, 1))
            results[self.image_key] = torch.from_numpy(image).float().div(255.0)
        return results


@TRANSFORMS.register_module()
class NormalizeImage(BaseTransform):
    def __init__(self, mean, std, image_key="image"):
        self.image_key = image_key
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, results: dict) -> dict:
        if self.image_key not in results:
            return results
        image = results[self.image_key]
        if isinstance(image, torch.Tensor):
            results[self.image_key] = (image - self.mean) / self.std
        return results


@TRANSFORMS.register_module()
class ToLongTensorI2S(BaseTransform):
    def __init__(self, keys=None):
        self.keys = list(keys) if keys is not None else ["label_index", "rough_label_index"]

    def __call__(self, results: dict) -> dict:
        for key in self.keys:
            if key not in results:
                continue
            value = results[key]
            if isinstance(value, np.ndarray):
                results[key] = torch.from_numpy(np.ascontiguousarray(value)).long()
            elif isinstance(value, torch.Tensor):
                results[key] = value.long()
        return results


@TRANSFORMS.register_module()
class SerializeLabelI2S(BaseTransform):
    def __init__(self, keys=None):
        self.keys = list(keys) if keys is not None else ["label_index", "rough_label_index"]
        self.lookup_table = None

    def set_lookup_table(self, lookup_table):
        self.lookup_table = lookup_table

    def __call__(self, results: dict) -> dict:
        if self.lookup_table is None:
            return results
        for key in self.keys:
            if key not in results:
                continue
            value = results[key]
            if isinstance(value, np.ndarray):
                results[key] = self.lookup_table[value]
        return results
