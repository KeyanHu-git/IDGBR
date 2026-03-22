from .registry import DATASETS, TRANSFORMS
from . import datasets


def build_dataset(cfg, **kwargs):
    return DATASETS.build(cfg, **kwargs)


__all__ = ["DATASETS", "TRANSFORMS", "build_dataset"]
