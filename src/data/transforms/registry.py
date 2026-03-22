from ..registry import TRANSFORMS


class BaseTransform:
    def __call__(self, results: dict) -> dict:
        raise NotImplementedError


__all__ = ["TRANSFORMS", "BaseTransform"]
