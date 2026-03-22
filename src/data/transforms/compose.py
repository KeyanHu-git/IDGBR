from .registry import TRANSFORMS, BaseTransform


@TRANSFORMS.register_module()
class Compose(BaseTransform):
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, results: dict) -> dict:
        for transform in self.transforms:
            results = transform(results)
            if results is None:
                return None
        return results


def build_pipeline(cfg_list):
    if not cfg_list:
        return None
    ops = [TRANSFORMS.build(cfg) for cfg in cfg_list]
    return TRANSFORMS.build({"type": "Compose", "transforms": ops})


__all__ = ["Compose", "build_pipeline"]
