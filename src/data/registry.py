from __future__ import annotations

from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T", bound=type)


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._modules: Dict[str, type] = {}

    def register_module(self, name: Optional[str] = None) -> Callable[[T], T]:
        def _register(cls: T) -> T:
            key = name or cls.__name__
            if key in self._modules:
                raise KeyError(f"{key} is already registered in {self.name}")
            self._modules[key] = cls
            return cls

        return _register

    def get(self, key: str) -> Optional[type]:
        return self._modules.get(key)

    def build(self, cfg: Dict[str, Any], **kwargs: Any) -> Any:
        if "type" not in cfg:
            raise KeyError("cfg['type'] is required")
        obj_type = cfg["type"]
        if not isinstance(obj_type, str):
            raise TypeError("cfg['type'] must be a str")
        cls = self.get(obj_type)
        if cls is None:
            raise KeyError(f"{obj_type} is not registered in {self.name}")
        init_kwargs = dict(cfg)
        init_kwargs.pop("type")
        init_kwargs.update(kwargs)
        return cls(**init_kwargs)


DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")
