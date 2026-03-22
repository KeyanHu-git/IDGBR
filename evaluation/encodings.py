from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass
class EncodingSpec:
    num_classes: int
    label_values: Optional[Sequence[int]] = None
    pred_values: Optional[Sequence[int]] = None
    ignore_values: Sequence[int] = field(default_factory=list)

class ValueEncoder:
    def __init__(
        self,
        num_classes: int,
        values: Optional[Sequence[int]] = None,
        ignore_values: Optional[Sequence[int]] = None,
    ):
        self.num_classes = int(num_classes)
        self.ignore_values = set(ignore_values or [])
        self.values = list(values) if values is not None else None
        self._value_to_index = None
        if self.values is not None:
            self._value_to_index = {v: i for i, v in enumerate(self.values)}

    def encode(self, array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.num_classes == 2 and (self.values is None or set(int(v) for v in self.values).issubset({0, 1, 255})):
            if array.ndim > 2:
                array = array.max(axis=-1)
            array = (array > 0).astype(np.uint8)
        elif array.ndim > 2:
            array = array[:, :, 0]
        array = array.astype(np.int64, copy=False)

        if self._value_to_index is None:
            encoded = array
            mask = (encoded >= 0) & (encoded < self.num_classes)
            if self.ignore_values:
                mask &= ~np.isin(encoded, list(self.ignore_values))
            return encoded, mask

        encoded = np.full(array.shape, fill_value=-1, dtype=np.int64)
        for raw_value, mapped_index in self._value_to_index.items():
            encoded[array == raw_value] = mapped_index
        mask = encoded >= 0
        return encoded, mask


def build_encoders(spec: EncodingSpec) -> Tuple[ValueEncoder, ValueEncoder]:
    label_encoder = ValueEncoder(
        num_classes=spec.num_classes,
        values=spec.label_values,
        ignore_values=spec.ignore_values,
    )
    pred_encoder = ValueEncoder(
        num_classes=spec.num_classes,
        values=spec.pred_values,
        ignore_values=spec.ignore_values,
    )
    return label_encoder, pred_encoder
