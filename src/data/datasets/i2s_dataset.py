import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils import data

from ..registry import DATASETS
from ..transforms import build_pipeline


@DATASETS.register_module(name="I2SDataset")
class I2SDataset(data.Dataset):
    def __init__(
        self,
        path: str,
        num_samples_to_use: int = None,
        num_classes: int = None,
        train: bool = True,
        load_rough=False,
        load_text: bool = False,
        metadata_file: str = "metadata_i2s.jsonl",
        pipeline=None,
    ):
        self.path = Path(path)
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.train = bool(train)
        self.load_rough = load_rough
        self.load_text = bool(load_text)

        with open(self.path / metadata_file, "r") as f:
            raw_datas = [json.loads(line) for line in f]
        if num_samples_to_use is not None:
            self.raw_datas = raw_datas[:num_samples_to_use]
        else:
            self.raw_datas = raw_datas

        self.pipeline = build_pipeline(pipeline)
        self.is_binary_task = self._detect_binary_task()

    def _detect_binary_task(self):
        if self.num_classes is not None:
            return self.num_classes == 2

        label_key = None
        if any("label_index" in item for item in self.raw_datas):
            label_key = "label_index"
        elif self.load_rough and any("rough_label_index" in item for item in self.raw_datas):
            label_key = "rough_label_index"

        if label_key is None:
            return False

        label_values = set()
        for item in self.raw_datas:
            label_path = item.get(label_key)
            if label_path is None:
                continue
            label = np.array(Image.open(self.path / label_path))
            label_values.update(np.unique(label).tolist())
            if len(label_values) > 3:
                return False

        if not label_values:
            return False
        return set(int(v) for v in label_values).issubset({0, 1, 255})

    def _load_image(self, rel_path: str):
        image = Image.open(self.path / rel_path).convert("RGB")
        return np.array(image)

    def _load_label(self, rel_path: str):
        label = Image.open(self.path / rel_path)
        return np.array(label)

    def __len__(self) -> int:
        return len(self.raw_datas)

    def __getitem__(self, i: int) -> dict:
        sample = self.raw_datas[i]
        data_dict = {"image": self._load_image(sample["image"])}

        if self.train:
            data_dict["label_index"] = self._load_label(sample["label_index"])
        if self.load_rough:
            data_dict["rough_label_index"] = self._load_label(sample["rough_label_index"])
        if self.load_text:
            data_dict["text"] = sample.get("text", "")
        if self.is_binary_task:
            for key in ("label_index", "rough_label_index"):
                if key in data_dict:
                    data_dict[key] = (data_dict[key] != 0).astype(np.int64)

        if self.pipeline is not None:
            data_dict = self.pipeline(data_dict)

        data_dict["img_name"] = os.path.basename(sample["image"])
        return data_dict
