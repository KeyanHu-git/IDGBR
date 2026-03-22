import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms


class Dataset_i2s(data.Dataset):
    def __init__(
        self,
        path: str,
        num_samples_to_use: int = None,
        resolution: int = 512,
        train=True,
        load_rough=False,
        metadata_file="metadata_i2s.jsonl",
    ):
        self.path = path
        self.resolution = resolution
        self.train = train
        self.load_rough = load_rough

        with open(Path(self.path, metadata_file), "r") as f:
            lines = f.readlines()
            raw_datas = [json.loads(line) for line in lines]
        if num_samples_to_use is not None:
            self.raw_datas = raw_datas[:num_samples_to_use]
        else:
            self.raw_datas = raw_datas
        self.is_binary_task = False
        if self.train:
            self.is_binary_task = self._is_binary_task()
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _is_binary_task(self):
        label_values = set()
        for item in self.raw_datas:
            label = np.array(Image.open(Path(self.path, item["label_index"])))
            label_values.update(np.unique(label).tolist())
        return label_values.issubset({0, 1, 255}) or (
            0 in label_values and len(label_values) == 2
        )

    def __len__(self) -> int:
        return len(self.raw_datas)

    def __getitem__(self, i: int) -> dict:
        data = dict()
        diction = self.raw_datas[i]

        img_path = diction["image"]
        img_name = os.path.basename(img_path)

        data["image"] = np.array(Image.open(Path(self.path, diction["image"])).convert("RGB").resize([512,512], Image.BILINEAR))
        if self.train:
            data["label_index"] = np.array(Image.open(Path(self.path, diction["label_index"])).resize([512,512], Image.NEAREST))
            if self.is_binary_task:
                data["label_index"] = np.where(data["label_index"] == 0, 0, 1)

        if self.load_rough:
            data["rough_label_index"] = np.array(Image.open(Path(self.path, diction["rough_label_index"])).resize([512,512], Image.NEAREST))
            if self.is_binary_task:
                data["rough_label_index"] = np.where(data["rough_label_index"] == 0, 0, 1)

        if self.train:
            input_images = list(data.values())
            output_image = join_transform(input_images)
            keys = list(data.keys())
            for i in range(len(keys)):
                data[keys[i]] = output_image[i]

        image_transforms = self.image_transforms
        data["image"] = image_transforms(Image.fromarray(data["image"]))
        
        if self.train:
            data["label_index"] = torch.from_numpy(np.ascontiguousarray(data["label_index"])).long()
        if self.load_rough:
            data["rough_label_index"] = torch.from_numpy(np.ascontiguousarray(data["rough_label_index"])).long()
            
        data["img_name"] = img_name

        return data

def join_transform(images: list):
    f = [1, 0, -1, 2, 2][np.random.randint(0, 5)]
    if f != 2:
        for i in range(len(images)):
            images[i] = filp_array(images[i], f)
    k = np.random.randint(0, 4)
    for i in range(len(images)):
        images[i] = np.rot90(images[i], k, (1, 0))

    return images


def filp_array(array, flipCode):
    if flipCode != -1:
        array = np.flip(array, flipCode)
    elif flipCode == -1:
        array = np.flipud(array)
        array = np.fliplr(array)
    return array


def colour_code_label(label, label_values):
    label, colour_codes = np.array(label), np.array(label_values)
    if len(label.shape) == 3:
        label = np.argmax(label, axis=2)
    color_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    mask = label < len(colour_codes)
    color_label[mask] = colour_codes[label[mask].astype(int)]
    return color_label

color_list = [
    [255, 0, 0],
    [255, 0, 255],
    [0, 0, 255],
    [0, 255, 255],
    [0, 255, 0],
    [255, 255, 0],
]
