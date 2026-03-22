from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ConfusionMatrixMeter:
    num_classes: int

    def __post_init__(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, label: np.ndarray, pred: np.ndarray, mask: np.ndarray):
        if not np.any(mask):
            return
        label_flat = label[mask].astype(np.int64)
        pred_flat = pred[mask].astype(np.int64)
        encoded = self.num_classes * label_flat + pred_flat
        bincount = np.bincount(encoded, minlength=self.num_classes ** 2)
        self.matrix += bincount.reshape(self.num_classes, self.num_classes)

    def total(self) -> int:
        return int(self.matrix.sum())


def compute_metrics(confm: np.ndarray) -> Dict[str, object]:
    confm = confm.astype(np.float64)
    total = confm.sum()
    if total == 0:
        return {
            "per_class_iou": [],
            "per_class_f1": [],
            "per_class_precision": [],
            "per_class_recall": [],
            "miou": 0.0,
            "mf1": 0.0,
            "oa": 0.0,
            "kappa": 0.0,
            "fw_iou": 0.0,
            "support": [],
        }

    intersection = np.diag(confm)
    gt_sum = confm.sum(axis=1)
    pred_sum = confm.sum(axis=0)
    union = gt_sum + pred_sum - intersection

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        precision = np.divide(intersection, pred_sum, out=np.zeros_like(intersection), where=pred_sum > 0)
        recall = np.divide(intersection, gt_sum, out=np.zeros_like(intersection), where=gt_sum > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(intersection), where=(precision + recall) > 0)

    miou = float(np.mean(iou))
    mf1 = float(np.mean(f1))
    oa = float(intersection.sum() / total)

    pa = intersection.sum() / total
    pe = (pred_sum * gt_sum).sum() / (total ** 2)
    kappa = float((pa - pe) / (1 - pe + 1e-10))

    cls_weight = np.divide(gt_sum, total, out=np.zeros_like(gt_sum), where=total > 0)
    fw_iou = float(np.sum(iou * cls_weight))

    return {
        "per_class_iou": iou.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "miou": miou,
        "mf1": mf1,
        "oa": oa,
        "kappa": kappa,
        "fw_iou": fw_iou,
        "support": gt_sum.tolist(),
    }
