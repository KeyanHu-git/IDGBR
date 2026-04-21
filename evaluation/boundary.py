from typing import Dict, List, Sequence
import shutil

import numpy as np

from evaluation.edge_generator import extract_edge_map_path
from evaluation.io import find_pred_path, list_images, read_image
from evaluation.wfmeasure import MultiClassWFmeasure


def _binarize(array: np.ndarray, mode: str) -> np.ndarray:
    if array.ndim > 2:
        array = array[:, :, 0]
    if mode == "lt255":
        return (array < 255).astype(np.uint8)
    if mode == "gt128":
        return (array > 128).astype(np.uint8)
    if mode == "binary":
        return (array > 0).astype(np.uint8)
    raise ValueError(f"Unknown binarize mode: {mode}")


def ensure_edge_maps(input_dir: str, edge_size: int, overwrite: bool = False) -> str:
    edge_dir = input_dir + f"_edge_map{edge_size}"
    extract_edge_map_path(input_dir, edge_size=edge_size, output_dir=edge_dir, overwrite=overwrite)
    return edge_dir


def compute_wfm_for_edge_size(
    pred_dir: str,
    label_dir: str,
    edge_size: int,
    pred_exts: Sequence[str],
    label_exts: Sequence[str],
    binarize_mode: str = "lt255",
    cleanup: bool = False,
    overwrite: bool = False,
) -> Dict[str, object]:
    pred_edge_dir = ensure_edge_maps(pred_dir, edge_size=edge_size, overwrite=overwrite)
    label_edge_dir = ensure_edge_maps(label_dir, edge_size=edge_size, overwrite=overwrite)

    try:
        label_files = list_images(label_edge_dir, label_exts)
        scores: Dict[int, float] = {}
        counts: Dict[int, int] = {}
        missing = 0

        for label_name in label_files:
            base_name, _ = label_name.rsplit(".", 1)
            pred_path = find_pred_path(pred_edge_dir, base_name, pred_exts)
            if pred_path is None:
                missing += 1
                continue
            label_path = f"{label_edge_dir}/{label_name}"
            label_img = _binarize(read_image(label_path), binarize_mode)
            pred_img = _binarize(read_image(pred_path), binarize_mode)

            wf_scores, wf_classes = MultiClassWFmeasure(pred_img, label_img)
            for score, cls in zip(wf_scores, wf_classes):
                scores[cls] = scores.get(cls, 0.0) + float(score)
                counts[cls] = counts.get(cls, 0) + 1

        class_ids = sorted(scores.keys())
        per_class = []
        for cls in class_ids:
            if counts.get(cls, 0) == 0:
                per_class.append(0.0)
            else:
                per_class.append(scores[cls] / counts[cls])

        valid_scores = [s for s in per_class if s > 0]
        mean_wf = float(np.mean(valid_scores)) if valid_scores else 0.0
        min_wf = float(np.min(valid_scores)) if valid_scores else 0.0

        return {
            "edge_size": edge_size,
            "class_ids": class_ids,
            "per_class_wf": per_class,
            "mean_wf": mean_wf,
            "min_wf": min_wf,
            "missing_predictions": missing,
        }
    finally:
        if cleanup:
            shutil.rmtree(pred_edge_dir, ignore_errors=True)
