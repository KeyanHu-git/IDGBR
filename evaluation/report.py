import json
import os
from typing import Dict, List, Optional


def format_metrics_text(metrics: Dict[str, object], num_classes: int) -> str:
    lines = []
    lines.append(f"num_classes: {num_classes}")
    lines.append("per_class_iou:")
    for i, v in enumerate(metrics.get("per_class_iou", [])):
        lines.append(f"  class {i}: {v:.6f}")
    lines.append("per_class_f1:")
    for i, v in enumerate(metrics.get("per_class_f1", [])):
        lines.append(f"  class {i}: {v:.6f}")
    lines.append(f"mIoU: {metrics.get('miou', 0.0):.6f}")
    lines.append(f"mF1: {metrics.get('mf1', 0.0):.6f}")
    lines.append(f"OA: {metrics.get('oa', 0.0):.6f}")
    lines.append(f"Kappa: {metrics.get('kappa', 0.0):.6f}")
    lines.append(f"FW IoU: {metrics.get('fw_iou', 0.0):.6f}")
    return "\n".join(lines)


def format_boundary_text(boundary: List[Dict[str, object]]) -> str:
    if not boundary:
        return ""
    lines = []
    lines.append("")
    lines.append("Boundary WFM:")
    for item in boundary:
        edge_size = item.get("edge_size")
        class_ids = item.get("class_ids", [])
        per_class = item.get("per_class_wf", [])
        mean_wf = item.get("mean_wf", 0.0)
        lines.append(f"  edge_size={edge_size}")
        for cls, wf in zip(class_ids, per_class):
            lines.append(f"    class {cls}: {wf:.6f}")
        lines.append(f"    mean: {mean_wf:.6f}")
    return "\n".join(lines)


def format_iou_f1_wf_block(
    name: str,
    metrics: Dict[str, object],
    boundary: List[Dict[str, object]],
    class_offset: int = 0,
    wfm_edge: int = 3,
    wfm_aggregate: str = "min",
) -> str:
    per_iou = metrics.get("per_class_iou", [])
    per_f1 = metrics.get("per_class_f1", [])
    lines = []
    lines.append(f"Evaluation Result for: {name}")
    lines.append("-" * 65)
    lines.append("Class      | IoU             | F1-Score       ")
    lines.append("-" * 65)
    for idx, (iou, f1) in enumerate(zip(per_iou, per_f1)):
        class_id = idx + class_offset
        lines.append(f"{class_id:<10} | {iou:.4f}{' ' * 10} | {f1:.4f}")
    lines.append("-" * 65)
    lines.append(f"mIoU       | {metrics.get('miou', 0.0):.4f}")
    lines.append(f"Mean F1    | {metrics.get('mf1', 0.0):.4f}")

    wfm_value: Optional[float] = None
    for item in boundary:
        if int(item.get("edge_size", -1)) == int(wfm_edge):
            if wfm_aggregate == "mean":
                wfm_value = item.get("mean_wf", 0.0)
            else:
                wfm_value = item.get("min_wf", 0.0)
            break
    if wfm_value is not None:
        lines.append(f"WF ({wfm_edge}px)   | {wfm_value:.4f}")
    lines.append("=" * 65)
    return "\n".join(lines)


def format_recall_precision_block(name: str, label_dir: str, pred_dir: str, metrics: Dict[str, object]) -> str:
    recall = metrics.get("per_class_recall", [])
    precision = metrics.get("per_class_precision", [])
    num_classes = len(recall)
    lines = []
    lines.append(f"=== Evaluation Report for {name} ===")
    lines.append(f"Label Dir: {label_dir}")
    lines.append(f"Pred Dir:  {pred_dir}")
    lines.append("-" * 65)
    lines.append("Class      | Recall       | Precision   ")
    lines.append("-" * 65)
    for idx in range(num_classes):
        r = recall[idx] if idx < len(recall) else 0.0
        p = precision[idx] if idx < len(precision) else 0.0
        lines.append(f"{idx:<10} | {r:.4f}       | {p:.4f}")
    mean_recall = sum(recall) / len(recall) if recall else 0.0
    mean_precision = sum(precision) / len(precision) if precision else 0.0
    global_score = metrics.get("oa", 0.0)
    lines.append("-" * 65)
    lines.append(f"{'Mean':<10} | {mean_recall:.4f}       | {mean_precision:.4f}")
    lines.append(f"{'Global':<10} | {global_score:.4f}       | {global_score:.4f}")
    lines.append("=" * 65)
    return "\n".join(lines)


def save_metrics(output_dir: str, name: str, payload: Dict[str, object]):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{name}.json")
    txt_path = os.path.join(output_dir, f"{name}.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as f:
        text = format_metrics_text(payload["metrics"], payload["encoding"]["num_classes"])
        boundary_text = format_boundary_text(payload.get("boundary", []))
        f.write(text + boundary_text)


def save_summary(output_dir: str, summary: List[Dict[str, object]]):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "summary.json")
    txt_path = os.path.join(output_dir, "summary.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as f:
        for item in summary:
            name = item.get("name", "unknown")
            miou = item.get("metrics", {}).get("miou", 0.0)
            mf1 = item.get("metrics", {}).get("mf1", 0.0)
            oa = item.get("metrics", {}).get("oa", 0.0)
            f.write(f"{name}\tmiou={miou:.6f}\tmf1={mf1:.6f}\toa={oa:.6f}\n")
