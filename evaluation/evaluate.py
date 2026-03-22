import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional

import yaml

from evaluation.encodings import EncodingSpec, build_encoders
from evaluation.boundary import compute_wfm_for_edge_size
from evaluation.io import find_pred_path, list_images, read_image
from evaluation.metrics import ConfusionMatrixMeter, compute_metrics
from evaluation.report import (
    save_metrics,
    save_summary,
    format_recall_precision_block,
    format_iou_f1_wf_block,
)


def _resolve_path_value(value: object, config_dir: str) -> object:
    if not isinstance(value, str):
        return value
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(config_dir, value))


def _resolve_config_paths(config: Dict[str, object], config_path: str) -> Dict[str, object]:
    config_dir = os.path.dirname(os.path.abspath(config_path))
    report_cfg = config.get("report")
    if isinstance(report_cfg, dict):
        if "path" in report_cfg:
            report_cfg["path"] = _resolve_path_value(report_cfg["path"], config_dir)
    jobs = config.get("jobs")
    if isinstance(jobs, list):
        for job in jobs:
            if not isinstance(job, dict):
                continue
            for key in ("pred_dir", "label_dir", "output_dir"):
                if key in job:
                    job[key] = _resolve_path_value(job[key], config_dir)
    return config


def load_config(config_path: str) -> Dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.endswith(".json"):
            config = json.load(f)
        else:
            config = yaml.safe_load(f)
    if isinstance(config, dict):
        config = _resolve_config_paths(config, config_path)
    return config


def is_legacy_config(config: Dict[str, object]) -> bool:
    if not isinstance(config, dict):
        return False
    if not config:
        return False
    for value in config.values():
        if not isinstance(value, dict):
            return False
        if "classes" not in value or "label" not in value or "pred2" not in value:
            return False
    return True


def normalize_job(job: Dict[str, object]) -> Dict[str, object]:
    job = dict(job)
    job.setdefault("name", "eval_job")
    job.setdefault("pred_exts", [".png", ".tif", ".jpg"])
    job.setdefault("label_exts", [".png", ".tif", ".jpg"])
    job.setdefault("output_dir", "evaluation_results")
    job.setdefault("batch", {})
    batch_cfg = job["batch"] or {}
    job["batch"] = {
        "enabled": bool(batch_cfg.get("enabled", False)),
        "regex": batch_cfg.get("regex", r"^checkpoint-\d+$"),
    }
    enc_cfg = job.get("encoding", {})
    job["encoding"] = {
        "num_classes": enc_cfg.get("num_classes"),
        "label_values": enc_cfg.get("label_values"),
        "pred_values": enc_cfg.get("pred_values"),
        "ignore_values": enc_cfg.get("ignore_values", []),
    }
    boundary_cfg = job.get("boundary", {}) or {}
    job["boundary"] = {
        "enabled": bool(boundary_cfg.get("enabled", False)),
        "edge_sizes": boundary_cfg.get("edge_sizes", [3]),
        "binarize": boundary_cfg.get("binarize", "lt255"),
        "cleanup": bool(boundary_cfg.get("cleanup", True)),
        "overwrite": bool(boundary_cfg.get("overwrite", False)),
    }
    return job


def evaluate_directory(job: Dict[str, object], pred_dir: str, label_dir: str) -> Dict[str, object]:
    encoding = EncodingSpec(**job["encoding"])
    label_encoder, pred_encoder = build_encoders(encoding)
    confm_meter = ConfusionMatrixMeter(encoding.num_classes)

    label_files = list_images(label_dir, job["label_exts"])
    missing = 0
    for label_name in label_files:
        base_name, _ = os.path.splitext(label_name)
        pred_path = find_pred_path(pred_dir, base_name, job["pred_exts"])
        if pred_path is None:
            missing += 1
            continue
        label_path = os.path.join(label_dir, label_name)
        label_raw = read_image(label_path)
        pred_raw = read_image(pred_path)

        label_encoded, label_mask = label_encoder.encode(label_raw)
        pred_encoded, pred_mask = pred_encoder.encode(pred_raw)
        mask = label_mask & pred_mask
        confm_meter.update(label_encoded, pred_encoded, mask)

    metrics = compute_metrics(confm_meter.matrix)
    boundary_results = []
    if job["boundary"]["enabled"]:
        for edge_size in job["boundary"]["edge_sizes"]:
            boundary_results.append(
                compute_wfm_for_edge_size(
                    pred_dir=pred_dir,
                    label_dir=label_dir,
                    edge_size=int(edge_size),
                    pred_exts=job["pred_exts"],
                    label_exts=job["label_exts"],
                    binarize_mode=job["boundary"]["binarize"],
                    cleanup=job["boundary"]["cleanup"],
                    overwrite=job["boundary"]["overwrite"],
                )
            )
    payload = {
        "name": job["name"],
        "pred_dir": pred_dir,
        "label_dir": label_dir,
        "encoding": job["encoding"],
        "metrics": metrics,
        "boundary": boundary_results,
        "missing_predictions": missing,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return payload


def run_job(job: Dict[str, object]) -> List[Dict[str, object]]:
    job = normalize_job(job)
    pred_root = job["pred_dir"]
    label_dir = job["label_dir"]
    summary: List[Dict[str, object]] = []

    if job["batch"]["enabled"]:
        pattern = re.compile(job["batch"]["regex"])
        subdirs = [
            d for d in os.listdir(pred_root)
            if pattern.match(d) and os.path.isdir(os.path.join(pred_root, d))
        ]
        subdirs.sort()
        for sub in subdirs:
            pred_dir = os.path.join(pred_root, sub)
            payload = evaluate_directory(job, pred_dir, label_dir)
            payload["name"] = sub
            summary.append(payload)
        return summary

    payload = evaluate_directory(job, pred_root, label_dir)
    return [payload]


def run_legacy_config(config_path: str, config: Dict[str, object]):
    output_path = os.path.splitext(config_path)[0] + "_recall_pre.txt"
    lines = []
    lines.append("=== 批量评估报告 (Generated by Evaluation) ===")
    lines.append(f"Config: {os.path.basename(config_path)}")
    lines.append(f"Time: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    for name, info in config.items():
        pred_dir = info["pred2"][0]
        label_dir = info["label"][0]
        job = {
            "name": name,
            "pred_dir": pred_dir,
            "label_dir": label_dir,
            "output_dir": os.path.dirname(output_path) or ".",
            "encoding": {
                "num_classes": int(info["classes"]),
                "label_values": None,
                "pred_values": None,
                "ignore_values": [],
            },
            "pred_exts": [".png", ".tif", ".jpg"],
            "label_exts": [".png", ".tif", ".jpg"],
            "batch": {"enabled": False, "regex": r"^checkpoint-\\d+$"},
        }
        payload = evaluate_directory(job, pred_dir, label_dir)
        block = format_recall_precision_block(name, label_dir, pred_dir, payload["metrics"])
        lines.append(block)
        lines.append("")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_recall_precision_report(
    config_path: str,
    report_cfg: Dict[str, object],
    payloads: List[Dict[str, object]],
):
    report_path = report_cfg.get("path")
    if not report_path:
        report_path = os.path.splitext(config_path)[0] + "_recall_pre.txt"

    lines = []
    lines.append("=== 批量评估报告 (Generated by Evaluation) ===")
    lines.append(f"Config: {os.path.basename(config_path)}")
    method = report_cfg.get("method")
    if method:
        lines.append(f"Method: {method}")
    experiment = report_cfg.get("experiment")
    if experiment:
        lines.append(f"Experiment: {experiment}")
    lines.append(f"Time: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    for payload in payloads:
        block = format_recall_precision_block(
            payload.get("name", "eval_job"),
            payload.get("label_dir", ""),
            payload.get("pred_dir", ""),
            payload.get("metrics", {}),
        )
        lines.append(block)
        lines.append("")

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _infer_class_offset(payload: Dict[str, object]) -> int:
    encoding = payload.get("encoding", {})
    label_values = encoding.get("label_values")
    if isinstance(label_values, list) and label_values:
        try:
            min_val = min(int(v) for v in label_values)
            if min_val >= 1:
                return min_val
        except Exception:
            return 0
    return 0


def build_iou_f1_wf_report(
    config_path: str,
    report_cfg: Dict[str, object],
    payloads: List[Dict[str, object]],
):
    report_path = report_cfg.get("path")
    if not report_path:
        report_path = os.path.splitext(config_path)[0] + "_batch_report.txt"

    if len(payloads) == 1:
        path_line = payloads[0].get("pred_dir", "")
    else:
        path_line = report_cfg.get("root_path", "")

    wfm_edge = int(report_cfg.get("wfm_edge", 3))
    wfm_aggregate = report_cfg.get("wfm_aggregate", "min")
    class_offset_override = report_cfg.get("class_offset")

    lines = []
    lines.append("Batch Evaluation Report")
    lines.append(f"Time: {time.strftime('%Y-%m-%d %H:%M')}")
    experiment = report_cfg.get("experiment")
    if experiment:
        lines.append(f"Experiment: {experiment}")
    lines.append(f"Path: {path_line}")
    lines.append("=" * 65)
    lines.append("")

    for payload in payloads:
        class_offset = class_offset_override
        if class_offset is None:
            class_offset = _infer_class_offset(payload)
        block = format_iou_f1_wf_block(
            payload.get("name", "eval_job"),
            payload.get("metrics", {}),
            payload.get("boundary", []),
            class_offset=int(class_offset),
            wfm_edge=wfm_edge,
            wfm_aggregate=wfm_aggregate,
        )
        lines.append(block)
        lines.append("")

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified evaluation runner")
    parser.add_argument("--config", type=str, default=None, help="Path to eval config (yaml/json)")
    parser.add_argument("--pred_dir", type=str, default=None, help="Prediction directory")
    parser.add_argument("--label_dir", type=str, default=None, help="Label directory")
    parser.add_argument("--num_classes", type=int, default=None, help="Number of classes")
    parser.add_argument("--label_values", type=str, default=None, help="Comma-separated label values")
    parser.add_argument("--pred_values", type=str, default=None, help="Comma-separated pred values")
    parser.add_argument("--ignore_values", type=str, default=None, help="Comma-separated ignore values")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory")
    return parser.parse_args()


def parse_values(text: Optional[str]) -> Optional[List[int]]:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    return [int(x) for x in text.split(",") if x.strip()]


def run_single_from_args(args: argparse.Namespace):
    if args.pred_dir is None or args.label_dir is None or args.num_classes is None:
        raise ValueError("pred_dir, label_dir, num_classes are required when --config is not used.")

    job = {
        "name": "eval_job",
        "pred_dir": args.pred_dir,
        "label_dir": args.label_dir,
        "output_dir": args.output_dir,
        "encoding": {
            "num_classes": args.num_classes,
            "label_values": parse_values(args.label_values),
            "pred_values": parse_values(args.pred_values),
            "ignore_values": parse_values(args.ignore_values) or [],
        },
    }
    run_job(job)


def main():
    args = parse_args()
    if args.config:
        config = load_config(args.config)
        if is_legacy_config(config):
            run_legacy_config(args.config, config)
            return
        if "jobs" in config:
            all_payloads: List[Dict[str, object]] = []
            for job in config["jobs"]:
                all_payloads.extend(run_job(job))
        else:
            all_payloads = run_job(config)

        report_cfg = config.get("report") if isinstance(config, dict) else None
        if isinstance(report_cfg, dict):
            if report_cfg.get("type") == "recall_precision":
                build_recall_precision_report(args.config, report_cfg, all_payloads)
            elif report_cfg.get("type") == "iou_f1_wf":
                build_iou_f1_wf_report(args.config, report_cfg, all_payloads)
        return
    run_single_from_args(args)


if __name__ == "__main__":
    main()
