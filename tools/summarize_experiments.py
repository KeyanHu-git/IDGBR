from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CURRENT_EXPERIMENTS = ROOT / "experiments"
TRASH_EXPERIMENTS = ROOT / "trash" / "experiments"
CURRENT_EVALS = ROOT / "evaluation_results"
TRASH_EVALS = ROOT / "trash" / "evaluation_results"
CURRENT_CONFIGS = ROOT / "configs" / "experiments"
TRASH_CONFIGS = ROOT / "trash" / "configs" / "experiments"
CURRENT_BASE_CONFIG = ROOT / "configs" / "base_config.yaml"
OUTPUT_PATH = ROOT / "EXPERIMENTS_SUMMARY.md"

PROGRESS_RE = re.compile(r"80000/80000 \[(\d+:\d+:\d+)<00:00")
START_RE = re.compile(r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}) - INFO - __main__ - \*\*\*\*\* Running training \*\*\*\*\*")
END_RE = re.compile(r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}) - INFO - __main__ - Saved state to .*checkpoint-80000")
MIOU_RE = re.compile(r"mIoU\s*\|\s*([0-9.]+)")
MEAN_F1_RE = re.compile(r"Mean F1\s*\|\s*([0-9.]+)")
WFM_RE = re.compile(r"WF \(3px\)\s*\|\s*([0-9.]+)")
TIME_RE = re.compile(r"Time:\s*(.+)")
PATH_RE = re.compile(r"Path:\s*(.+)")


@dataclass
class Row:
    dataset: str
    experiment: str
    source: str
    exp_dir: Path | None = None
    eval_file: Path | None = None
    runtime: str = "—"
    runtime_exact: bool = False
    status: str = ""
    rough: str = "—"
    aug: str = "—"
    align: str = "—"
    cubic: str = "—"
    sampling: str = "—"
    miou: str = "—"
    mean_f1: str = "—"
    wfm: str = "—"
    note: str = ""
    eval_time: str | None = None
    eval_path: str | None = None
    raw_config: dict[str, Any] = field(default_factory=dict)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = read_text(path)
    try:
        data = yaml.safe_load(text)
        return data or {}
    except yaml.YAMLError:
        data: dict[str, Any] = {}
        for key in ("_base_", "output_dir", "dataset", "enable_alignment", "time_sample_strategy", "load_rough"):
            match = re.search(rf"^{re.escape(key)}\s*:\s*(.+?)\s*$", text, re.MULTILINE)
            if not match:
                continue
            raw = match.group(1).strip()
            if raw.lower() == "true":
                data[key] = True
            elif raw.lower() == "false":
                data[key] = False
            elif raw.startswith('"') and raw.endswith('"'):
                data[key] = raw[1:-1]
            elif raw.startswith("'") and raw.endswith("'"):
                data[key] = raw[1:-1]
            else:
                data[key] = raw
        return data


def merge_config(path: Path) -> dict[str, Any]:
    data = load_yaml(path)
    base_rel = data.pop("_base_", None)
    if not base_rel:
        return data
    base_path = (path.parent / base_rel).resolve()
    if not base_path.exists():
        base_path = CURRENT_BASE_CONFIG
    merged = merge_config(base_path)
    merged.update(data)
    return merged


def fmt_seconds(seconds: float) -> str:
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_dt(text: str) -> datetime:
    return datetime.strptime(text, "%m/%d/%Y %H:%M:%S")


def find_exact_runtime(exp_dir: Path) -> str | None:
    candidates: list[tuple[float, str]] = []
    for log_file in exp_dir.rglob("*.log"):
        text = read_text(log_file)
        matches = PROGRESS_RE.findall(text)
        if matches:
            candidates.append((log_file.stat().st_mtime, matches[-1]))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def find_log_window_runtime(exp_dir: Path) -> str | None:
    candidates: list[tuple[float, str]] = []
    for log_file in exp_dir.rglob("*.log"):
        text = read_text(log_file)
        starts = START_RE.findall(text)
        ends = END_RE.findall(text)
        if starts and ends:
            start_dt = parse_dt(starts[-1])
            end_dt = parse_dt(ends[-1])
            if end_dt >= start_dt:
                candidates.append((log_file.stat().st_mtime, fmt_seconds((end_dt - start_dt).total_seconds())))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def estimate_runtime(exp_dir: Path) -> str | None:
    cp40 = exp_dir / "checkpoint-40000"
    cp80 = exp_dir / "checkpoint-80000"
    if cp40.exists() and cp80.exists():
        return "~" + fmt_seconds((cp80.stat().st_mtime - cp40.stat().st_mtime) * 2)
    if cp80.exists():
        starts = sorted(p for p in exp_dir.iterdir() if p.name.startswith("code_backup_"))
        if starts:
            return "~" + fmt_seconds(cp80.stat().st_mtime - starts[-1].stat().st_mtime)
    return None


def parse_eval_metrics(path: Path) -> dict[str, str]:
    text = read_text(path)
    result: dict[str, str] = {}
    miou = MIOU_RE.search(text)
    mean_f1 = MEAN_F1_RE.search(text)
    wfm = WFM_RE.search(text)
    timestamp = TIME_RE.search(text)
    eval_path = PATH_RE.search(text)
    if miou:
        result["miou"] = miou.group(1)
    if mean_f1:
        result["mean_f1"] = mean_f1.group(1)
    if wfm:
        result["wfm"] = wfm.group(1)
    if timestamp:
        result["time"] = timestamp.group(1).strip()
    if eval_path:
        result["path"] = eval_path.group(1).strip()
    return result


def config_index() -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for source, base in (("current", CURRENT_CONFIGS), ("trash", TRASH_CONFIGS)):
        if not base.exists():
            continue
        for cfg_path in sorted(base.rglob("*.yaml")):
            merged = merge_config(cfg_path)
            output_dir = merged.get("output_dir")
            if not output_dir:
                continue
            exp_name = Path(str(output_dir)).name
            dataset = cfg_path.parent.name
            index[(source, dataset, exp_name)] = {
                "path": cfg_path,
                "merged": merged,
            }
    return index


def parse_dataset_flags(cfg_entry: dict[str, Any]) -> tuple[str, str]:
    cfg_path = cfg_entry["path"]
    merged = cfg_entry["merged"]
    dataset_rel = merged.get("dataset")
    if not dataset_rel:
        return "—", "—"
    dataset_path = (cfg_path.parent / dataset_rel).resolve()
    data = load_yaml(dataset_path)
    load_rough = data.get("load_rough")
    pipeline = data.get("pipeline", [])
    rough = "On" if load_rough else "Off"
    aug = "On" if any(step.get("type") == "LegacySpatialAugment" for step in pipeline) else "Off"
    return rough, aug


def apply_config_flags(row: Row, cfg_entry: dict[str, Any]) -> None:
    merged = cfg_entry["merged"]
    rough, aug = parse_dataset_flags(cfg_entry)
    row.rough = rough
    row.aug = aug
    row.align = "On" if merged.get("enable_alignment", True) else "Off"
    row.cubic = "On" if merged.get("time_sample_strategy", False) else "Off"
    row.raw_config = merged


def apply_heuristics(row: Row) -> None:
    name = row.experiment
    if name in {"segformer", "segformer_rough_labels"}:
        row.rough = "Baseline"
        row.aug = "—"
        row.align = "—"
        row.cubic = "—"
        row.sampling = "—"
        if not row.note:
            row.note = "rough_labels baseline"
        return
    row.rough = "Off" if "no_rough" in name else "On"
    row.aug = "Off" if row.source == "current" else "On"
    row.align = "Off" if "no_align" in name else "On"
    if "no_cubic" in name:
        row.cubic = "Off"
    elif "cubic" in name:
        row.cubic = "On"
    else:
        row.cubic = "Off" if row.dataset == "CHN6-CUG" else "On"


def apply_name_overrides(row: Row) -> None:
    name = row.experiment
    if "no_rough" in name:
        row.rough = "Off"
    if "no_align" in name:
        row.align = "Off"
    if "no_cubic" in name:
        row.cubic = "Off"
    elif "cubic" in name:
        row.cubic = "On"


def canonical_base_name(name: str) -> str | None:
    match = re.match(r"(.+?)_rand[a-z0-9]+$", name)
    if match:
        return match.group(1)
    return None


def gather_rows() -> dict[tuple[str, str, str], Row]:
    rows: dict[tuple[str, str, str], Row] = {}
    for source, base in (("current", CURRENT_EXPERIMENTS), ("trash", TRASH_EXPERIMENTS)):
        if not base.exists():
            continue
        for dataset_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            for exp_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
                key = (source, dataset_dir.name, exp_dir.name)
                rows[key] = Row(dataset=dataset_dir.name, experiment=exp_dir.name, source=source, exp_dir=exp_dir)
    for source, base in (("current", CURRENT_EVALS), ("trash", TRASH_EVALS)):
        if not base.exists():
            continue
        for dataset_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            for eval_file in sorted(dataset_dir.glob("*.txt")):
                exp_name = eval_file.stem
                key = (source, dataset_dir.name, exp_name)
                row = rows.get(key)
                if row is None:
                    row = Row(dataset=dataset_dir.name, experiment=exp_name, source=source)
                    rows[key] = row
                row.eval_file = eval_file
    return rows


def source_label(source: str) -> str:
    return "current" if source == "current" else "trash"


def status_order(status: str) -> int:
    order = {
        "Evaluated": 0,
        "Eval-only": 1,
        "Train only": 2,
        "Failed/unfinished": 3,
        "Baseline": 4,
        "Alias/ambiguous": 5,
    }
    return order.get(status, 9)


def finalize_rows(rows: dict[tuple[str, str, str], Row]) -> list[Row]:
    cfgs = config_index()
    for key, row in rows.items():
        source, dataset, exp_name = key
        cfg_entry = cfgs.get(key)
        base_name = canonical_base_name(exp_name)
        if cfg_entry:
            apply_config_flags(row, cfg_entry)
        elif base_name and (source, dataset, base_name) in cfgs:
            apply_config_flags(row, cfgs[(source, dataset, base_name)])
        else:
            apply_heuristics(row)
        apply_name_overrides(row)

        if row.eval_file:
            metrics = parse_eval_metrics(row.eval_file)
            row.miou = metrics.get("miou", "—")
            row.mean_f1 = metrics.get("mean_f1", "—")
            row.wfm = metrics.get("wfm", "—")
            row.eval_time = metrics.get("time")
            row.eval_path = metrics.get("path")

        if base_name:
            row.sampling = "Random"
            if not row.note:
                row.note = f"same train as {base_name}"
        elif row.eval_file and row.rough != "Baseline":
            row.sampling = "Default"

        if row.experiment in {"segformer", "segformer_rough_labels"}:
            row.status = "Baseline"
        elif row.experiment == "segformer_no_align_002":
            row.status = "Alias/ambiguous"
            row.note = "legacy misnamed artifact"
            row.rough = "?"
            row.aug = "?"
            row.align = "?"
            row.cubic = "?"
            row.sampling = "?"
        elif row.eval_file and row.exp_dir:
            row.status = "Evaluated"
        elif row.eval_file and not row.exp_dir:
            row.status = "Eval-only"
        elif row.exp_dir and (row.exp_dir / "checkpoint-80000").exists():
            row.status = "Train only"
        elif row.exp_dir:
            row.status = "Failed/unfinished"
        else:
            row.status = "Alias/ambiguous"

        if row.exp_dir:
            runtime = find_exact_runtime(row.exp_dir)
            if runtime:
                row.runtime = runtime
                row.runtime_exact = True
            else:
                runtime = find_log_window_runtime(row.exp_dir)
                if runtime:
                    row.runtime = runtime
                    row.runtime_exact = True
                else:
                    runtime = estimate_runtime(row.exp_dir)
                    if runtime:
                        row.runtime = runtime

    row_map = {(row.source, row.dataset, row.experiment): row for row in rows.values()}
    for row in rows.values():
        base_name = canonical_base_name(row.experiment)
        if base_name and row.runtime == "—":
            base_row = row_map.get((row.source, row.dataset, base_name))
            if base_row is not None:
                row.runtime = base_row.runtime

        if row.experiment == "segformer":
            row.note = "same path as segformer_rough_labels"
        elif row.experiment == "segformer_rough_labels":
            row.note = "same path as segformer"
        elif row.exp_dir and not row.eval_file and row.note == "":
            row.note = "no eval file"

    ordered = sorted(
        rows.values(),
        key=lambda row: (
            row.dataset,
            0 if row.source == "current" else 1,
            status_order(row.status),
            row.experiment,
        ),
    )
    return ordered


def render_table(rows: list[Row]) -> str:
    header = [
        "Experiment",
        "Runtime",
        "Source",
        "Status",
        "Rough",
        "Aug",
        "Align",
        "Cubic",
        "Sampling",
        "mIoU",
        "Mean F1",
        "WFm (3px)",
        "Note",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        values = [
            row.experiment,
            row.runtime,
            source_label(row.source),
            row.status,
            row.rough,
            row.aug,
            row.align,
            row.cubic,
            row.sampling,
            row.miou,
            row.mean_f1,
            row.wfm,
            row.note or "—",
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    all_rows = finalize_rows(gather_rows())
    chn6 = [row for row in all_rows if row.dataset == "CHN6-CUG"]
    potsdam = [row for row in all_rows if row.dataset == "Potsdam"]
    content = "\n".join(
        [
            "# Experiment Summary",
            "",
            "Runtime is exact when parsed from a successful training log.",
            "Entries prefixed with `~` are estimated from checkpoint timestamps.",
            "",
            "## CHN6-CUG",
            "",
            render_table(chn6),
            "",
            "## Potsdam",
            "",
            render_table(potsdam),
            "",
        ]
    )
    OUTPUT_PATH.write_text(content, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
