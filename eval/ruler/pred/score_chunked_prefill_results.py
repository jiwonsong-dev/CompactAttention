import argparse
import csv
import importlib
import json
from pathlib import Path

import yaml

TASK_ORDER = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]


def ordered_tasks(rows_by_task: dict) -> list[str]:
    ordered = [task for task in TASK_ORDER if task in rows_by_task]
    ordered.extend(task for task in rows_by_task if task not in TASK_ORDER)
    return ordered


def write_horizontal_summary(summary_file: Path, rows_by_task: dict) -> None:
    ordered = ordered_tasks(rows_by_task)
    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", *ordered])
        writer.writerow(["Score", *[rows_by_task[task]["Score"] for task in ordered]])
        writer.writerow(["Nulls", *[rows_by_task[task]["Nulls"] for task in ordered]])


def load_metric(task_name: str):
    with open("eval/ruler/synthetic.yaml", "r", encoding="utf-8") as f:
        task_cfg = yaml.safe_load(f)[task_name]
    data_mod = importlib.import_module("eval.ruler.data.synthetic.constants")
    eval_mod = importlib.import_module("eval.ruler.eval.synthetic.constants")
    merged = dict(data_mod.TASKS[task_cfg["task"]])
    merged.update(task_cfg)
    metric_fn = eval_mod.TASKS[task_cfg["task"]]["metric_fn"]
    return merged, metric_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if (data_dir / "pred").is_dir():
        data_dir = data_dir / "pred"

    rows_by_task = {}
    for pred_file in data_dir.glob("*.jsonl"):
        task = pred_file.stem
        _, metric_fn = load_metric(task)
        predicts = []
        references = []
        nulls = 0
        with open(pred_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                pred = row["pred"].strip()
                predicts.append(pred)
                references.append(row.get("outputs", [""]))
                if not pred:
                    nulls += 1
        score = metric_fn(predicts, references) if references and references[0][0] is not None else 0.0
        rows_by_task[task] = {
            "Tasks": task,
            "Score": score,
            "Nulls": f"{nulls}/{len(predicts)}",
        }

    summary_file = data_dir / "summary.csv"
    write_horizontal_summary(summary_file, rows_by_task)

    print(summary_file)


if __name__ == "__main__":
    main()
