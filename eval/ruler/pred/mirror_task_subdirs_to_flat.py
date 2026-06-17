import argparse
import csv
import os
import time
from pathlib import Path


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


def _safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        try:
            if dst.is_symlink() and dst.resolve() == src.resolve():
                return
            if not dst.is_dir():
                dst.unlink()
        except FileNotFoundError:
            pass
    if not dst.exists():
        rel = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(rel)


def _collect_summary_rows(root: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for task in TASK_ORDER:
        summary_path = root / task / "summary.csv"
        if not summary_path.exists():
            continue
        with open(summary_path, newline="", encoding="utf-8") as f:
            raw_rows = list(csv.reader(f))
        if raw_rows and raw_rows[0] and raw_rows[0][0] == "Metric":
            tasks = raw_rows[0][1:]
            metrics = {row[0]: row[1:] for row in raw_rows[1:] if row}
            scores = metrics.get("Score", [])
            nulls = metrics.get("Nulls", [])
            for idx, task_name in enumerate(tasks):
                rows[task_name] = {
                    "Tasks": task_name,
                    "Score": scores[idx] if idx < len(scores) else "",
                    "Nulls": nulls[idx] if idx < len(nulls) else "",
                }
            continue
        with open(summary_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                task_name = row.get("Tasks")
                if task_name:
                    rows[task_name] = {
                        "Tasks": task_name,
                        "Score": row.get("Score", ""),
                        "Nulls": row.get("Nulls", ""),
                    }
    return rows


def mirror_once(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for task in TASK_ORDER:
        src = root / task / f"{task}.jsonl"
        dst = root / f"{task}.jsonl"
        if src.exists():
            _safe_symlink(src, dst)

    rows = _collect_summary_rows(root)
    summary_path = root / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        ordered = [task for task in TASK_ORDER if task in rows]
        ordered.extend(task for task in rows if task not in TASK_ORDER)
        writer.writerow(["Metric", *ordered])
        writer.writerow(["Score", *[rows[task]["Score"] for task in ordered]])
        writer.writerow(["Nulls", *[rows[task]["Nulls"] for task in ordered]])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Root directory containing per-task subdirectories.")
    p.add_argument("--poll-seconds", type=float, default=30.0)
    p.add_argument("--watch-seconds", type=float, default=0.0)
    args = p.parse_args()

    root = Path(args.root)
    if args.watch_seconds <= 0:
        mirror_once(root)
        return

    deadline = time.time() + float(args.watch_seconds)
    while True:
        mirror_once(root)
        if time.time() >= deadline:
            break
        time.sleep(float(args.poll_seconds))


if __name__ == "__main__":
    main()
