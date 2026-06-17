#!/usr/bin/env python
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

try:
    from eval.longbench_v2.common import extract_answer
except ModuleNotFoundError:
    from common import extract_answer


def _safe_pct(correct: float, total: int) -> float:
    return round(100.0 * correct / total, 1) if total else 0.0


def score_rows(rows: list[dict]) -> dict:
    buckets: dict[str, list[int]] = defaultdict(list)
    by_domain: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        pred = row.get("pred_answer")
        if pred is None:
            pred = extract_answer(row.get("response", ""))
        answer = str(row.get("answer", "")).strip().upper()
        judge = int(pred == answer)
        buckets["overall"].append(judge)
        buckets[str(row.get("difficulty", "unknown")).lower()].append(judge)
        buckets[str(row.get("length", "unknown")).lower()].append(judge)
        by_domain[str(row.get("domain", "unknown"))].append(judge)

    out = {
        "count": len(rows),
        "overall": _safe_pct(sum(buckets["overall"]), len(buckets["overall"])),
        "easy": _safe_pct(sum(buckets["easy"]), len(buckets["easy"])),
        "hard": _safe_pct(sum(buckets["hard"]), len(buckets["hard"])),
        "short": _safe_pct(sum(buckets["short"]), len(buckets["short"])),
        "medium": _safe_pct(sum(buckets["medium"]), len(buckets["medium"])),
        "long": _safe_pct(sum(buckets["long"]), len(buckets["long"])),
        "nulls": sum(1 for row in rows if not (row.get("pred_answer") or extract_answer(row.get("response", "")))),
        "domains": {
            domain: {
                "count": len(vals),
                "score": _safe_pct(sum(vals), len(vals)),
            }
            for domain, vals in sorted(by_domain.items())
        },
    }
    return out


def write_csv(path: Path, result: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in ["count", "overall", "easy", "hard", "short", "medium", "long", "nulls"]:
            writer.writerow([key, result[key]])
        writer.writerow([])
        writer.writerow(["domain", "count", "score"])
        for domain, row in result["domains"].items():
            writer.writerow([domain, row["count"], row["score"]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-file", required=True, type=Path, nargs="+")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    seen = set()
    for pred_file in args.pred_file:
        for line in pred_file.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row.get("_id")
            if row_id is not None and row_id in seen:
                continue
            if row_id is not None:
                seen.add(row_id)
            rows.append(row)
    result = score_rows(rows)
    summary_json = args.summary_json or args.pred_file[0].with_name("summary.json")
    summary_csv = args.summary_csv or args.pred_file[0].with_name("summary.csv")
    summary_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(summary_csv, result)
    print(summary_json)
    print(summary_csv)
    print(
        {
            key: result[key]
            for key in ["count", "overall", "easy", "hard", "short", "medium", "long", "nulls"]
        }
    )


if __name__ == "__main__":
    main()
