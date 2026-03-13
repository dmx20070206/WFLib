from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Append one evaluation result into a CSV summary.")
    parser.add_argument("--result-json", required=True, help="Path to exp/test.py output JSON")
    parser.add_argument("--summary-csv", required=True, help="CSV file to append summary row")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--n", required=True, type=int, help="RandAugment n")
    parser.add_argument("--m", required=True, type=int, help="RandAugment m")
    parser.add_argument("--test-file", required=True, help="Test file tag")
    args = parser.parse_args()

    result_path = Path(args.result_json)
    if not result_path.exists():
        raise FileNotFoundError(f"Result json not found: {result_path}")

    with result_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "dataset",
        "model",
        "n",
        "m",
        "test_file",
        "result_json",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
    ]

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset,
        "model": args.model,
        "n": args.n,
        "m": args.m,
        "test_file": args.test_file,
        "result_json": str(result_path),
        "Accuracy": metrics.get("Accuracy"),
        "Precision": metrics.get("Precision"),
        "Recall": metrics.get("Recall"),
        "F1-score": metrics.get("F1-score"),
    }

    write_header = not summary_path.exists() or summary_path.stat().st_size == 0
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(
        f"[record_eval_result] dataset={args.dataset} model={args.model} "
        f"n={args.n} m={args.m} test={args.test_file} "
        f"acc={row['Accuracy']} -> {summary_path}"
    )


if __name__ == "__main__":
    main()
