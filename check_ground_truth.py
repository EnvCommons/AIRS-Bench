#!/usr/bin/env python3
"""
Check every task's ground truth against what its grader reads.

For each task this loads the ground truth with the grader's own loader (which
fails if a column the grader reads is missing), grades a submission built from
those labels and checks that it reaches the task's optimal score, and, when the
agent's test split is available, checks that it has one row per label.

Usage:
    python check_ground_truth.py [DATA_ROOT]

DATA_ROOT is the environment's data root, holding server_data/ and optionally
sandbox_data/ (default: /orwd_data if present, else this directory). Tasks in
EXCLUDED_TASKS are listed but not checked. Exits non-zero if any served task
fails.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import airs_bench
from task_config import TASK_NAMES, TASKS


def gold_submission(task_name: str, labels) -> str | None:
    """A submission.csv that reproduces the labels exactly, or None for
    metrics graded in the sandbox (Pass@5)."""
    config = TASKS[task_name]
    metric = config.metric
    if metric == "Pass@5":
        return None
    if metric == "DuoRCAccuracy":
        answers, no_answers = labels
        df = pd.DataFrame({
            "answer": [a[0] if (a and not na) else "" for a, na in zip(answers, no_answers)],
            "has_answer": [not na for na in no_answers],
        })
    elif metric == "MRR":
        # The grader keeps one id per query text (the last), so rank that one.
        best = dict(zip(labels["query"], labels["id"]))
        df = pd.DataFrame({
            "query": labels["query"],
            "rankings": [json.dumps([best[q]]) for q in labels["query"]],
        })
    elif metric == "MASE":
        full, _ = labels
        df = pd.DataFrame({"label_target": [json.dumps(t.tolist()) for t in full]})
    elif metric == "TimeSeriesMAE":
        df = pd.DataFrame({"label_target": [json.dumps(t.tolist()) for t in labels]})
    elif metric == "ExactMatch":
        df = pd.DataFrame({"answers": [a[0] for a in labels]})
    else:
        df = pd.DataFrame({config.submission_columns[0]: list(labels)})
    return df.to_csv(index=False)


def agent_row_count(test_dir: Path, metric: str) -> int | None:
    """Rows (or forecast series) the agent predicts for its test split."""
    from datasets import load_from_disk

    if metric in ("MRR", "Pass@5") or not test_dir.exists():
        return None
    test = load_from_disk(str(test_dir))
    if metric == "TimeSeriesMAE":
        first = test[0]["target"]
        if first and isinstance(first[0], list):
            return sum(len(t) for t in test["target"])
    return len(test)


def check_task(task_name: str, root: Path) -> list[str]:
    config = TASKS[task_name]
    env = airs_bench.AIRSBench(
        task_spec={"id": task_name, "task_name": task_name},
        secrets={"api_key": "unused"},
    )
    problems = []
    labels = env._load_labels()

    csv = gold_submission(task_name, labels)
    if csv is not None:
        score = env._evaluate_submission(csv, labels)
        # A few label rows can't be reproduced through a CSV (e.g. an answer
        # pandas reads as NaN), so allow a 0.001 shortfall in reward.
        span = config.estimated_worst_score - config.optimal_score
        shortfall = (score - config.optimal_score) / span
        if not math.isclose(score, config.optimal_score, abs_tol=1e-6):
            print(f"    note: gold submission scores {score:.6f} on {task_name}")
        if shortfall > 1e-3:
            problems.append(f"gold submission scores {score}, optimum is {config.optimal_score}")

    n_agent = agent_row_count(root / "sandbox_data" / task_name / "data" / "test", config.metric)
    if isinstance(labels, int):
        n_labels = labels
    elif config.metric in ("MASE", "DuoRCAccuracy"):
        n_labels = len(labels[0])
    else:
        n_labels = len(labels)
    if n_agent is not None and n_agent != n_labels:
        problems.append(f"agent test split has {n_agent} rows, ground truth has {n_labels} labels")
    return problems


def main() -> int:
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    elif Path("/orwd_data").exists():
        root = Path("/orwd_data")
    else:
        root = Path(__file__).parent
    airs_bench.DATA_DIR = root / "server_data"

    failed = False
    for task_name in TASK_NAMES:
        if task_name in airs_bench.EXCLUDED_TASKS:
            print(f"{'excluded':10} {task_name}: {airs_bench.EXCLUDED_TASKS[task_name]}")
            continue
        try:
            problems = check_task(task_name, root)
        except Exception as e:
            problems = [f"{type(e).__name__}: {e}"]
        failed = failed or bool(problems)
        status = "FAIL" if problems else "ok"
        print(f"{status:10} {task_name}" + "".join(f"\n    {p}" for p in problems))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
