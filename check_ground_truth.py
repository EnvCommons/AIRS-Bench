#!/usr/bin/env python3
"""
Check every task's ground truth against what its grader reads.

For each task this loads the ground truth with the grader's own loader (which
fails if a column the grader reads is missing), grades a submission built from
those labels and checks that it reaches the task's optimal score, and, when the
agent's test split is available, checks that it has one row per label.

For regression tasks it also grades trivial models built from the agent's data
(a constant train-set mean or median; for QM9 total energies, a per-element
reference model; for forecasting, each series' last value or historical mean)
and checks that the task's estimated_worst_score is the best
of their scores, so a trivial model earns reward 0.

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

# QM9 total energies: their trivial model is a sum of per-element reference
# energies over the molecule's atom counts (the atom reference QM9 models
# subtract), since a constant misses the near-linear dependence on composition.
ATOM_REFERENCE_TARGETS = {"U_0", "U", "H", "G"}
QM9_ELEMENTS = (1, 6, 7, 8, 9)


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


def _constant_csv(column: str, value: float, n: int) -> str:
    return pd.DataFrame({column: [value] * n}).to_csv(index=False)


def _agent_series(test_dir: Path) -> list[np.ndarray]:
    """The agent's forecast inputs, one per series, in submission order."""
    from datasets import load_from_disk

    series = []
    for target in load_from_disk(str(test_dir))["target"]:
        rows = target if (target and isinstance(target[0], list)) else [target]
        series.extend(np.asarray(r, dtype=float) for r in rows)
    return series


def element_counts(atomic_numbers) -> np.ndarray:
    """Per-molecule counts of each QM9 element (H, C, N, O, F), one row per molecule."""
    counts = np.zeros((len(atomic_numbers), len(QM9_ELEMENTS)))
    for i, numbers in enumerate(atomic_numbers):
        for j, z in enumerate(QM9_ELEMENTS):
            counts[i, j] = sum(1 for n in numbers if n == z)
        if counts[i].sum() != len(numbers):
            raise ValueError(f"molecule {i} has an element outside {QM9_ELEMENTS}")
    return counts


def atom_reference_predictions(train_numbers, train_y, test_numbers) -> np.ndarray:
    """Predict each test molecule as a sum of per-element reference energies,
    fitted on train by least squares (no intercept)."""
    weights, *_ = np.linalg.lstsq(element_counts(train_numbers), np.asarray(train_y, dtype=float), rcond=None)
    return element_counts(test_numbers) @ weights


def trivial_submissions(task_name: str, root: Path, n_labels: int) -> dict[str, str] | None:
    """Trivial-model submission.csv files for a regression task, by name, or
    None for other tasks or when the agent's data isn't available."""
    from datasets import load_from_disk

    config = TASKS[task_name]
    data_dir = root / "sandbox_data" / task_name / "data"
    column = config.submission_columns[0]
    if config.metric in ("MAE", "SpearmanCorrelation"):
        if not (data_dir / "train").exists():
            return None
        train = load_from_disk(str(data_dir / "train"))
        y = np.asarray(train[config.scoring_column], dtype=float).ravel()
        trivial = {
            "train mean": _constant_csv(column, float(y.mean()), n_labels),
            "train median": _constant_csv(column, float(np.median(y)), n_labels),
        }
        if config.scoring_column in ATOM_REFERENCE_TARGETS:
            test = load_from_disk(str(data_dir / "test"))
            predictions = atom_reference_predictions(train["atomic_numbers"], y, test["atomic_numbers"])
            trivial["per-element reference"] = pd.DataFrame({column: predictions}).to_csv(index=False)
        return trivial
    if config.metric == "TimeSeriesMAE":
        if not (data_dir / "test").exists():
            return None
        horizon = config.forecast_horizon
        forecasts = {"naive (last value)": [], "historical mean": []}
        for history in _agent_series(data_dir / "test"):
            observed = history[~np.isnan(history)]
            last = float(observed[-1]) if len(observed) else 0.0
            mean = float(observed.mean()) if len(observed) else 0.0
            forecasts["naive (last value)"].append([last] * horizon)
            forecasts["historical mean"].append([mean] * horizon)
        return {
            name: pd.DataFrame({"label_target": [json.dumps(r) for r in rows]}).to_csv(index=False)
            for name, rows in forecasts.items()
        }
    return None


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

    trivial = trivial_submissions(task_name, root, n_labels)
    if trivial:
        scores = {name: env._evaluate_submission(sub, labels) for name, sub in trivial.items()}
        for name, value in scores.items():
            print(f"    trivial model, {name}: {config.metric} {value:.6f}")
        best = min(scores.values()) if config.lower_is_better else max(scores.values())
        if not math.isclose(config.estimated_worst_score, best, rel_tol=1e-5, abs_tol=1e-9):
            problems.append(
                f"estimated_worst_score is {config.estimated_worst_score}, "
                f"the best trivial model scores {best:.7g}"
            )
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
