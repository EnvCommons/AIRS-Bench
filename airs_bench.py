"""
AIRS-Bench — OpenReward sandbox environment for end-to-end AI research evaluation.

20 tasks spanning NLP, Code, Math, Molecular Property Prediction, Graph ML,
and Time Series Forecasting. Agents get a sandbox to write code, train models,
and produce predictions (submission.csv). Server-side evaluation computes
task-specific metrics.

Paper: arXiv:2602.06855
"""

import asyncio
import base64
import io
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from datasets import load_from_disk
from openreward import AsyncOpenReward, SandboxBucketConfig, SandboxSettings
from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool
from openreward.toolsets import PDFToolset
from pydantic import BaseModel

from evaluate import (
    accuracy_eq,
    accuracy_int,
    duorc_accuracy,
    exact_match,
    finqa_accuracy,
    mae,
    mase,
    mrr,
    rouge1,
    spearman_correlation,
    time_series_mae,
)
from task_config import TASKS, TASK_NAMES

logger = logging.getLogger(__name__)

# --- Module-level data loading ---

if os.path.exists("/orwd_data"):
    DATA_DIR = Path("/orwd_data") / "server_data"
else:
    DATA_DIR = Path(__file__).parent / "server_data"

# Tasks that are not served, with the reason.
EXCLUDED_TASKS = {
    # A submission is 145,063 full-length series (about 1 GB of CSV), which is
    # too large to download and grade inside the environment server.
    "TimeSeriesForecastingKaggleWebTrafficMASE": "submission too large to grade",
}

# Tasks served in the test split but not in train, with the reason.
TEST_ONLY_TASKS = {
    # The reward starts above the majority-class accuracy (66/104), which the
    # small models that fit the 2-CPU / 8 GB sandbox do not beat, so every
    # attempt earns 0 and the task gives no training signal.
    "CoreferenceResolutionSuperGLUEWSCAccuracy": "majority class not beaten with sandbox compute",
}

# Build task specs at module level for stable ordering
# Only include tasks that have data available
_task_specs: list[JSONObject] = []
for task_name in TASK_NAMES:
    if task_name in EXCLUDED_TASKS:
        continue
    task_data_dir = DATA_DIR / task_name
    if not task_data_dir.exists():
        logging.getLogger(__name__).warning(f"Skipping {task_name}: no data at {task_data_dir}")
        continue
    config = TASKS[task_name]
    _task_specs.append({
        "id": task_name,
        "task_name": task_name,
        "metric": config.metric,
        "lower_is_better": config.lower_is_better,
        "category": config.category,
        "research_problem": config.research_problem,
    })


# --- Pydantic parameter models ---

# Reward for a submission made after the task has already been graded. Negative
# so repeat submissions are actively discouraged, not merely left unscored.
REPEAT_SUBMISSION_PENALTY = -0.1

SUBMISSION_PATH = "/home/ubuntu/submission.csv"
PASS_AT_5_COLUMNS = ["code1", "code2", "code3", "code4", "code5"]

SANDBOX_IMAGE = "generalreasoning/airs-bench-sandbox:latest"

# Pass@5 runs the submitted programs against the hidden test cases in separate
# grading sandboxes, which mount the task's ground truth. The agent's sandbox
# never holds the hidden tests, and the agent can't reach the grading sandboxes.
# Each program run costs about a CPU-second, so the problems are split across
# several sandboxes graded in parallel.
APPS_EVAL_DIR = Path(__file__).parent / "apps_eval"
PASS_AT_5_GRADING_MACHINE = "4:16"
PASS_AT_5_GRADING_SHARDS = 4
PASS_AT_5_GROUND_TRUTH_MOUNT = "/home/ubuntu/grading/test_with_labels"
# Programs not started within the time budget count as failing, which bounds
# grading time for submissions whose programs mostly time out.
PASS_AT_5_TIME_BUDGET = 75 * 60
PASS_AT_5_EVAL_TIMEOUT = PASS_AT_5_TIME_BUDGET + 20 * 60
# Raw bytes per upload command; each command carries ~4/3 of this as base64.
UPLOAD_CHUNK_BYTES = 768 * 1024

# Errors raised while parsing or scoring a malformed submission.csv (wrong row
# count, missing column, unparsable value). These are the agent's to fix, so the
# submission is reported back ungraded instead of failing the tool call.
SUBMISSION_ERRORS = (ValueError, KeyError, TypeError, IndexError, SyntaxError)


class PassAt5MemoryKill(RuntimeError):
    """The Pass@5 runner was killed for exhausting its grading sandbox's
    memory. Each program's memory is capped, so this is a grader fault; the
    same submission would be killed again, so it is not retried."""


class GroundTruthError(RuntimeError):
    """The server-side ground truth can't be read the way the task's grader
    needs. This is the environment's fault, so it is raised rather than
    reported to the agent as a submission it should fix."""


class BashParams(BaseModel, extra="forbid"):
    command: str
    # Optional per-call wall-clock cap in seconds. None defers to the sandbox's
    # own default rather than imposing a second, shorter one from the schema.
    timeout: Optional[float] = None


class SubmitParams(BaseModel, extra="forbid"):
    """Submit predictions for evaluation. Ensure submission.csv exists at /home/ubuntu/submission.csv."""
    pass


class ListFilesInput(BaseModel, extra="forbid"):
    path: str = "."
    show_hidden: bool = False
    recursive: bool = False


class ReadFileInput(BaseModel, extra="forbid"):
    path: str


class WriteFileInput(BaseModel, extra="forbid"):
    path: str
    content: str


class TodoWriteParams(BaseModel, extra="forbid"):
    todos: List[Dict[str, Any]]


# --- Pass@5 evaluation helpers (uploaded to sandbox at eval time) ---

# pyext shim: the real pyext package is broken on Python 3.12+ (uses removed
# inspect.getargspec). RuntimeModule.from_string is all testing_util.py needs.
_PYEXT_SHIM = """\
import types

class RuntimeModule(types.ModuleType):
    @classmethod
    def from_string(cls, name, doc, source):
        mod = cls(name, doc)
        exec(compile(source, name, 'exec'), mod.__dict__)
        return mod
"""

# Grades one shard: problems shard, shard + n_shards, ... of the test set.
# usage: run_eval.py GROUND_TRUTH_DIR SHARD N_SHARDS TIME_BUDGET_SECONDS
_PASS_AT_5_RUNNER_SCRIPT = r"""
import json
import sys
import time
sys.path.insert(0, '/tmp/eval')

import pandas as pd
from datasets import load_from_disk

import utils

ground_truth, shard, n_shards, budget = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
deadline = time.monotonic() + budget

print("Loading APPS test dataset with labels...", file=sys.stderr)
ds = load_from_disk(ground_truth)
print(f"Loaded {len(ds)} test problems", file=sys.stderr)

print("Loading submission...", file=sys.stderr)
sub = pd.read_csv('/tmp/eval/submission.csv')
submissions = sub[['code1', 'code2', 'code3', 'code4', 'code5']].values.tolist()
print(f"Loaded {len(submissions)} submissions", file=sys.stderr)

assert len(submissions) == len(ds), \
    f"Mismatch: {len(submissions)} submissions vs {len(ds)} test problems"

indices = list(range(shard, len(ds), n_shards))
# Identical programs give identical results, so each distinct one runs once.
shard_submissions = [list(dict.fromkeys(submissions[i])) for i in indices]

skipped = []
run_program = utils.solves_testcases

def run_within_budget(submission, testcases, verbose=False):
    if time.monotonic() > deadline:
        skipped.append(1)
        return False
    return run_program(submission, testcases, verbose)

utils.solves_testcases = run_within_budget

print(f"Running Pass@5 evaluation on {len(indices)} problems...", file=sys.stderr)
score = utils.evaluate_all_testcases(shard_submissions, ds.select(indices), max_workers=8)
print(json.dumps({"correct": round(score * len(indices)), "problems": len(indices),
                  "skipped_programs": len(skipped)}))
"""


# --- Environment class ---

class AIRSBench(Environment):
    toolsets = [PDFToolset]

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)

        self.task_name = str(task_spec["task_name"])
        if self.task_name not in TASKS:
            raise ValueError(f"Unknown task: {self.task_name}")

        self.config = TASKS[self.task_name]

        api_key = secrets.get("OPENREWARD_API_KEY") or secrets.get("api_key") or os.environ.get("OPENREWARD_API_KEY", "").strip('"')
        if not api_key:
            raise ValueError("OpenReward API key is required (pass as OPENREWARD_API_KEY)")

        self.sandbox_settings = SandboxSettings(
            environment="GeneralReasoning/AIRS-Bench",
            image=SANDBOX_IMAGE,
            machine_size="2:8",
            block_network=False,
            bucket_config=SandboxBucketConfig(
                mount_path="/home/ubuntu/data",
                read_only=True,
                only_dir=f"sandbox_data/{self.task_name}/data",
            ),
        )

        self.or_client = AsyncOpenReward(api_key=api_key)
        self.sandbox = self.or_client.sandbox(self.sandbox_settings)

        # Load project description for prompt
        desc_path = DATA_DIR / self.task_name / "project_description.md"
        if desc_path.exists():
            self.project_description = desc_path.read_text()
        else:
            logger.warning(f"project_description.md not found for {self.task_name}")
            self.project_description = f"Task: {self.task_name}"

        # Track submission. The lock serialises submit calls so a submission is
        # graded at most once even when calls arrive concurrently.
        self.submitted = False
        self._submit_lock = asyncio.Lock()
        self.todos: List[Dict[str, Any]] = []

    async def setup(self) -> None:
        await self.sandbox.start()

    async def teardown(self) -> None:
        await self.sandbox.stop()

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split == "train":
            return [t for t in _task_specs if t["id"] not in TEST_ONLY_TASKS]
        if split == "test":
            return _task_specs
        return []

    async def get_prompt(self) -> list[TextBlock]:
        prompt = f"""You are an AI research agent working on the AIRS-Bench benchmark.

## Task: {self.task_name}
## Category: {self.config.category}
## Research Problem: {self.config.research_problem}

{self.project_description}

## Environment

You have access to a Linux environment with Python 3.13 and common ML libraries installed.
Your data is mounted at `/home/ubuntu/data/` with `train/` and `test/` subdirectories
in HuggingFace datasets format. Load them with:
```python
from datasets import load_from_disk
train = load_from_disk('/home/ubuntu/data/train')
test = load_from_disk('/home/ubuntu/data/test')
```

## Objective

Build a model or solution and produce predictions for the test set.
Save your predictions as `/home/ubuntu/submission.csv` following the format
described above. Then use the `submit` tool to evaluate your submission.

## Evaluation

Your submission will be evaluated using the **{self.config.metric}** metric.
{"Lower is better." if self.config.lower_is_better else "Higher is better."}

You should work from the `/home/ubuntu` directory. Good luck!"""

        return [TextBlock(text=prompt)]

    @tool
    async def bash(self, params: BashParams) -> ToolOutput:
        """Executes a bash command in the sandbox environment."""
        run_kwargs = {} if params.timeout is None else {"timeout": params.timeout}
        result = await self.sandbox.run(params.command.strip(), **run_kwargs)
        output, code = result

        if result.truncated:
            output = f"...(truncated, output exceeded limit)\n{output}"

        return ToolOutput(
            blocks=[TextBlock(text=f"{output}\n\n(exit {code})")],
            metadata={"output": output, "exit_code": code, "truncated": result.truncated},
            reward=0.0,
            finished=False,
        )

    @tool
    async def list_files(self, params: ListFilesInput) -> ToolOutput:
        """Lists files in the sandbox environment."""
        try:
            ls_cmd = "ls"
            if params.show_hidden:
                ls_cmd += " -a"
            if params.recursive:
                ls_cmd += " -R"
            ls_cmd += f" -l {params.path}"

            result = await self.sandbox.run(ls_cmd)
            output, code = result
            if result.truncated:
                output = f"...(truncated)\n{output}"
            if code != 0:
                return ToolOutput(
                    blocks=[TextBlock(text=f"Failed to list files: exit code {code}")],
                    metadata={"error": f"exit code {code}"},
                    reward=0.0,
                    finished=False,
                )

            return ToolOutput(
                blocks=[TextBlock(text=output)],
                metadata={"output": output, "truncated": result.truncated},
                reward=0.0,
                finished=False,
            )
        except Exception as e:
            return ToolOutput(
                blocks=[TextBlock(text=f"Error listing files: {e}")],
                metadata={"error": str(e)},
                reward=0.0,
                finished=False,
            )

    @tool
    async def read_file(self, params: ReadFileInput) -> ToolOutput:
        """Reads the content of a file in the sandbox."""
        try:
            result = await self.sandbox.run(f"cat {params.path}")
            output, code = result
            truncated = result.truncated
            if code != 0:
                # Exit 141 = SIGPIPE, typically from binary files (PDFs, images, etc.)
                if code == 141:
                    msg = (
                        f"Cannot read {params.path}: binary file. "
                        "Use the bash tool with Python to extract content "
                        "(e.g. pymupdf for PDFs, PIL for images)."
                    )
                else:
                    msg = f"Error reading file: exit code {code}"
                return ToolOutput(
                    blocks=[TextBlock(text=msg)],
                    metadata={"error": msg},
                    reward=0.0,
                    finished=False,
                )
            if truncated or len(output) > 50000:
                truncated = True
                output = output[:50000] + "\n...(truncated, file too large)"

            return ToolOutput(
                blocks=[TextBlock(text=output)],
                metadata={"output": output, "truncated": truncated},
                reward=0.0,
                finished=False,
            )
        except Exception as e:
            return ToolOutput(
                blocks=[TextBlock(text=f"Error reading file: {e}")],
                metadata={"error": str(e)},
                reward=0.0,
                finished=False,
            )

    @tool
    async def write_file(self, params: WriteFileInput) -> ToolOutput:
        """Writes content to a file in the sandbox."""
        try:
            # Escape content for shell and write via heredoc
            # Use base64 encoding to avoid shell escaping issues
            import base64
            encoded = base64.b64encode(params.content.encode()).decode()
            cmd = f"echo '{encoded}' | base64 -d > {params.path}"
            result = await self.sandbox.run(cmd)
            _, code = result
            if code != 0:
                return ToolOutput(
                    blocks=[TextBlock(text=f"Error writing file: exit code {code}")],
                    metadata={"error": f"exit code {code}"},
                    reward=0.0,
                    finished=False,
                )

            return ToolOutput(
                blocks=[TextBlock(text=f"File written to {params.path}")],
                metadata={"success": True},
                reward=0.0,
                finished=False,
            )
        except Exception as e:
            return ToolOutput(
                blocks=[TextBlock(text=f"Error writing file: {e}")],
                metadata={"error": str(e)},
                reward=0.0,
                finished=False,
            )

    async def _with_retry(self, label: str, call, *, max_attempts: int = 4, no_retry: tuple = ()):
        """Run a flaky sandbox/grader op with exponential backoff, re-raising on
        persistent failure so the SDK turns it into ToolFailed -> a clean terminal,
        instead of swallowing a grader/sandbox failure into a fabricated reward.
        `call` returns a fresh awaitable on each attempt. Exceptions in
        `no_retry` are deterministic and re-raised at once.
        """
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                return await call()
            except no_retry:
                raise
            except Exception as e:
                last_exc = e
                if attempt < max_attempts - 1:
                    wait = min(2 ** attempt, 30)
                    logger.warning(
                        "AIRS grader op %r failed (attempt %d/%d): %s — retry in %ss",
                        label, attempt + 1, max_attempts, e, wait,
                    )
                    await asyncio.sleep(wait)
        assert last_exc is not None
        raise last_exc

    @tool
    async def submit(self, params: SubmitParams) -> ToolOutput:
        """
        Submit predictions for evaluation.

        Reads /home/ubuntu/submission.csv from the sandbox, evaluates it against
        ground truth labels using the task's metric, and returns the score.
        This is a terminal action — you get one graded submission. A submission
        that cannot be evaluated (missing file, wrong row count, bad format) is
        not graded and can be fixed and submitted again.
        """
        async with self._submit_lock:
            return await self._submit()

    async def _submit(self) -> ToolOutput:
        if self.submitted:
            return ToolOutput(
                blocks=[TextBlock(text="Already submitted. Only one submission is allowed: this "
                                       "episode is not re-graded, and repeat submissions are "
                                       "penalised (reward -0.1).")],
                metadata={"error": "Already submitted", "already_submitted": True},
                reward=REPEAT_SUBMISSION_PENALTY,
                finished=True,
            )

        # A missing submission.csv is the agent's to fix: report it ungraded so
        # it can write the file and submit again. A raised sandbox error is
        # infra: _with_retry retries, then re-raises on a persistent failure
        # (-> SDK ToolFailed -> clean terminal) rather than fabricating 0.0.
        _, exists_code = await self._with_retry(
            "check_submission",
            lambda: self.sandbox.run(f"test -f {SUBMISSION_PATH}"),
        )
        if exists_code != 0:
            return ToolOutput(
                blocks=[TextBlock(text=f"Error: submission.csv not found at {SUBMISSION_PATH}. "
                                       "Nothing was graded. Write your predictions there and "
                                       "call submit again.")],
                metadata={"error": "submission.csv not found", "graded": False},
                reward=0.0,
                finished=False,
            )

        # download() returns the whole file; run("cat ...") caps output at
        # 50 KB and would silently cut large submissions.
        csv_bytes = await self._with_retry(
            "read_submission",
            lambda: self.sandbox.download(SUBMISSION_PATH),
        )

        # Ground truth is loaded outside the submission-error handler below, so
        # a problem with it (missing file, missing column) raises instead of
        # asking the agent to fix a CSV that isn't at fault.
        labels = self._load_labels()

        # Evaluate. The Pass@5 path runs an eval harness in the sandbox (flaky
        # infra) -> retry-then-raise. A grader/eval failure (sandbox harness crash,
        # missing ground truth) is allowed to propagate (-> ToolFailed -> clean
        # terminal) instead of being scored as a fabricated 0.0 — we never
        # conflate "couldn't grade" with "graded as worst". A malformed
        # submission is reported back ungraded so the agent can fix it.
        try:
            csv_content = csv_bytes.decode("utf-8")
            if self.config.metric == "Pass@5":
                self._check_pass_at_5_submission(csv_content, labels)
            else:
                raw_score = self._evaluate_submission(csv_content, labels)
                # A NaN score would pass the [0, 1] clamp below as reward 1.0.
                if math.isnan(raw_score):
                    raise ValueError("the score is undefined (NaN); check the predictions "
                                     "for NaN or missing values")
        except SUBMISSION_ERRORS as e:
            msg = f"{type(e).__name__}: {e}"
            if len(msg) > 1000:
                msg = msg[:1000] + "...(truncated)"
            return ToolOutput(
                blocks=[TextBlock(text=f"Error: submission.csv could not be evaluated: {msg}\n"
                                       "Nothing was graded. Fix the file and call submit again.")],
                metadata={"error": msg, "graded": False},
                reward=0.0,
                finished=False,
            )

        if self.config.metric == "Pass@5":
            raw_score = await self._with_retry(
                "pass_at_5_eval",
                lambda: self._eval_pass_at_5(csv_content, labels),
                no_retry=(PassAt5MemoryKill,),
            )

        self.submitted = True

        # Normalize to [0, 1] higher-is-better
        worst = self.config.estimated_worst_score
        optimal = self.config.optimal_score
        if worst != optimal:
            reward = (worst - raw_score) / (worst - optimal)
            reward = max(0.0, min(1.0, reward))
        else:
            reward = 1.0 if raw_score == optimal else 0.0

        result_text = f"""Submission Results:
- Task: {self.task_name}
- Metric: {self.config.metric}
- Raw Score: {raw_score:.6f}
- Normalized Reward: {reward:.6f}
- {"Lower is better" if self.config.lower_is_better else "Higher is better"} (raw metric)"""

        return ToolOutput(
            blocks=[TextBlock(text=result_text)],
            metadata={
                "task_name": self.task_name,
                "metric": self.config.metric,
                "raw_score": raw_score,
                "reward": reward,
                "lower_is_better": self.config.lower_is_better,
            },
            reward=reward,
            finished=True,
        )

    @tool
    def todo_write(self, params: TodoWriteParams) -> ToolOutput:
        """
        Manage todo list for task planning and progress tracking.

        Each todo item should have: id, content, status, priority.
        Status options: "pending", "in_progress", "completed"
        Priority options: "high", "medium", "low"
        """
        self.todos = params.todos

        output_lines = ["=== TODO LIST ==="]
        for todo in self.todos:
            status_icon = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(
                todo.get("status", "pending"), "[?]"
            )
            priority_icon = {"high": "!!!", "medium": "!!", "low": "!"}.get(
                todo.get("priority", "medium"), "?"
            )
            output_lines.append(f"{status_icon} {priority_icon} {todo.get('content', 'No description')}")

        content = "\n".join(output_lines)

        return ToolOutput(
            blocks=[TextBlock(text=content)],
            metadata={"todos": self.todos, "count": len(self.todos)},
            finished=False,
            reward=0.0,
        )

    # --- Evaluation logic ---

    def _ground_truth_columns(self) -> list[str]:
        """Columns of the ground-truth dataset that the task's grader reads."""
        metric = self.config.metric
        if metric == "DuoRCAccuracy":
            return ["answers", "no_answer"]
        if metric in ("ExactMatch", "Rouge1"):
            return ["answers"]
        if metric == "MRR":
            return ["docstring_tokens", "id"]
        if metric == "MASE":
            return ["target"]
        if metric == "TimeSeriesMAE":
            return ["target", "feat_dynamic_real"]
        return [self.config.scoring_column]

    def _load_labels(self) -> Any:
        """Load the task's ground truth and extract the labels its metric needs.

        The ground truth is the source dataset's evaluation split as published,
        without per-task preprocessing; this method applies it.
        Problems with it are the environment's, so they raise instead of being
        reported as a submission error.
        """
        labels_dir = DATA_DIR / self.task_name / "test_with_labels"
        if not labels_dir.exists():
            raise FileNotFoundError(f"Ground truth not found at {labels_dir}")
        labels_ds = load_from_disk(str(labels_dir))

        missing = [c for c in self._ground_truth_columns() if c not in labels_ds.column_names]
        if missing:
            raise GroundTruthError(f"Ground truth for {self.task_name} lacks column(s) {missing}")
        try:
            return self._extract_labels(labels_ds)
        except Exception as e:
            # The message names the exception type only, so no label values
            # reach the agent.
            raise GroundTruthError(
                f"Ground truth for {self.task_name} could not be read ({type(e).__name__})"
            ) from e

    def _extract_labels(self, labels_ds) -> Any:
        metric = self.config.metric
        col = self.config.scoring_column
        if metric == "Accuracy":
            return list(labels_ds[col])
        if metric in ("FinQAAccuracy", "SpearmanCorrelation"):
            return np.array(labels_ds[col])
        if metric == "MAE":
            return np.asarray(labels_ds[col], dtype=float) * self.config.label_scale
        if metric == "DuoRCAccuracy":
            return list(labels_ds["answers"]), list(labels_ds["no_answer"])
        if metric == "ExactMatch":
            # Labels are lists of acceptable answer texts
            return [x["text"] for x in labels_ds["answers"]]
        if metric == "Rouge1":
            # ELI5 labels: first answer text
            return [x["text"][0] if x["text"] else "" for x in labels_ds["answers"]]
        if metric == "MRR":
            # Queries are the joined docstring tokens, as in the agent's test queries.
            return {
                "query": [" ".join(tokens) for tokens in labels_ds["docstring_tokens"]],
                "id": list(labels_ds["id"]),
            }
        if metric == "MASE":
            # Predictions are full sequences (history + forecast); the history
            # is each test series without its last forecast_horizon steps.
            horizon = self.config.forecast_horizon
            full = [np.asarray(t, dtype=float) for t in labels_ds["target"]]
            return full, [t[:-horizon] for t in full]
        if metric == "TimeSeriesMAE":
            return self._forecast_labels(labels_ds)
        if metric == "Pass@5":
            return len(labels_ds)
        raise ValueError(f"Unknown metric: {metric}")

    def _forecast_labels(self, labels_ds) -> list[np.ndarray]:
        """One label per forecast series: its last forecast_horizon steps.

        A multivariate row (target is a list of series) contributes each target
        series followed by each feat_dynamic_real series, matching the order of
        the agent's test split, where every one of them is forecast.
        """
        horizon = self.config.forecast_horizon
        labels = []
        for target, dynamic in zip(labels_ds["target"], labels_ds["feat_dynamic_real"]):
            if target and isinstance(target[0], list):
                series = list(target) + list(dynamic or [])
            else:
                series = [target]
            labels.extend(np.asarray(s, dtype=float)[-horizon:] for s in series)
        return labels

    def _evaluate_submission(self, csv_content: str, labels: Any) -> float:
        """
        Evaluate submission CSV content against the labels from _load_labels.
        Returns the raw metric score.
        """
        metric = self.config.metric

        # Parse submission CSV
        submission_df = pd.read_csv(io.StringIO(csv_content), header=0)

        # Dispatch to appropriate metric
        if metric == "DuoRCAccuracy":
            label_answers, label_no_answers = labels
            return duorc_accuracy(
                list(submission_df["answer"]), list(submission_df["has_answer"]),
                label_answers, label_no_answers,
            )
        if metric == "MRR":
            return mrr(submission_df, labels)

        # atleast_1d keeps a one-row submission a sequence rather than a scalar.
        preds = np.atleast_1d(submission_df.values.squeeze())
        if metric == "MASE":
            label_targets, train_targets = labels
            return mase(preds, label_targets, train_targets)
        if metric == "TimeSeriesMAE":
            return time_series_mae(preds, labels)

        if len(preds) != len(labels):
            raise ValueError(
                f"Row count mismatch: {len(preds)} predictions vs {len(labels)} labels"
            )
        if metric == "Accuracy":
            return accuracy_int(preds, labels)
        if metric == "FinQAAccuracy":
            return finqa_accuracy(preds, labels)
        if metric == "ExactMatch":
            return exact_match(preds, labels)
        if metric == "MAE":
            return mae(preds, labels)
        if metric == "Rouge1":
            return rouge1(preds, labels)
        if metric == "SpearmanCorrelation":
            return spearman_correlation(preds, labels)
        raise ValueError(f"Unknown metric: {metric}")

    def _check_pass_at_5_submission(self, csv_content: str, n_problems: int) -> None:
        """Validate the Pass@5 submission's shape before running the sandbox
        harness, so a malformed file is reported to the agent rather than
        surfacing as a harness failure."""
        submission_df = pd.read_csv(io.StringIO(csv_content))
        missing = [c for c in PASS_AT_5_COLUMNS if c not in submission_df.columns]
        if missing:
            raise KeyError(f"missing column(s) {missing}; expected {PASS_AT_5_COLUMNS}")
        if len(submission_df) != n_problems:
            raise ValueError(
                f"Row count mismatch: {len(submission_df)} submissions vs {n_problems} test problems"
            )

    def _pass_at_5_grading_sandbox(self):
        """A fresh sandbox for one Pass@5 grading run, with the task's ground
        truth (hidden test cases) mounted. Network is blocked so the submitted
        programs run offline."""
        return self.or_client.sandbox(SandboxSettings(
            environment="GeneralReasoning/AIRS-Bench",
            image=SANDBOX_IMAGE,
            machine_size=PASS_AT_5_GRADING_MACHINE,
            block_network=True,
            bucket_config=SandboxBucketConfig(
                mount_path=PASS_AT_5_GROUND_TRUTH_MOUNT,
                read_only=True,
                only_dir=f"server_data/{self.task_name}/test_with_labels",
            ),
        ))

    async def _eval_pass_at_5(self, csv_content: str, n_problems: int) -> float:
        """
        Run Pass@5 evaluation across grading sandboxes, one per shard of the
        problems, and combine their counts. A failed shard cancels the others.
        """
        n_shards = max(1, min(PASS_AT_5_GRADING_SHARDS, n_problems))
        tasks = [
            asyncio.create_task(self._eval_pass_at_5_shard(csv_content, shard, n_shards))
            for shard in range(n_shards)
        ]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        correct = sum(r["correct"] for r in results)
        problems = sum(r["problems"] for r in results)
        skipped = sum(r["skipped_programs"] for r in results)
        if problems != n_problems:
            raise RuntimeError(f"Pass@5 graded {problems} problems, expected {n_problems}")
        if skipped:
            logger.warning("Pass@5: %d programs not run within the time budget", skipped)
        logger.info("Pass@5: %d/%d problems solved", correct, problems)
        return correct / problems

    async def _eval_pass_at_5_shard(self, csv_content: str, shard: int, n_shards: int) -> dict:
        """
        Grade one shard of the problems in a fresh grading sandbox.

        Uploads the evaluation harness (testing_util.py, utils.py, pyext shim),
        a runner script and the submission to a grading sandbox that mounts the
        hidden test cases, runs it there and stops the sandbox.
        """
        grader = self._pass_at_5_grading_sandbox()
        await grader.start()
        try:
            return await self._run_pass_at_5_shard(grader, csv_content, shard, n_shards)
        finally:
            await grader.stop()

    async def _run_pass_at_5_shard(self, grader, csv_content: str, shard: int, n_shards: int) -> dict:
        await grader.run("mkdir -p /tmp/eval")

        # pyext shim (real pyext is broken on Python 3.12+)
        await self._upload_file_to_sandbox(_PYEXT_SHIM, "/tmp/eval/pyext.py", grader)
        for name in ("testing_util.py", "utils.py"):
            await self._upload_file_to_sandbox(
                (APPS_EVAL_DIR / name).read_text(), f"/tmp/eval/{name}", grader
            )
        await self._upload_file_to_sandbox(_PASS_AT_5_RUNNER_SCRIPT, "/tmp/eval/run_eval.py", grader)
        await self._upload_file_to_sandbox(csv_content, "/tmp/eval/submission.csv", grader)

        logger.info("Starting Pass@5 evaluation of shard %d/%d...", shard + 1, n_shards)
        eval_result = await grader.run(
            f"cd /tmp/eval && python run_eval.py {PASS_AT_5_GROUND_TRUTH_MOUNT} "
            f"{shard} {n_shards} {PASS_AT_5_TIME_BUDGET} 2>/tmp/eval/eval.log",
            timeout=PASS_AT_5_EVAL_TIMEOUT,
        )
        eval_output, eval_code = eval_result

        if eval_code != 0:
            # Read error log for diagnostics
            # The log can echo hidden test cases, so it stays in the server log
            # and out of the raised message, which the agent sees.
            error_log, log_code = await grader.run("tail -50 /tmp/eval/eval.log")
            if log_code != 0:
                error_log = "could not read error log"
            logger.error("Pass@5 evaluation failed (exit %s):\n%s\n%s", eval_code, eval_output[-500:], error_log)
            if eval_code == 137 and "memory usage exceeded" in eval_output:
                raise PassAt5MemoryKill(f"Pass@5 evaluation failed (exit {eval_code})")
            raise RuntimeError(f"Pass@5 evaluation failed (exit {eval_code})")

        # Parse the JSON result from stdout
        try:
            result_data = json.loads(eval_output.strip().splitlines()[-1])
            return {key: int(result_data[key]) for key in ("correct", "problems", "skipped_programs")}
        except (json.JSONDecodeError, KeyError, IndexError, ValueError, TypeError) as e:
            # The output can include what the submitted programs printed, so it
            # goes to the server log rather than the raised message.
            logger.error("Unparsable Pass@5 runner output:\n%s", eval_output[-2000:])
            raise RuntimeError("Failed to parse Pass@5 result from the runner output") from e

    async def _upload_file_to_sandbox(self, content: str, remote_path: str, sandbox=None) -> None:
        """Upload a text file to a sandbox (default: the agent's) via base64,
        in chunks so a large file never becomes one oversized command."""
        sandbox = sandbox or self.sandbox
        data = content.encode()
        # Chunks are a multiple of 3 bytes, so each one base64-encodes on its own.
        offsets = range(0, max(len(data), 1), UPLOAD_CHUNK_BYTES)
        for i, start in enumerate(offsets):
            encoded = base64.b64encode(data[start:start + UPLOAD_CHUNK_BYTES]).decode()
            redirect = ">" if i == 0 else ">>"
            output, code = await sandbox.run(
                f"printf '%s' '{encoded}' | base64 -d {redirect} {remote_path}"
            )
            if code != 0:
                raise RuntimeError(f"Failed to upload {remote_path}: {output}")
