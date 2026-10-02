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

# Errors raised while parsing or scoring a malformed submission.csv (wrong row
# count, missing column, unparsable value). These are the agent's to fix, so the
# submission is reported back ungraded instead of failing the tool call.
SUBMISSION_ERRORS = (ValueError, KeyError, TypeError, IndexError, SyntaxError)


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

_PASS_AT_5_RUNNER_SCRIPT = r"""
import json
import sys
sys.path.insert(0, '/tmp/eval')

import pandas as pd
from datasets import load_from_disk

print("Loading APPS test dataset with labels...", file=sys.stderr)
ds = load_from_disk('/home/ubuntu/data/test_with_labels')
print(f"Loaded {len(ds)} test problems", file=sys.stderr)

print("Loading submission...", file=sys.stderr)
sub = pd.read_csv('/home/ubuntu/submission.csv')
submissions = sub[['code1', 'code2', 'code3', 'code4', 'code5']].values.tolist()
print(f"Loaded {len(submissions)} submissions", file=sys.stderr)

assert len(submissions) == len(ds), \
    f"Mismatch: {len(submissions)} submissions vs {len(ds)} test problems"

print("Running Pass@5 evaluation...", file=sys.stderr)
from utils import evaluate_all_testcases
score = evaluate_all_testcases(submissions, ds, max_workers=4)
print(json.dumps({"pass_at_5": float(score)}))
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
            image="generalreasoning/airs-bench-sandbox:latest",
            machine_size="2:8",
            block_network=False,
            bucket_config=SandboxBucketConfig(
                mount_path="/home/ubuntu/data",
                read_only=True,
                only_dir=f"sandbox_data/{self.task_name}/data",
            ),
        )

        or_client = AsyncOpenReward(api_key=api_key)
        self.sandbox = or_client.sandbox(self.sandbox_settings)

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
        # Same 20 tasks in both splits
        if split in ("train", "test"):
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

    async def _with_retry(self, label: str, call, *, max_attempts: int = 4):
        """Run a flaky sandbox/grader op with exponential backoff, re-raising on
        persistent failure so the SDK turns it into ToolFailed -> a clean terminal,
        instead of swallowing a grader/sandbox failure into a fabricated reward.
        `call` returns a fresh awaitable on each attempt.
        """
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                return await call()
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
                lambda: self._eval_pass_at_5_sandbox(csv_content),
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

    async def _eval_pass_at_5_sandbox(self, csv_content: str) -> float:
        """
        Run Pass@5 evaluation inside the sandbox.

        Uploads the evaluation harness (testing_util.py, utils.py, pyext shim)
        and a runner script to the sandbox. Test data (test_with_labels) must be
        pre-mounted in the sandbox bucket at /home/ubuntu/data/test_with_labels/.
        """
        # Create eval directory
        await self.sandbox.run("mkdir -p /tmp/eval")

        # Upload pyext shim (real pyext is broken on Python 3.12+)
        await self._upload_file_to_sandbox(_PYEXT_SHIM, "/tmp/eval/pyext.py")

        # Upload testing_util.py
        testing_util_path = DATA_DIR / self.task_name / "testing_util.py"
        if not testing_util_path.exists():
            raise FileNotFoundError(f"testing_util.py not found at {testing_util_path}")
        await self._upload_file_to_sandbox(
            testing_util_path.read_text(), "/tmp/eval/testing_util.py"
        )

        # Upload utils.py (contains evaluate_all_testcases)
        utils_path = DATA_DIR / self.task_name / "utils.py"
        if not utils_path.exists():
            raise FileNotFoundError(f"utils.py not found at {utils_path}")
        await self._upload_file_to_sandbox(
            utils_path.read_text(), "/tmp/eval/utils.py"
        )

        # Create and upload the runner script
        runner_script = _PASS_AT_5_RUNNER_SCRIPT
        await self._upload_file_to_sandbox(runner_script, "/tmp/eval/run_eval.py")

        # Run evaluation — this can take a long time (5000 problems × 5 submissions)
        logger.info("Starting Pass@5 evaluation in sandbox...")
        eval_result = await self.sandbox.run(
            "cd /tmp/eval && python run_eval.py 2>/tmp/eval/eval.log"
        )
        eval_output, eval_code = eval_result

        if eval_code != 0:
            # Read error log for diagnostics
            # The log can echo hidden test cases, so it stays in the server log
            # and out of the raised message, which the agent sees.
            error_log, log_code = await self.sandbox.run("tail -50 /tmp/eval/eval.log")
            if log_code != 0:
                error_log = "could not read error log"
            logger.error("Pass@5 evaluation failed (exit %s):\n%s", eval_code, error_log)
            raise RuntimeError(f"Pass@5 evaluation failed (exit {eval_code})")

        # Parse the JSON result from stdout
        try:
            result_data = json.loads(eval_output.strip().splitlines()[-1])
            score = float(result_data["pass_at_5"])
            logger.info(f"Pass@5 score: {score}")
            return score
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            raise RuntimeError(
                f"Failed to parse Pass@5 result from output: {eval_output}"
            ) from e

    async def _upload_file_to_sandbox(self, content: str, remote_path: str) -> None:
        """Upload a text file to the sandbox via base64 encoding."""
        encoded = base64.b64encode(content.encode()).decode()
        output, code = await self.sandbox.run(
            f"printf '%s' '{encoded}' | base64 -d > {remote_path}"
        )
        if code != 0:
            raise RuntimeError(f"Failed to upload {remote_path}: {output}")
