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

# Build task specs at module level for stable ordering
# Only include tasks that have data available
_task_specs: list[JSONObject] = []
for task_name in TASK_NAMES:
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

        # Track submission
        self.submitted = False
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
            if len(output) > 50000:
                output = output[:50000] + "\n...(truncated, file too large)"

            return ToolOutput(
                blocks=[TextBlock(text=output)],
                metadata={"output": output},
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
        This is a terminal action — you get one submission.
        """
        if self.submitted:
            return ToolOutput(
                blocks=[TextBlock(text="Already submitted. Only one submission is allowed: this "
                                       "episode is not re-graded, and repeat submissions are "
                                       "penalised (reward -0.1).")],
                metadata={"error": "Already submitted", "already_submitted": True},
                reward=REPEAT_SUBMISSION_PENALTY,
                finished=True,
            )

        self.submitted = True

        # Read submission.csv from the sandbox. A non-zero exit = the agent never
        # produced a submission -> a legitimate incorrect (handled below). A raised
        # sandbox error is infra: _with_retry retries, then re-raises on a persistent
        # failure (-> SDK ToolFailed -> clean terminal) rather than fabricating 0.0.
        csv_content, code = await self._with_retry(
            "read_submission",
            lambda: self.sandbox.run("cat /home/ubuntu/submission.csv"),
        )
        if code != 0:
            return ToolOutput(
                blocks=[TextBlock(text="Error: submission.csv not found at /home/ubuntu/submission.csv")],
                metadata={"error": "submission.csv not found"},
                reward=0.0,
                finished=True,
            )

        # Evaluate. The Pass@5 path runs an eval harness in the sandbox (flaky
        # infra) -> retry-then-raise. A grader/eval failure (sandbox harness crash,
        # missing ground truth, unparseable submission) is allowed to propagate
        # (-> ToolFailed -> clean terminal) instead of being scored as a fabricated
        # 0.0 — we never conflate "couldn't grade" with "graded as worst".
        if self.config.metric == "Pass@5":
            raw_score = await self._with_retry(
                "pass_at_5_eval",
                lambda: self._eval_pass_at_5_sandbox(csv_content),
            )
        else:
            raw_score = self._evaluate_submission(csv_content)

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

    def _evaluate_submission(self, csv_content: str) -> float:
        """
        Evaluate submission CSV content against ground truth.
        Returns the raw metric score.
        """
        metric = self.config.metric
        task_name = self.task_name

        # Load ground truth
        labels_dir = DATA_DIR / task_name / "test_with_labels"
        if not labels_dir.exists():
            raise FileNotFoundError(f"Ground truth not found at {labels_dir}")

        labels_ds = load_from_disk(str(labels_dir))

        # Parse submission CSV
        submission_df = pd.read_csv(io.StringIO(csv_content), header=0)

        # Dispatch to appropriate metric
        if metric == "DuoRCAccuracy":
            return self._eval_duorc(submission_df, labels_ds)
        elif metric == "FinQAAccuracy":
            return self._eval_finqa(submission_df, labels_ds)
        elif metric == "Accuracy":
            return self._eval_accuracy(submission_df, labels_ds)
        elif metric == "ExactMatch":
            return self._eval_exact_match(submission_df, labels_ds)
        elif metric == "MAE":
            return self._eval_mae(submission_df, labels_ds)
        elif metric == "Rouge1":
            return self._eval_rouge1(submission_df, labels_ds)
        elif metric == "SpearmanCorrelation":
            return self._eval_spearman(submission_df, labels_ds)
        elif metric == "MRR":
            return self._eval_mrr(submission_df, labels_ds)
        elif metric == "MASE":
            return self._eval_mase(submission_df, labels_ds)
        elif metric == "TimeSeriesMAE":
            return self._eval_time_series_mae(submission_df, labels_ds)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _eval_accuracy(self, submission_df: pd.DataFrame, labels_ds) -> float:
        """Evaluate integer accuracy."""
        preds = submission_df.values.squeeze()
        scoring_col = self.config.scoring_column
        labels = list(labels_ds[scoring_col])
        if len(preds) != len(labels):
            raise ValueError(
                f"Row count mismatch: {len(preds)} predictions vs {len(labels)} labels"
            )
        return accuracy_int(preds, labels)

    def _eval_finqa(self, submission_df: pd.DataFrame, labels_ds) -> float:
        """Evaluate FinQA accuracy with numeric tolerance."""
        preds = submission_df.values.squeeze()
        scoring_col = self.config.scoring_column
        labels = np.array(labels_ds[scoring_col])
        if len(preds) != len(labels):
            raise ValueError(
                f"Row count mismatch: {len(preds)} predictions vs {len(labels)} labels"
            )
        return finqa_accuracy(preds, labels)

    def _eval_duorc(self, submission_df: pd.DataFrame, labels_ds) -> float:
        """Evaluate DuoRC accuracy with answer/has_answer columns."""
        sub_answers = list(submission_df["answer"])
        sub_has_answers = list(submission_df["has_answer"])
        label_answers = list(labels_ds["answers"])
        label_no_answers = list(labels_ds["no_answer"])
        return duorc_accuracy(sub_answers, sub_has_answers, label_answers, label_no_answers)

    def _eval_exact_match(self, submission_df: pd.DataFrame, labels_ds) -> float:
        """Evaluate SQuAD-style exact match."""
        preds = submission_df.values.squeeze()
        # Labels are lists of acceptable answer texts
        labels = [x["text"] for x in labels_ds["answers"]]
        if len(preds) != len(labels):
            raise ValueError(
                f"Row count mismatch: {len(preds)} predictions vs {len(labels)} labels"
            )
        return exact_match(preds, labels)

    def _eval_mae(self, submission_df: pd.DataFrame, labels_ds) -> float:
        """Evaluate Mean Absolute Error."""
        preds = submission_df.values.squeeze()
        scoring_col = self.config.scoring_column
        labels = np.array(labels_ds[scoring_col])
        if len(preds) != len(labels):
            raise ValueError(
                f"Row count mismatch: {len(preds)} predictions vs {len(labels)} labels"
            )
        return mae(preds, labels)

    def _eval_rouge1(self, submission_df: pd.DataFrame, labels_ds) -> float:
        """Evaluate ROUGE-1 F-measure."""
        preds = submission_df.values.squeeze()
        # ELI5 labels: first answer text
        labels = [
            x["text"][0] if x["text"] else ""
            for x in labels_ds["answers"]
        ]
        if len(preds) != len(labels):
            raise ValueError(
                f"Row count mismatch: {len(preds)} predictions vs {len(labels)} labels"
            )
        return rouge1(preds, labels)

    def _eval_spearman(self, submission_df: pd.DataFrame, labels_ds) -> float:
        """Evaluate Spearman correlation."""
        preds = submission_df.values.squeeze()
        scoring_col = self.config.scoring_column
        labels = np.array(labels_ds[scoring_col])
        if len(preds) != len(labels):
            raise ValueError(
                f"Row count mismatch: {len(preds)} predictions vs {len(labels)} labels"
            )
        return spearman_correlation(preds, labels)

    def _eval_mrr(self, submission_df: pd.DataFrame, labels_ds) -> float:
        """Evaluate Mean Reciprocal Rank."""
        return mrr(submission_df, labels_ds)

    def _eval_mase(self, submission_df: pd.DataFrame, labels_ds) -> float:
        """Evaluate MASE for time series forecasting."""
        preds = submission_df.values.squeeze()
        label_targets = labels_ds["label_target"]
        train_targets = labels_ds["target"]
        return mase(preds, label_targets, train_targets)

    def _eval_time_series_mae(self, submission_df: pd.DataFrame, labels_ds) -> float:
        """Evaluate MAE for time series forecasting (Rideshare, Solar)."""
        preds = submission_df.values.squeeze()
        label_targets = labels_ds["label_target"]
        train_targets = labels_ds["target"]
        return time_series_mae(preds, label_targets, train_targets)

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
            error_log_result = await self.sandbox.run("tail -50 /tmp/eval/eval.log")
            error_log = error_log_result[0] if error_log_result[1] == 0 else "could not read error log"
            raise RuntimeError(
                f"Pass@5 evaluation failed (exit {eval_code}):\n{error_log}"
            )

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
        result = await self.sandbox.run(
            f"printf '%s' '{encoded}' | base64 -d > {remote_path}"
        )
        if result[1] != 0:
            raise RuntimeError(f"Failed to upload {remote_path}: {result[0]}")
