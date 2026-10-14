"""
Unit tests for AIRS-Bench environment.

Tests cover:
- Evaluation metric functions with known inputs
- Task configuration and listing
- Environment initialization and prompt generation
"""

import numpy as np
import pandas as pd
import pytest

from evaluate import (
    accuracy_eq,
    finqa_accuracy,
    accuracy_int,
    duorc_accuracy,
    exact_match,
    mae,
    mase,
    rouge1,
    spearman_correlation,
    time_series_mae,
)
from task_config import TASK_NAMES, TASKS


# --- Metric tests ---


class TestAccuracy:
    def test_perfect(self):
        preds = [1, 2, 3, 4, 5]
        labels = [1, 2, 3, 4, 5]
        assert accuracy_int(preds, labels) == 1.0

    def test_zero(self):
        preds = [0, 0, 0]
        labels = [1, 2, 3]
        assert accuracy_int(preds, labels) == 0.0

    def test_partial(self):
        preds = [1, 0, 3]
        labels = [1, 2, 3]
        assert accuracy_int(preds, labels) == pytest.approx(2.0 / 3.0)

    def test_string_inputs(self):
        preds = ["1", "2", "3"]
        labels = ["1", "2", "4"]
        assert accuracy_int(preds, labels) == pytest.approx(2.0 / 3.0)

    def test_accuracy_eq_int(self):
        preds = [0, 1, 2, 3, 4]
        labels = [0, 1, 2, 3, 4]
        assert accuracy_eq(preds, labels) == 1.0

    def test_accuracy_eq_strings(self):
        preds = ["0", "1", "2"]
        labels = ["0", "1", "3"]
        assert accuracy_eq(preds, labels) == pytest.approx(2.0 / 3.0)


class TestDuoRCAccuracy:
    def test_all_correct(self):
        sub_answers = ["hello", "world"]
        sub_has_answers = [True, True]
        label_answers = [["Hello"], ["World"]]
        label_no_answers = [False, False]
        assert duorc_accuracy(sub_answers, sub_has_answers, label_answers, label_no_answers) == 1.0

    def test_no_answer_correct(self):
        sub_answers = ["", ""]
        sub_has_answers = [False, False]
        label_answers = [[], []]
        label_no_answers = [True, True]
        assert duorc_accuracy(sub_answers, sub_has_answers, label_answers, label_no_answers) == 1.0

    def test_wrong_answer(self):
        sub_answers = ["wrong"]
        sub_has_answers = [True]
        label_answers = [["correct"]]
        label_no_answers = [False]
        assert duorc_accuracy(sub_answers, sub_has_answers, label_answers, label_no_answers) == 0.0


class TestFinQAAccuracy:
    def test_exact_numeric(self):
        preds = ["0.34", "12345.67"]
        labels = ["0.34", "12345.67"]
        assert finqa_accuracy(preds, labels) == 1.0

    def test_numeric_tolerance(self):
        preds = ["0.34001"]
        labels = ["0.34"]
        assert finqa_accuracy(preds, labels) == 1.0

    def test_currency_symbols(self):
        preds = ["$1,234.56"]
        labels = ["1234.56"]
        assert finqa_accuracy(preds, labels) == 1.0

    def test_percent(self):
        preds = ["5%"]
        labels = ["0.05"]
        assert finqa_accuracy(preds, labels) == 1.0

    def test_parenthetical_negative(self):
        preds = ["(123)"]
        labels = ["-123"]
        assert finqa_accuracy(preds, labels) == 1.0

    def test_string_fallback(self):
        preds = ["yes"]
        labels = ["yes"]
        assert finqa_accuracy(preds, labels) == 1.0

    def test_string_case_insensitive(self):
        preds = ["YES"]
        labels = ["yes"]
        assert finqa_accuracy(preds, labels) == 1.0

    def test_wrong_answer(self):
        preds = ["100"]
        labels = ["200"]
        assert finqa_accuracy(preds, labels) == 0.0


class TestExactMatch:
    def test_match_in_list(self):
        preds = ["Paris", "London"]
        labels = [["Paris", "paris"], ["London", "london"]]
        assert exact_match(preds, labels) == 1.0

    def test_no_match(self):
        preds = ["Berlin"]
        labels = [["Paris", "paris"]]
        assert exact_match(preds, labels) == 0.0

    def test_none_edge_case(self):
        preds = ['"None"']
        labels = [["None"]]
        assert exact_match(preds, labels) == 1.0


class TestMAE:
    def test_perfect(self):
        preds = [1.0, 2.0, 3.0]
        labels = [1.0, 2.0, 3.0]
        assert mae(preds, labels) == 0.0

    def test_known_error(self):
        preds = [1.0, 2.0, 3.0]
        labels = [2.0, 3.0, 4.0]
        assert mae(preds, labels) == 1.0

    def test_string_preds(self):
        preds = ["1.5", "2.5"]
        labels = [1.0, 2.0]
        assert mae(preds, labels) == 0.5

    def test_list_string_preds(self):
        preds = ["[1.5]", "[2.5]"]
        labels = [1.0, 2.0]
        assert mae(preds, labels) == 0.5


class TestRouge1:
    def test_identical(self):
        preds = ["the cat sat on the mat"]
        labels = ["the cat sat on the mat"]
        score = rouge1(preds, labels)
        assert score == pytest.approx(1.0)

    def test_no_overlap(self):
        preds = ["hello world"]
        labels = ["goodbye universe"]
        score = rouge1(preds, labels)
        assert score < 0.5

    def test_partial_overlap(self):
        preds = ["the cat"]
        labels = ["the dog"]
        score = rouge1(preds, labels)
        assert 0.0 < score < 1.0


class TestSpearmanCorrelation:
    def test_perfect_positive(self):
        preds = [1, 2, 3, 4, 5]
        labels = [1, 2, 3, 4, 5]
        assert spearman_correlation(preds, labels) == pytest.approx(1.0)

    def test_perfect_negative(self):
        preds = [5, 4, 3, 2, 1]
        labels = [1, 2, 3, 4, 5]
        assert spearman_correlation(preds, labels) == pytest.approx(-1.0)

    def test_no_correlation(self):
        preds = [1, 2, 3, 4, 5]
        labels = [3, 1, 4, 2, 5]
        result = spearman_correlation(preds, labels)
        assert -1.0 <= result <= 1.0


class TestMASE:
    def test_perfect_forecast(self):
        """Perfect prediction should give MASE = 0."""
        # Train: [1, 2, 3, 4, 5], Forecast portion: [6, 7, 8]
        train = [1.0, 2.0, 3.0, 4.0, 5.0]
        full_label = train + [6.0, 7.0, 8.0]  # full sequence
        # Prediction is the full sequence
        pred_str = str(full_label)

        result = mase([pred_str], [full_label], [train])
        assert result == pytest.approx(0.0)

    def test_nonzero_error(self):
        """Non-zero forecast error should give positive MASE."""
        train = [1.0, 2.0, 3.0, 4.0, 5.0]
        full_label = train + [6.0, 7.0, 8.0]
        pred_full = train + [7.0, 8.0, 9.0]  # off by 1 each
        pred_str = str(pred_full)

        result = mase([pred_str], [full_label], [train])
        # MAE of forecast = 1.0, MAE of naive on train = 1.0, so MASE = 1.0
        assert result == pytest.approx(1.0)


class TestTimeSeriesMAE:
    def test_perfect(self):
        """Perfect prediction should give MAE = 0."""
        train = [1.0, 2.0, 3.0]
        full_label = train + [4.0, 5.0]
        pred_str = "[4.0, 5.0]"

        result = time_series_mae([pred_str], [full_label], [train])
        assert result == pytest.approx(0.0)

    def test_known_error(self):
        """Known error should give expected MAE."""
        train = [1.0, 2.0, 3.0]
        full_label = train + [4.0, 5.0]
        pred_str = "[5.0, 6.0]"

        result = time_series_mae([pred_str], [full_label], [train])
        assert result == pytest.approx(1.0)


# --- Task config tests ---


class TestTaskConfig:
    def test_all_20_tasks(self):
        assert len(TASK_NAMES) == 20

    def test_stable_ordering(self):
        names1 = TASK_NAMES
        names2 = sorted(TASKS.keys())
        assert names1 == names2

    def test_all_tasks_have_required_fields(self):
        for name, config in TASKS.items():
            assert config.name == name
            assert config.metric
            assert isinstance(config.lower_is_better, bool)
            assert config.dataset
            assert config.dataset_config
            assert config.category
            assert config.research_problem

    def test_metric_direction(self):
        """Lower-is-better tasks should have optimal_score=0.0."""
        for name, config in TASKS.items():
            if config.lower_is_better:
                assert config.optimal_score == 0.0, f"{name}: lower_is_better but optimal != 0"
            else:
                assert config.optimal_score == 1.0, f"{name}: higher_is_better but optimal != 1"


# --- Environment class tests ---


class TestAIRSBenchEnv:
    def test_list_splits(self):
        from airs_bench import AIRSBench
        splits = AIRSBench.list_splits()
        assert "train" in splits
        assert "test" in splits

    def test_list_tasks_count(self):
        from airs_bench import AIRSBench
        tasks = AIRSBench.list_tasks("train")
        # 19 tasks available (Rideshare dataset has a pandas compat issue)
        assert len(tasks) >= 19

    def test_list_tasks_structure(self):
        from airs_bench import AIRSBench
        tasks = AIRSBench.list_tasks("train")
        task = tasks[0]
        assert "id" in task
        assert "task_name" in task
        assert "metric" in task
        assert "lower_is_better" in task
        assert "category" in task

    def test_list_tasks_stable_ordering(self):
        from airs_bench import AIRSBench
        tasks1 = AIRSBench.list_tasks("train")
        tasks2 = AIRSBench.list_tasks("train")
        assert [t["id"] for t in tasks1] == [t["id"] for t in tasks2]

    def test_unknown_split_returns_empty(self):
        from airs_bench import AIRSBench
        tasks = AIRSBench.list_tasks("nonexistent")
        assert tasks == []


# --- Reward normalization tests ---


class TestRewardNormalization:
    """Test the reward normalization formula: (worst - raw) / (worst - optimal), clipped to [0, 1]."""

    def _normalize(self, raw_score: float, worst: float, optimal: float) -> float:
        """Replicate the normalization logic from airs_bench.py submit method."""
        if worst != optimal:
            reward = (worst - raw_score) / (worst - optimal)
            return max(0.0, min(1.0, reward))
        return 1.0 if raw_score == optimal else 0.0

    # --- Higher-is-better tasks (accuracy, optimal=1.0) ---

    def test_accuracy_perfect(self):
        # SVAMP: optimal=1.0, worst=0.0
        assert self._normalize(1.0, worst=0.0, optimal=1.0) == pytest.approx(1.0)

    def test_accuracy_zero(self):
        assert self._normalize(0.0, worst=0.0, optimal=1.0) == pytest.approx(0.0)

    def test_accuracy_partial(self):
        # raw=0.7, optimal=1.0, worst=0.0 → reward=0.7
        assert self._normalize(0.7, worst=0.0, optimal=1.0) == pytest.approx(0.7)

    def test_accuracy_with_nonzero_worst(self):
        # Winogrande: optimal=1.0, worst=0.4665
        # raw=1.0 → reward=1.0
        assert self._normalize(1.0, worst=0.4665, optimal=1.0) == pytest.approx(1.0)
        # raw=0.4665 → reward=0.0
        assert self._normalize(0.4665, worst=0.4665, optimal=1.0) == pytest.approx(0.0)
        # raw=0.7 → reward=(0.4665-0.7)/(0.4665-1.0)=0.2335/0.5335≈0.4377
        assert self._normalize(0.7, worst=0.4665, optimal=1.0) == pytest.approx(0.4377, abs=0.001)

    # --- Lower-is-better tasks (MAE, optimal=0.0) ---

    def test_mae_perfect(self):
        # CvQM9: optimal=0.0, worst=132.633
        assert self._normalize(0.0, worst=132.633, optimal=0.0) == pytest.approx(1.0)

    def test_mae_at_worst(self):
        assert self._normalize(132.633, worst=132.633, optimal=0.0) == pytest.approx(0.0)

    def test_mae_partial(self):
        # raw=10.0, worst=132.633, optimal=0.0 → reward = (132.633-10)/132.633 ≈ 0.9246
        assert self._normalize(10.0, worst=132.633, optimal=0.0) == pytest.approx(0.9246, abs=0.001)

    def test_mae_worse_than_worst_clips_to_zero(self):
        # raw=200.0, worse than worst=132.633 → clipped to 0.0
        assert self._normalize(200.0, worst=132.633, optimal=0.0) == pytest.approx(0.0)

    # --- Spearman with negative worst ---

    def test_spearman_negative_worst(self):
        # Spearman: optimal=1.0, worst=-0.587
        assert self._normalize(1.0, worst=-0.587, optimal=1.0) == pytest.approx(1.0)
        assert self._normalize(-0.587, worst=-0.587, optimal=1.0) == pytest.approx(0.0)
        # raw=0.8 → (−0.587−0.8)/(−0.587−1.0) = −1.387/−1.587 ≈ 0.874
        assert self._normalize(0.8, worst=-0.587, optimal=1.0) == pytest.approx(0.874, abs=0.001)

    # --- Edge case: equal worst and optimal ---

    def test_equal_worst_optimal(self):
        # When worst == optimal, raw == optimal → 1.0, otherwise 0.0
        assert self._normalize(0.5, worst=0.5, optimal=0.5) == 1.0
        assert self._normalize(0.6, worst=0.5, optimal=0.5) == 0.0

    # --- Higher-is-better above optimal clips to 1.0 ---

    def test_above_optimal_clips(self):
        # If somehow raw_score > optimal for accuracy, clip to 1.0
        assert self._normalize(1.1, worst=0.0, optimal=1.0) == pytest.approx(1.0)


# --- submit / sandbox I/O tests ---


import asyncio
import base64
import re
from pathlib import Path

from openreward.api.sandboxes.types import RunResult


class FakeSandbox:
    """In-memory stand-in for the SDK sandbox handle.

    Mirrors the SDK contract: run() returns a RunResult (unpacks as
    (output, return_code), not subscriptable) whose output keeps only the first
    max_bytes characters (default 50_000); download() returns the whole file and
    raises when the file is missing.
    """

    def __init__(self, files=None):
        self.files: dict[str, bytes] = dict(files or {})
        self.commands: list[str] = []

    async def run(self, cmd, timeout=300, max_bytes=50_000, sanitise=True):
        self.commands.append(cmd)
        await asyncio.sleep(0)
        if m := re.fullmatch(r"test -f (\S+)", cmd):
            return RunResult("", 0 if m.group(1) in self.files else 1)
        if m := re.fullmatch(r"cat (\S+)", cmd):
            if m.group(1) not in self.files:
                return RunResult(f"cat: {m.group(1)}: No such file or directory", 1)
            output = self.files[m.group(1)].decode()
            truncated = max_bytes is not None and len(output) > max_bytes
            if truncated:
                output = output[:max_bytes]
            return RunResult(output, 0, truncated=truncated)
        if m := re.fullmatch(r"printf '%s' '([A-Za-z0-9+/=]*)' \| base64 -d (>>?) (\S+)", cmd):
            data = base64.b64decode(m.group(1))
            if m.group(2) == ">>":
                data = self.files.get(m.group(3), b"") + data
            self.files[m.group(3)] = data
            return RunResult("", 0)
        if cmd.startswith("mkdir -p "):
            return RunResult("", 0)
        raise NotImplementedError(cmd)

    async def download(self, path):
        await asyncio.sleep(0)
        if path not in self.files:
            raise RuntimeError(f"Command failed: base64 {path}\nNo such file or directory")
        return self.files[path]


YELP = "SentimentAnalysisYelpReviewFullAccuracy"
APPS = "CodeGenerationAPPSPassAt5"
SUBMISSION = "/home/ubuntu/submission.csv"


def _make_env(monkeypatch, tmp_path, task_name, labels=None, files=None):
    import airs_bench
    from datasets import Dataset

    if labels is not None:
        Dataset.from_dict(labels).save_to_disk(str(tmp_path / task_name / "test_with_labels"))
    monkeypatch.setattr(airs_bench, "DATA_DIR", tmp_path)

    real_sleep = asyncio.sleep

    async def no_sleep(_):
        # Skip retry backoff but still yield to the event loop.
        await real_sleep(0)

    monkeypatch.setattr(airs_bench.asyncio, "sleep", no_sleep)
    env = airs_bench.AIRSBench(
        task_spec={"id": task_name, "task_name": task_name},
        secrets={"api_key": "test-key"},
    )
    env.sandbox = FakeSandbox(files)
    return env


def _csv(header, values):
    return ("\n".join([header] + [str(v) for v in values]) + "\n").encode()


class TestSubmit:
    @pytest.mark.asyncio
    async def test_large_submission_is_read_in_full(self, monkeypatch, tmp_path):
        # 30,000 single-digit rows is ~60 KB, past the 50 KB run() output cap.
        n = 30_000
        env = _make_env(
            monkeypatch, tmp_path, YELP,
            labels={"label": [3] * n},
            files={SUBMISSION: _csv("label", [3] * n)},
        )
        from airs_bench import SubmitParams
        result = await env.submit(SubmitParams())
        assert result.finished is True
        assert result.metadata["raw_score"] == pytest.approx(1.0)
        assert result.reward == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_missing_submission_is_not_graded(self, monkeypatch, tmp_path):
        env = _make_env(monkeypatch, tmp_path, YELP, labels={"label": [1, 2, 3]})
        from airs_bench import SubmitParams
        result = await env.submit(SubmitParams())
        assert result.finished is False
        assert result.reward == 0.0
        assert "not found" in result.blocks[0].text

        env.sandbox.files[SUBMISSION] = _csv("label", [1, 2, 3])
        result = await env.submit(SubmitParams())
        assert result.finished is True
        assert result.reward == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_row_count_mismatch_returns_feedback_and_allows_resubmit(self, monkeypatch, tmp_path):
        labels = [0, 1, 2, 3, 4] * 4
        env = _make_env(
            monkeypatch, tmp_path, YELP,
            labels={"label": labels},
            files={SUBMISSION: _csv("label", labels[:10])},
        )
        from airs_bench import SubmitParams
        result = await env.submit(SubmitParams())
        assert result.finished is False
        assert result.reward == 0.0
        assert "Row count mismatch: 10 predictions vs 20 labels" in result.blocks[0].text
        assert result.metadata["graded"] is False

        env.sandbox.files[SUBMISSION] = _csv("label", labels)
        result = await env.submit(SubmitParams())
        assert result.finished is True
        assert result.reward == pytest.approx(1.0)
        assert "already_submitted" not in result.metadata

    @pytest.mark.asyncio
    async def test_unparsable_value_returns_feedback(self, monkeypatch, tmp_path):
        env = _make_env(
            monkeypatch, tmp_path, YELP,
            labels={"label": [1, 2, 3]},
            files={SUBMISSION: _csv("label", [1, "abc", 3])},
        )
        from airs_bench import SubmitParams
        result = await env.submit(SubmitParams())
        assert result.finished is False
        assert "ValueError" in result.blocks[0].text

    @pytest.mark.asyncio
    async def test_missing_ground_truth_raises(self, monkeypatch, tmp_path):
        env = _make_env(monkeypatch, tmp_path, YELP, files={SUBMISSION: _csv("label", [1])})
        from airs_bench import SubmitParams
        with pytest.raises(FileNotFoundError):
            await env.submit(SubmitParams())

    @pytest.mark.asyncio
    async def test_concurrent_submits_grade_once(self, monkeypatch, tmp_path):
        env = _make_env(
            monkeypatch, tmp_path, YELP,
            labels={"label": [1, 2, 3]},
            files={SUBMISSION: _csv("label", [1, 2, 3])},
        )
        from airs_bench import SubmitParams
        results = await asyncio.gather(env.submit(SubmitParams()), env.submit(SubmitParams()))
        graded = [r for r in results if "raw_score" in r.metadata]
        repeats = [r for r in results if r.metadata.get("already_submitted")]
        assert len(graded) == 1 and len(repeats) == 1
        assert graded[0].reward == pytest.approx(1.0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("header,rows", [
        ("code1,code2,code3,code4,code5", ["a,b,c,d,e"] * 2),
        ("code1,code2,code3,code4", ["a,b,c,d"] * 3),
    ])
    async def test_pass_at_5_malformed_submission_returns_feedback(self, monkeypatch, tmp_path, header, rows):
        env = _make_env(
            monkeypatch, tmp_path, APPS,
            labels={"input_output": ["{}"] * 3},
            files={SUBMISSION: _csv(header, rows)},
        )
        from airs_bench import SubmitParams
        result = await env.submit(SubmitParams())
        assert result.finished is False
        assert result.reward == 0.0
        assert result.metadata["graded"] is False
        assert not any("run_eval.py" in c for c in env.sandbox.commands)

    @pytest.mark.asyncio
    async def test_upload_file_to_sandbox(self, monkeypatch, tmp_path):
        env = _make_env(monkeypatch, tmp_path, APPS)
        await env._upload_file_to_sandbox("print('hi')\n", "/tmp/eval/x.py")
        assert env.sandbox.files["/tmp/eval/x.py"] == b"print('hi')\n"


class TestReadFile:
    @pytest.mark.asyncio
    async def test_reports_truncation(self, monkeypatch, tmp_path):
        env = _make_env(
            monkeypatch, tmp_path, YELP,
            files={"/home/ubuntu/big.txt": b"x" * 60_000},
        )
        from airs_bench import ReadFileInput
        result = await env.read_file(ReadFileInput(path="/home/ubuntu/big.txt"))
        assert "truncated" in result.blocks[0].text
        assert result.metadata["truncated"] is True


# --- ground truth as deployed ---

# Columns of each task's deployed ground truth (server_data/<task>/test_with_labels),
# which is the source dataset's evaluation split as published. check_ground_truth.py
# checks the deployed data itself.
DEPLOYED_GROUND_TRUTH_COLUMNS = {
    "CodeGenerationAPPSPassAt5": ['problem_id', 'question', 'solutions', 'input_output', 'difficulty', 'url', 'starter_code'],
    "CodeRetrievalCodeXGlueMRR": ['id', 'repo', 'path', 'func_name', 'original_string', 'language', 'code', 'code_tokens', 'docstring', 'docstring_tokens', 'sha', 'url', 'docstring_summary', 'parameters', 'return_statement', 'argument_list', 'identifier', 'nwo', 'score'],
    "CoreferenceResolutionSuperGLUEWSCAccuracy": ['text', 'span1_index', 'span2_index', 'span1_text', 'span2_text', 'idx', 'label'],
    "CoreferenceResolutionWinograndeAccuracy": ['sentence', 'option1', 'option2', 'answer'],
    "CvMolecularPropertyPredictionQm9MeanAbsoluteError": ['pos', 'atomic_numbers', 'mu', 'alpha', 'eps_HOMO', 'eps_LUMO', 'delta_eps', 'R_2_Abs', 'ZPVE', 'U_0', 'U', 'H', 'G', 'c_v', 'U_0_ATOM', 'U_ATOM', 'H_ATOM', 'G_ATOM', 'A', 'B', 'C', 'tags', 'natoms', 'id', 'composition'],
    "GMolecularPropertyPredictionQm9MeanAbsoluteError": ['pos', 'atomic_numbers', 'mu', 'alpha', 'eps_HOMO', 'eps_LUMO', 'delta_eps', 'R_2_Abs', 'ZPVE', 'U_0', 'U', 'H', 'G', 'c_v', 'U_0_ATOM', 'U_ATOM', 'H_ATOM', 'G_ATOM', 'A', 'B', 'C', 'tags', 'natoms', 'id', 'composition'],
    "GraphRegressionZincMae": ['node_feat', 'edge_index', 'edge_attr', 'y', 'num_nodes'],
    "MathQuestionAnsweringSVAMPAccuracy": ['ID', 'Body', 'Question', 'Equation', 'Answer', 'Type', 'question_concat'],
    "QuestionAnsweringDuoRCAccuracy": ['plot_id', 'plot', 'title', 'question_id', 'question', 'answers', 'no_answer'],
    "QuestionAnsweringEli5Rouge1": ['q_id', 'title', 'selftext', 'document', 'subreddit', 'url', 'answers', 'title_urls', 'selftext_urls', 'answers_urls'],
    "QuestionAnsweringFinqaAccuracy": ['id', 'post_text', 'pre_text', 'question', 'answer', 'gold_evidence', 'table'],
    "R2AbsMolecularPropertyPredictionQm9MeanAbsoluteError": ['pos', 'atomic_numbers', 'mu', 'alpha', 'eps_HOMO', 'eps_LUMO', 'delta_eps', 'R_2_Abs', 'ZPVE', 'U_0', 'U', 'H', 'G', 'c_v', 'U_0_ATOM', 'U_ATOM', 'H_ATOM', 'G_ATOM', 'A', 'B', 'C', 'tags', 'natoms', 'id', 'composition'],
    "ReadingComprehensionSquadExactMatch": ['id', 'title', 'context', 'question', 'answers'],
    "SentimentAnalysisYelpReviewFullAccuracy": ['label', 'text'],
    "TextualClassificationSickAccuracy": ['id', 'sentence_A', 'sentence_B', 'label', 'relatedness_score', 'entailment_AB', 'entailment_BA', 'sentence_A_original', 'sentence_B_original', 'sentence_A_dataset', 'sentence_B_dataset'],
    "TextualSimilaritySickSpearmanCorrelation": ['id', 'sentence_A', 'sentence_B', 'label', 'relatedness_score', 'entailment_AB', 'entailment_BA', 'sentence_A_original', 'sentence_B_original', 'sentence_A_dataset', 'sentence_B_dataset'],
    "TimeSeriesForecastingKaggleWebTrafficMASE": ['start', 'target', 'feat_static_cat', 'feat_dynamic_real', 'item_id'],
    "TimeSeriesForecastingRideshareMAE": ['start', 'target', 'feat_static_cat', 'feat_dynamic_real', 'item_id'],
    "TimeSeriesForecastingSolarWeeklyMAE": ['start', 'target', 'feat_static_cat', 'feat_dynamic_real', 'item_id'],
    "U0MolecularPropertyPredictionQm9MeanAbsoluteError": ['pos', 'atomic_numbers', 'mu', 'alpha', 'eps_HOMO', 'eps_LUMO', 'delta_eps', 'R_2_Abs', 'ZPVE', 'U_0', 'U', 'H', 'G', 'c_v', 'U_0_ATOM', 'U_ATOM', 'H_ATOM', 'G_ATOM', 'A', 'B', 'C', 'tags', 'natoms', 'id', 'composition'],
}

SOLAR = "TimeSeriesForecastingSolarWeeklyMAE"
RIDESHARE = "TimeSeriesForecastingRideshareMAE"
KAGGLE = "TimeSeriesForecastingKaggleWebTrafficMASE"
CODE_RETRIEVAL = "CodeRetrievalCodeXGlueMRR"
QM9_G = "GMolecularPropertyPredictionQm9MeanAbsoluteError"
QM9_U0 = "U0MolecularPropertyPredictionQm9MeanAbsoluteError"
QM9_R2 = "R2AbsMolecularPropertyPredictionQm9MeanAbsoluteError"


def _series(n, start=0.0):
    return [float(start + i) for i in range(n)]


def _json_rows(header, rows):
    import json
    return ("\n".join([header] + ['"' + json.dumps(r) + '"' for r in rows]) + "\n").encode()


def _deployed_case(task):
    """(ground truth in the deployed schema, submission that reproduces it).

    Submissions follow each task's project description: time series rows are
    JSON lists (Solar and Rideshare: the forecast steps of one series; Kaggle:
    the full series), QM9 G and U_0 are in meV.
    """
    if task == SOLAR:
        targets = [_series(10), _series(10, 100.0)]
        gt = {"target": targets, "feat_dynamic_real": [None, None]}
        return gt, _json_rows("label_target", [t[-5:] for t in targets])
    if task == RIDESHARE:
        # Each row holds two target series and one covariate series, all forecast.
        rows = [([_series(50), _series(50, 1.0)], [_series(50, 2.0)]),
                ([_series(50, 3.0), _series(50, 4.0)], [_series(50, 5.0)])]
        gt = {"target": [t for t, _ in rows], "feat_dynamic_real": [d for _, d in rows]}
        series = [s for t, d in rows for s in t + d]
        return gt, _json_rows("label_target", [s[-48:] for s in series])
    if task == KAGGLE:
        targets = [[float((i * 7) % 5 + j) for i in range(70)] for j in range(2)]
        return {"target": targets}, _json_rows("label_target", targets)
    if task == CODE_RETRIEVAL:
        gt = {"docstring_tokens": [["sort", "a", "list"], ["read", "a", "file"]], "id": [0, 1]}
        csv = b'query,rankings\nsort a list,"[0, 1]"\nread a file,"[1, 0]"\n'
        return gt, csv
    if task in (QM9_G, QM9_U0):
        col = TASKS[task].scoring_column
        ev = [-11185.25, -10500.5, -9800.75]
        return {col: ev}, _csv(col, [v * 1000 for v in ev])
    if task == YELP:
        return {"label": [0, 4, 2]}, _csv("label", [0, 4, 2])
    raise KeyError(task)


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_grader_reads_only_deployed_columns(task_name):
    import airs_bench
    env = airs_bench.AIRSBench(
        task_spec={"id": task_name, "task_name": task_name}, secrets={"api_key": "test-key"},
    )
    missing = set(env._ground_truth_columns()) - set(DEPLOYED_GROUND_TRUTH_COLUMNS[task_name])
    assert not missing


class TestDeployedGroundTruth:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("task_name", [SOLAR, RIDESHARE, KAGGLE, CODE_RETRIEVAL, QM9_G, QM9_U0, YELP])
    async def test_correct_submission_scores_optimum(self, monkeypatch, tmp_path, task_name):
        gt, csv = _deployed_case(task_name)
        assert set(gt) <= set(DEPLOYED_GROUND_TRUTH_COLUMNS[task_name])
        env = _make_env(monkeypatch, tmp_path, task_name, labels=gt, files={SUBMISSION: csv})
        from airs_bench import SubmitParams
        result = await env.submit(SubmitParams())
        assert result.finished is True, result.blocks[0].text
        assert result.metadata["raw_score"] == pytest.approx(TASKS[task_name].optimal_score, abs=1e-9)
        assert result.reward == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_missing_ground_truth_column_raises(self, monkeypatch, tmp_path):
        # The grader's data is at fault, so the agent must not be told to fix its CSV.
        env = _make_env(
            monkeypatch, tmp_path, YELP,
            labels={"text": ["a", "b", "c"]},
            files={SUBMISSION: _csv("label", [1, 2, 3])},
        )
        from airs_bench import GroundTruthError, SubmitParams
        with pytest.raises(GroundTruthError, match=r"lacks column\(s\) \['label'\]"):
            await env.submit(SubmitParams())
        assert env.submitted is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rows,expected", [
        ([[1.0] * 5], "Row count mismatch: 1 predictions vs 2 series"),
        ([[1.0] * 4, [1.0] * 5], "Row 0: prediction has shape (4,), expected (5,)"),
    ])
    async def test_time_series_malformed_submission_returns_feedback(self, monkeypatch, tmp_path, rows, expected):
        gt, _ = _deployed_case(SOLAR)
        env = _make_env(monkeypatch, tmp_path, SOLAR, labels=gt,
                        files={SUBMISSION: _json_rows("label_target", rows)})
        from airs_bench import SubmitParams
        result = await env.submit(SubmitParams())
        assert result.finished is False
        assert result.metadata["graded"] is False
        assert expected in result.blocks[0].text

    def test_unservable_task_is_not_listed(self, tmp_path):
        # Import a copy of the env next to a server_data/ holding every task,
        # so list_tasks sees data for all of them.
        import json
        import shutil
        import subprocess
        import sys
        from pathlib import Path
        here = Path(__file__).parent
        for f in ("airs_bench.py", "evaluate.py", "task_config.py"):
            shutil.copy(here / f, tmp_path / f)
        for name in TASK_NAMES:
            (tmp_path / "server_data" / name).mkdir(parents=True)
        out = subprocess.run(
            [sys.executable, "-c",
             "import json, airs_bench; print(json.dumps([t['id'] for t in airs_bench.AIRSBench.list_tasks('test')]))"],
            cwd=tmp_path, capture_output=True, text=True, check=True,
        ).stdout
        ids = json.loads(out.strip().splitlines()[-1])
        assert KAGGLE not in ids
        assert ids == [n for n in TASK_NAMES if n != KAGGLE]

    def test_wsc_is_served_in_test_but_not_train(self, tmp_path):
        import json
        import shutil
        import subprocess
        import sys
        from pathlib import Path
        here = Path(__file__).parent
        for f in ("airs_bench.py", "evaluate.py", "task_config.py"):
            shutil.copy(here / f, tmp_path / f)
        for name in TASK_NAMES:
            (tmp_path / "server_data" / name).mkdir(parents=True)
        out = subprocess.run(
            [sys.executable, "-c",
             "import json, airs_bench; print(json.dumps({s: [t['id'] for t in airs_bench.AIRSBench.list_tasks(s)]"
             " for s in ('train', 'test')}))"],
            cwd=tmp_path, capture_output=True, text=True, check=True,
        ).stdout
        ids = json.loads(out.strip().splitlines()[-1])
        served = [n for n in TASK_NAMES if n != KAGGLE]
        assert ids["test"] == served
        assert ids["train"] == [n for n in served if n != WSC]


# --- trivial-model baselines ---

CV = "CvMolecularPropertyPredictionQm9MeanAbsoluteError"
ZINC = "GraphRegressionZincMae"
SICK_SIMILARITY = "TextualSimilaritySickSpearmanCorrelation"


def _mae_case(task, prediction, offsets):
    """Ground truth whose labels sit at the given offsets from a constant
    prediction (in the units the agent submits), and that prediction's CSV."""
    config = TASKS[task]
    col = config.scoring_column
    labels = [(prediction + o) / config.label_scale for o in offsets]
    return {col: labels}, _csv(col, [prediction] * len(offsets))


# Error of the best trivial model on each regression task's deployed ground
# truth: a constant train-set median (QM9 c_v and R_2_Abs, ZINC), the
# per-element reference model (QM9 G and U_0, in meV), each series' historical
# mean (Rideshare) or last value (Solar). check_ground_truth.py computes these
# from the deployed data.
TRIVIAL_MODEL_ERROR = {
    CV: 3.210379,
    QM9_G: 865.8952,
    QM9_R2: 198.2839,
    QM9_U0: 862.9109,
    ZINC: 1.499522,
    RIDESHARE: 1.267263,
    SOLAR: 1729.409,
}


class TestTrivialBaselineNormalisation:
    """A trivial model earns 0 on each regression task, a correct submission 1,
    and a model in between is rewarded by how much of the trivial model's error
    it removes."""

    MAE_TASKS = [CV, QM9_G, QM9_R2, QM9_U0, ZINC]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("task_name", MAE_TASKS)
    @pytest.mark.parametrize("fraction,expected", [(1.0, 0.0), (0.25, 0.75), (0.0, 1.0)])
    async def test_mae_reward(self, monkeypatch, tmp_path, task_name, fraction, expected):
        # A prediction whose MAE is `fraction` of the trivial model's.
        error = fraction * TRIVIAL_MODEL_ERROR[task_name]
        gt, csv = _mae_case(task_name, 1000.0, [error, -error])
        env = _make_env(monkeypatch, tmp_path, task_name, labels=gt, files={SUBMISSION: csv})
        from airs_bench import SubmitParams
        result = await env.submit(SubmitParams())
        assert result.finished is True, result.blocks[0].text
        assert result.reward == pytest.approx(expected, abs=1e-6)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("task_name", [SOLAR, RIDESHARE])
    @pytest.mark.parametrize("fraction,expected", [(1.0, 0.0), (0.25, 0.75), (0.0, 1.0)])
    async def test_forecast_reward(self, monkeypatch, tmp_path, task_name, fraction, expected):
        horizon = TASKS[task_name].forecast_horizon
        error = fraction * TRIVIAL_MODEL_ERROR[task_name]
        history = _series(10)
        target = history + [9.0 + error] * horizon
        if task_name == RIDESHARE:
            gt = {"target": [[target]], "feat_dynamic_real": [[]]}
        else:
            gt = {"target": [target], "feat_dynamic_real": [None]}
        csv = _json_rows("label_target", [[9.0] * horizon])
        env = _make_env(monkeypatch, tmp_path, task_name, labels=gt, files={SUBMISSION: csv})
        from airs_bench import SubmitParams
        result = await env.submit(SubmitParams())
        assert result.finished is True, result.blocks[0].text
        assert result.reward == pytest.approx(expected, abs=1e-6)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("predictions,expected", [
        ([3.0, 3.0, 3.0, 3.0], 0.0),  # constant: no ranking information
        ([1.0, 2.0, 4.0, 3.0], 0.8),  # Spearman 0.8
        ([1.0, 2.0, 3.0, 4.0], 1.0),
    ])
    async def test_spearman_reward(self, monkeypatch, tmp_path, predictions, expected):
        env = _make_env(
            monkeypatch, tmp_path, SICK_SIMILARITY,
            labels={"relatedness_score": [1.5, 2.5, 3.5, 4.5]},
            files={SUBMISSION: _csv("relatedness_score", predictions)},
        )
        from airs_bench import SubmitParams
        result = await env.submit(SubmitParams())
        assert result.finished is True, result.blocks[0].text
        assert result.reward == pytest.approx(expected, abs=1e-6)


WSC = "CoreferenceResolutionSuperGLUEWSCAccuracy"
WINOGRANDE = "CoreferenceResolutionWinograndeAccuracy"
SICK_CLASSIFICATION = "TextualClassificationSickAccuracy"
SVAMP = "MathQuestionAnsweringSVAMPAccuracy"
DUORC = "QuestionAnsweringDuoRCAccuracy"

# Label counts of each Accuracy task's deployed ground truth, majority class
# first (Yelp is balanced; SVAMP lists its most common answer and the rest).
# check_ground_truth.py computes the best constant answer from the deployed data.
DEPLOYED_LABEL_COUNTS = {
    WSC: ("label", [(0, 66), (1, 38)]),
    WINOGRANDE: ("answer", [("2", 639), ("1", 628)]),
    YELP: ("label", [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]),
    SICK_CLASSIFICATION: ("label", [(1, 2790), (0, 1404), (2, 712)]),
    SVAMP: ("Answer", [("1", 25)] + [(str(100 + i), 1) for i in range(275)]),
}


class TestConstantAnswerFloor:
    """A constant answer earns 0: the best constant (the majority class, or
    "no answer" on DuoRC) scores each task's worst score."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("task_name", list(DEPLOYED_LABEL_COUNTS))
    async def test_majority_class_earns_nothing(self, monkeypatch, tmp_path, task_name):
        col, counts = DEPLOYED_LABEL_COUNTS[task_name]
        labels = [value for value, count in counts for _ in range(count)]
        majority = counts[0][0]
        env = _make_env(monkeypatch, tmp_path, task_name, labels={col: labels},
                        files={SUBMISSION: _csv(col, [majority] * len(labels))})
        from airs_bench import SubmitParams
        result = await env.submit(SubmitParams())
        assert result.finished is True, result.blocks[0].text
        assert result.metadata["raw_score"] == pytest.approx(TASKS[task_name].estimated_worst_score, abs=1e-9)
        assert result.reward == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.asyncio
    async def test_duorc_no_answer_earns_nothing(self, monkeypatch, tmp_path):
        # 2408 of the 15857 deployed questions have no answer.
        n, unanswerable = 15857, 2408
        gt = {"answers": [[] if i < unanswerable else ["x"] for i in range(n)],
              "no_answer": [i < unanswerable for i in range(n)]}
        csv = ("answer,has_answer\n" + ",False\n" * n).encode()
        env = _make_env(monkeypatch, tmp_path, DUORC, labels=gt, files={SUBMISSION: csv})
        from airs_bench import SubmitParams
        result = await env.submit(SubmitParams())
        assert result.finished is True, result.blocks[0].text
        assert result.metadata["raw_score"] == pytest.approx(TASKS[DUORC].estimated_worst_score, abs=1e-9)
        assert result.reward == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("task_name", list(DEPLOYED_LABEL_COUNTS) + [DUORC])
    def test_checker_grades_every_constant(self, task_name):
        from check_ground_truth import trivial_submissions
        if task_name == DUORC:
            subs = trivial_submissions(task_name, Path("/nonexistent"), ([[]], [True]), 1)
            assert list(subs) == ["constant no-answer"]
            return
        col, counts = DEPLOYED_LABEL_COUNTS[task_name]
        labels = [value for value, _ in counts]
        subs = trivial_submissions(task_name, Path("/nonexistent"), labels, 1)
        assert sorted(subs) == sorted(f"constant {int(v)}" for v in labels)


class TestAtomReferenceBaseline:
    """QM9 total energies are almost a sum of per-element energies, so their
    trivial model is that sum fitted on train, not a constant."""

    # Approximate per-element energies (meV) for H, C, N, O, F.
    ELEMENT_ENERGY = {1: -16_400.0, 6: -1_036_000.0, 7: -1_490_000.0, 8: -2_047_000.0, 9: -2_717_000.0}
    TRAIN = [[6, 1, 1, 1, 1], [7, 1, 1, 1], [8, 1, 1], [9, 1], [6, 6, 8, 1, 1, 1, 1, 1, 1],
             [6, 7, 1, 1, 1, 1, 1], [6, 9, 1, 1, 1], [6, 6, 1, 1, 1, 1]]
    TEST = [[6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1], [7, 7, 1, 1, 1, 1], [6, 8, 9, 1, 1, 1]]

    def _energy(self, numbers):
        return sum(self.ELEMENT_ENERGY[z] for z in numbers)

    def test_recovers_per_element_energies(self):
        from check_ground_truth import atom_reference_predictions
        train_y = [self._energy(m) for m in self.TRAIN]
        predictions = atom_reference_predictions(self.TRAIN, train_y, self.TEST)
        assert predictions == pytest.approx([self._energy(m) for m in self.TEST], rel=1e-9)

    def test_rejects_unknown_element(self):
        from check_ground_truth import element_counts
        with pytest.raises(ValueError):
            element_counts([[6, 16, 1, 1]])

    @pytest.mark.parametrize("task_name", [QM9_G, QM9_U0])
    def test_targets_use_the_atom_reference(self, task_name):
        from check_ground_truth import ATOM_REFERENCE_TARGETS
        assert TASKS[task_name].scoring_column in ATOM_REFERENCE_TARGETS

    @pytest.mark.asyncio
    @pytest.mark.parametrize("task_name", [QM9_G, QM9_U0])
    async def test_atom_reference_model_earns_nothing(self, monkeypatch, tmp_path, task_name):
        # Labels the atom reference model misses by its deployed test-set MAE.
        error = TRIVIAL_MODEL_ERROR[task_name]
        predictions = [self._energy(m) for m in self.TEST]
        offsets = [error, -error, error]
        col = TASKS[task_name].scoring_column
        gt = {col: [(p + o) / TASKS[task_name].label_scale for p, o in zip(predictions, offsets)]}
        env = _make_env(monkeypatch, tmp_path, task_name, labels=gt, files={SUBMISSION: _csv(col, predictions)})
        from airs_bench import SubmitParams
        result = await env.submit(SubmitParams())
        assert result.finished is True, result.blocks[0].text
        assert result.metadata["raw_score"] == pytest.approx(error, rel=1e-6)
        assert result.reward == pytest.approx(0.0, abs=1e-6)


class TestSotaReference:
    """Each task's reference SOTA is a better score than its trivial baseline,
    so a SOTA-level submission earns a positive reward."""

    @pytest.mark.parametrize("task_name", TASK_NAMES)
    def test_sota_between_worst_and_optimal(self, task_name):
        config = TASKS[task_name]
        lo, hi = sorted([config.estimated_worst_score, config.optimal_score])
        assert lo < config.sota_score < hi

    @pytest.mark.asyncio
    async def test_rideshare_reward_decreases_with_error(self, monkeypatch, tmp_path):
        horizon = TASKS[RIDESHARE].forecast_horizon
        rewards = []
        for error in [0.0, TASKS[RIDESHARE].sota_score, TRIVIAL_MODEL_ERROR[RIDESHARE], 2.0]:
            gt = {"target": [[_series(10) + [9.0 + error] * horizon]], "feat_dynamic_real": [[]]}
            csv = _json_rows("label_target", [[9.0] * horizon])
            env = _make_env(monkeypatch, tmp_path / str(error), RIDESHARE, labels=gt, files={SUBMISSION: csv})
            from airs_bench import SubmitParams
            result = await env.submit(SubmitParams())
            assert result.finished is True, result.blocks[0].text
            rewards.append(result.reward)
        assert rewards[0] == pytest.approx(1.0)
        assert 0.0 < rewards[1] < 1.0
        assert rewards[2] == pytest.approx(0.0, abs=1e-6) and rewards[3] == 0.0


class TestNonFinitePredictions:
    """NaN predictions are reported back ungraded rather than scored: a NaN
    score would otherwise clamp to reward 1.0, and skipped NaN forecast steps
    would let a submission choose which steps count."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("values", [["nan"] * 3, [1.0, "nan", 2.0], [1.0, "inf", 2.0]])
    async def test_mae_nan_prediction_is_not_graded(self, monkeypatch, tmp_path, values):
        env = _make_env(
            monkeypatch, tmp_path, CV,
            labels={"c_v": [30.0, 31.0, 32.0]},
            files={SUBMISSION: _csv("c_v", values)},
        )
        from airs_bench import SubmitParams
        result = await env.submit(SubmitParams())
        assert result.finished is False
        assert result.metadata["graded"] is False
        assert "finite" in result.blocks[0].text
        assert env.submitted is False

    @pytest.mark.asyncio
    async def test_forecast_nan_steps_are_not_graded(self, monkeypatch, tmp_path):
        gt = {"target": [_series(10)], "feat_dynamic_real": [None]}
        csv = b'label_target\n"[NaN, NaN, NaN, NaN, 9.0]"\n'
        env = _make_env(monkeypatch, tmp_path, SOLAR, labels=gt, files={SUBMISSION: csv})
        from airs_bench import SubmitParams
        result = await env.submit(SubmitParams())
        assert result.finished is False
        assert result.metadata["graded"] is False
        assert "finite" in result.blocks[0].text

    @pytest.mark.asyncio
    async def test_nan_score_is_not_rewarded(self, monkeypatch, tmp_path):
        # Any metric path that still yields NaN is reported, never clamped to 1.0.
        env = _make_env(
            monkeypatch, tmp_path, CV,
            labels={"c_v": [30.0, 31.0]},
            files={SUBMISSION: _csv("c_v", [30.0, 31.0])},
        )
        monkeypatch.setattr(env, "_evaluate_submission", lambda csv, labels: float("nan"))
        from airs_bench import SubmitParams
        result = await env.submit(SubmitParams())
        assert result.finished is False
        assert result.reward == 0.0


# --- Pass@5 grading sandbox ---

import io
import json
import sys
import types


class FakeGradingSandbox(FakeSandbox):
    """Fake of a fresh grading sandbox. Like the SDK handle it must be started
    before use and can't be restarted once stopped. Running the harness needs
    every harness file and the submission to have been uploaded; it then grades
    its shard of the problems, counting a problem solved when any of its
    programs is "ok"."""

    HARNESS = ["pyext.py", "testing_util.py", "utils.py", "run_eval.py", "submission.csv"]

    def __init__(self, settings, eval_code=0, eval_output=""):
        super().__init__()
        self.settings = settings
        self.eval_code = eval_code
        self.eval_output = eval_output
        self.state = "new"

    async def start(self):
        if self.state != "new":
            raise RuntimeError("sandbox handle can't be restarted")
        self.state = "running"

    async def stop(self):
        self.state = "stopped"

    async def run(self, cmd, timeout=300, max_bytes=50_000, sanitise=True):
        if self.state != "running":
            raise RuntimeError("Sandbox not started")
        pattern = r"cd /tmp/eval && python run_eval.py (\S+) (\d+) (\d+) (\d+) 2>/tmp/eval/eval.log"
        if m := re.fullmatch(pattern, cmd):
            self.commands.append(cmd)
            missing = [f for f in self.HARNESS if f"/tmp/eval/{f}" not in self.files]
            assert not missing, missing
            assert m.group(1) == self.settings.bucket_config.mount_path
            assert timeout is not None and timeout > int(m.group(4))
            if self.eval_code:
                return RunResult(self.eval_output, self.eval_code)
            shard, n_shards = int(m.group(2)), int(m.group(3))
            rows = pd.read_csv(io.BytesIO(self.files["/tmp/eval/submission.csv"])).values.tolist()
            mine = rows[shard::n_shards]
            result = {"correct": sum("ok" in r for r in mine), "problems": len(mine), "skipped_programs": 0}
            return RunResult("program output\n" + json.dumps(result) + "\n", 0)
        if cmd == "tail -50 /tmp/eval/eval.log":
            self.commands.append(cmd)
            return RunResult("Traceback ...", 0)
        return await super().run(cmd, timeout, max_bytes, sanitise)


# Ten problems; 4 of them have an "ok" program.
PASS_AT_5_CSV = _csv(
    "code1,code2,code3,code4,code5",
    ["a,b,c,d,ok", "a,a,a,a,a", "ok,b,c,d,e", "a,b,c,d,e", "a,b,ok,ok,e",
     "a,b,c,d,e", "a,b,c,d,e", "a,b,c,d,e", "a,b,c,d,ok", "a,b,c,d,e"],
)


def _apps_env(monkeypatch, tmp_path, **grader_kwargs):
    env = _make_env(
        monkeypatch, tmp_path, APPS,
        labels={"input_output": ["{}"] * 10},
        files={SUBMISSION: PASS_AT_5_CSV},
    )
    graders = []

    def make_grader():
        graders.append(FakeGradingSandbox(_grading_settings(env), **grader_kwargs))
        return graders[-1]

    monkeypatch.setattr(env, "_pass_at_5_grading_sandbox", make_grader)
    return env, graders


def _grading_settings(env):
    """The SandboxSettings the env asks for when it creates a grading sandbox."""
    captured = []

    class Capture:
        def sandbox(self, settings):
            captured.append(settings)
            return None

    real = env.or_client
    env.or_client = Capture()
    try:
        type(env)._pass_at_5_grading_sandbox(env)
    finally:
        env.or_client = real
    return captured[0]


class TestPassAt5Grading:
    def test_agent_sandbox_mounts_only_agent_data(self, monkeypatch, tmp_path):
        env = _make_env(monkeypatch, tmp_path, APPS)
        assert env.sandbox_settings.bucket_config.only_dir == f"sandbox_data/{APPS}/data"

    def test_grading_sandbox_mounts_ground_truth_offline(self, monkeypatch, tmp_path):
        env = _make_env(monkeypatch, tmp_path, APPS)
        settings = _grading_settings(env)
        assert settings.bucket_config.only_dir == f"server_data/{APPS}/test_with_labels"
        assert not settings.bucket_config.mount_path.startswith("/home/ubuntu/data")
        assert settings.block_network is True

    def test_runner_reads_nothing_from_the_agent_mount(self):
        import airs_bench
        assert "/home/ubuntu/data" not in airs_bench._PASS_AT_5_RUNNER_SCRIPT

    def test_harness_is_shipped_with_the_env(self):
        import airs_bench
        for name in ("testing_util.py", "utils.py"):
            assert (airs_bench.APPS_EVAL_DIR / name).is_file()
        dockerfile = (airs_bench.APPS_EVAL_DIR.parent / "Dockerfile").read_text()
        assert "apps_eval" in dockerfile

    @pytest.mark.asyncio
    async def test_eval_runs_in_grading_sandboxes(self, monkeypatch, tmp_path):
        import airs_bench
        env, graders = _apps_env(monkeypatch, tmp_path)
        from airs_bench import SubmitParams
        result = await env.submit(SubmitParams())
        assert result.finished is True
        assert result.reward == pytest.approx(0.4)
        assert len(graders) == airs_bench.PASS_AT_5_GRADING_SHARDS
        assert all(g.state == "stopped" for g in graders)
        assert all(g.files["/tmp/eval/submission.csv"] == PASS_AT_5_CSV for g in graders)
        shards = sorted(re.search(r"run_eval.py \S+ (\d+) ", g.commands[-1]).group(1) for g in graders)
        assert shards == [str(k) for k in range(len(graders))]
        # Nothing but reading the submission touches the agent's sandbox.
        assert env.sandbox.commands == [f"test -f {SUBMISSION}"]

    @pytest.mark.asyncio
    async def test_failed_eval_retries_on_fresh_sandboxes_then_raises(self, monkeypatch, tmp_path):
        import airs_bench
        env, graders = _apps_env(monkeypatch, tmp_path, eval_code=1)
        from airs_bench import SubmitParams
        with pytest.raises(RuntimeError, match=r"Pass@5 evaluation failed \(exit 1\)"):
            await env.submit(SubmitParams())
        assert len(graders) == 4 * airs_bench.PASS_AT_5_GRADING_SHARDS
        assert all(g.state == "stopped" for g in graders)
        assert env.submitted is False

    @pytest.mark.asyncio
    async def test_memory_kill_is_not_retried(self, monkeypatch, tmp_path):
        # A run killed for exhausting the grading sandbox's memory would be
        # killed again on the same programs, so it raises after one round.
        import airs_bench
        env, graders = _apps_env(monkeypatch, tmp_path, eval_code=137,
                                 eval_output="Command killed: memory usage exceeded container limit")
        from airs_bench import SubmitParams
        with pytest.raises(RuntimeError, match=r"Pass@5 evaluation failed \(exit 137\)"):
            await env.submit(SubmitParams())
        assert len(graders) == airs_bench.PASS_AT_5_GRADING_SHARDS
        assert all(g.state == "stopped" for g in graders)
        assert env.submitted is False

    @pytest.mark.asyncio
    async def test_other_kill_is_retried(self, monkeypatch, tmp_path):
        import airs_bench
        env, graders = _apps_env(monkeypatch, tmp_path, eval_code=137)
        from airs_bench import SubmitParams
        with pytest.raises(RuntimeError, match=r"Pass@5 evaluation failed \(exit 137\)"):
            await env.submit(SubmitParams())
        assert len(graders) == 4 * airs_bench.PASS_AT_5_GRADING_SHARDS

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="grading sandboxes run Linux")
    def test_memory_hungry_program_fails_its_test_case(self, monkeypatch):
        """Run the real harness: a program that reserves more memory than a
        program may use fails its test case, and the next program still runs."""
        import airs_bench
        monkeypatch.syspath_prepend(str(airs_bench.APPS_EVAL_DIR))
        shim = types.ModuleType("pyext")
        exec(airs_bench._PYEXT_SHIM, shim.__dict__)
        monkeypatch.setitem(sys.modules, "pyext", shim)
        import utils

        testcases = {"input_output": json.dumps({"inputs": ["3\n"], "outputs": ["6\n"]})}
        hungry = "n = int(input())\nbuf = bytearray(4 * 2**30)\nprint(2 * n)\n"
        assert not utils.solves_testcases(hungry, testcases)
        assert utils.solves_testcases("n = int(input())\nprint(2 * n)\n", testcases)

    @pytest.mark.asyncio
    async def test_runner_dedupes_and_respects_budget(self, tmp_path):
        """Run the real runner script against a stub harness: identical
        programs run once, and once the time budget is spent no program runs."""
        import subprocess
        import sys
        import airs_bench
        from datasets import Dataset

        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        (eval_dir / "utils.py").write_text(
            "import json\n"
            "def solves_testcases(submission, testcases, verbose=False):\n"
            f"    with open({str(eval_dir / 'runs.log')!r}, 'a') as f:\n"
            "        f.write(submission + '\\n')\n"
            "    return submission == 'ok'\n"
            "def evaluate_all_testcases(subs, tests, verbose=False, max_workers=None):\n"
            "    return sum(any(solves_testcases(s, t) for s in row) for row, t in zip(subs, tests)) / len(subs)\n"
        )
        Dataset.from_dict({"input_output": ["{}"] * 4}).save_to_disk(str(tmp_path / "gt"))
        script = airs_bench._PASS_AT_5_RUNNER_SCRIPT.replace("/tmp/eval", str(eval_dir))
        (eval_dir / "run_eval.py").write_text(script)
        (eval_dir / "submission.csv").write_bytes(_csv(
            "code1,code2,code3,code4,code5",
            ["a,a,a,a,ok", "b,b,b,b,b", "ok,ok,ok,ok,ok", "c,d,c,d,c"],
        ))

        def run(shard, n_shards, budget):
            (eval_dir / "runs.log").unlink(missing_ok=True)
            out = subprocess.run(
                [sys.executable, "run_eval.py", str(tmp_path / "gt"), str(shard), str(n_shards), str(budget)],
                cwd=eval_dir, capture_output=True, text=True,
            )
            assert out.returncode == 0, out.stderr
            out = out.stdout
            log = eval_dir / "runs.log"
            return json.loads(out.strip().splitlines()[-1]), log.read_text().split() if log.exists() else []

        result, runs = run(0, 1, 3600)
        assert result == {"correct": 2, "problems": 4, "skipped_programs": 0}
        assert runs == ["a", "ok", "b", "ok", "c", "d"]
        result, runs = run(1, 2, 3600)
        assert result == {"correct": 0, "problems": 2, "skipped_programs": 0}
        assert runs == ["b", "c", "d"]
        result, runs = run(0, 1, 0)
        assert result == {"correct": 0, "problems": 4, "skipped_programs": 6}
        assert runs == []

    @pytest.mark.asyncio
    async def test_large_upload_is_chunked(self, monkeypatch, tmp_path):
        import airs_bench
        env = _make_env(monkeypatch, tmp_path, APPS)
        content = "".join(chr(32 + i % 90) for i in range(3 * airs_bench.UPLOAD_CHUNK_BYTES + 7))
        await env._upload_file_to_sandbox(content, "/tmp/eval/big.csv")
        assert env.sandbox.files["/tmp/eval/big.csv"] == content.encode()
        uploads = [c for c in env.sandbox.commands if "base64 -d" in c]
        assert len(uploads) == 4
        assert max(len(c) for c in uploads) < 1_100_000
