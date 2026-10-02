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
        if m := re.fullmatch(r"printf '%s' '([A-Za-z0-9+/=]*)' \| base64 -d > (\S+)", cmd):
            self.files[m.group(2)] = base64.b64decode(m.group(1))
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
             "import json, airs_bench; print(json.dumps([t['id'] for t in airs_bench.AIRSBench.list_tasks('train')]))"],
            cwd=tmp_path, capture_output=True, text=True, check=True,
        ).stdout
        ids = json.loads(out.strip().splitlines()[-1])
        assert KAGGLE not in ids
        assert ids == [n for n in TASK_NAMES if n != KAGGLE]


class TestR2AbsNormalisation:
    """The worst score for R_2_Abs is the MAE of predicting the train mean, so
    a trivial model scores 0 and a good model's score still varies with its MAE."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("offset,expected", [
        (201.874, 0.0),
        (3.0, 1 - 3.0 / 201.874),
        (0.0, 1.0),
    ])
    async def test_reward_for_mae(self, monkeypatch, tmp_path, offset, expected):
        labels = [1000.0, 1200.0, 1400.0]
        env = _make_env(
            monkeypatch, tmp_path, QM9_R2,
            labels={"R_2_Abs": labels},
            files={SUBMISSION: _csv("R_2_Abs", [v + offset for v in labels])},
        )
        from airs_bench import SubmitParams
        result = await env.submit(SubmitParams())
        assert result.metadata["raw_score"] == pytest.approx(offset)
        assert result.reward == pytest.approx(expected, abs=1e-6)
