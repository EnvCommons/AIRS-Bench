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
