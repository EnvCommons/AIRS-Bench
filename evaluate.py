"""
AIRS-Bench evaluation metrics.

Re-implements the evaluation logic from the original AIRS-Bench repo
without importing external evaluation frameworks (torch, sktime) where possible.
Each metric function takes (predictions, labels, **kwargs) and returns a float score.
"""

import ast
import json
import logging

import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Accuracy variants
# ---------------------------------------------------------------------------

def accuracy_int(predictions, labels) -> float:
    """
    Integer accuracy: int(pred) == int(label).
    Used for SVAMP, Winogrande, SuperGLUE WSC.
    """
    correct = np.fromiter(
        (int(p) == int(y) for p, y in zip(predictions, labels)),
        dtype=bool,
    )
    return float(correct.mean())


def accuracy_eq(predictions, labels) -> float:
    """
    Equality accuracy after int conversion if string.
    Used for Yelp, SICK Classification.
    """
    labels = [int(x) if isinstance(x, str) else x for x in labels]
    predictions = [int(x) if isinstance(x, str) else x for x in predictions]
    return float(np.mean(np.array(predictions) == np.array(labels)))


def finqa_accuracy(predictions, labels) -> float:
    """
    FinQA accuracy with numeric tolerance and currency/percent handling.
    Matching rules:
      - If both parse as numbers (after stripping currency symbols, commas,
        handling parenthetical negatives, converting percents), compare
        numerically with tolerance.
      - Otherwise, compare normalized strings (lowercased, trimmed).
    """
    ABS_TOL = 1e-4
    REL_TOL = 1e-4

    def is_nan(x):
        return x is None or (isinstance(x, float) and np.isnan(x))

    def normalize_text(s: str) -> str:
        s = str(s).strip().lower()
        return " ".join(s.split())

    def to_number(s: str):
        if s is None:
            return None
        ss = str(s).strip()
        if ss == "":
            return None
        neg = False
        if ss.startswith("(") and ss.endswith(")"):
            neg = True
            ss = ss[1:-1].strip()
        ss = ss.replace("$", "").replace("£", "").replace("€", "")
        ss = ss.replace(",", "").replace(" ", "")
        is_percent = False
        if ss.endswith("%"):
            is_percent = True
            ss = ss[:-1]
        try:
            val = float(ss)
        except Exception:
            return None
        if neg:
            val = -val
        if is_percent:
            val = val / 100.0
        return val

    preds = ["" if is_nan(p) else str(p) for p in np.asarray(predictions, dtype=object)]
    gts = ["" if is_nan(t) else str(t) for t in np.asarray(labels, dtype=object)]

    if len(preds) != len(gts):
        raise ValueError(
            f"Number of predictions ({len(preds)}) does not match "
            f"number of labels ({len(gts)})."
        )

    correct = 0
    n = len(gts)
    for p, t in zip(preds, gts):
        pn = to_number(p)
        tn = to_number(t)
        if pn is not None and tn is not None:
            if abs(pn - tn) <= max(ABS_TOL, REL_TOL * max(1.0, abs(tn))):
                correct += 1
            continue
        if normalize_text(p) == normalize_text(t):
            correct += 1

    return correct / n if n > 0 else 0.0


def duorc_accuracy(submission_answers, submission_has_answers,
                   label_answers, label_no_answers) -> float:
    """
    DuoRC accuracy with case-insensitive matching and no_answer handling.
    Submission has two columns: answer, has_answer.
    """
    correct = 0
    total = 0
    for sub_answer, sub_has_answer, lbl_answers, lbl_no_answer in zip(
        submission_answers, submission_has_answers, label_answers, label_no_answers
    ):
        total += 1
        if not sub_has_answer:
            correct += lbl_no_answer
        else:
            sub_lower = str(sub_answer).lower()
            for candidate in lbl_answers:
                if candidate.lower() == sub_lower:
                    correct += 1
                    break
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Exact Match (SQuAD)
# ---------------------------------------------------------------------------

def exact_match(predictions, labels) -> float:
    """
    SQuAD-style exact match: checks if prediction is in the list of valid answers.
    Labels should be a list of lists of acceptable answer strings.
    """
    matches = 0
    for pred, label_list in zip(predictions, labels):
        pred_str = str(pred)
        # Handle edge case from original code
        if pred_str == '"None"':
            pred_str = "None"
        if pred_str in label_list:
            matches += 1
    return matches / len(labels) if len(labels) > 0 else 0.0


# ---------------------------------------------------------------------------
# Mean Absolute Error
# ---------------------------------------------------------------------------

def mae(predictions, labels) -> float:
    """
    Mean Absolute Error for regression tasks (QM9, ZINC).
    Handles string predictions (e.g. "[0.95]") via ast.literal_eval.
    """
    clean_preds = []
    for p in predictions:
        if isinstance(p, str):
            parsed = ast.literal_eval(p)
            if isinstance(parsed, list):
                clean_preds.append(parsed[0])
            else:
                clean_preds.append(float(parsed))
        else:
            clean_preds.append(float(p))

    y_true = np.asarray(labels, dtype=float)
    y_pred = np.asarray(clean_preds, dtype=float)

    # Squeeze trailing singleton dims
    if y_pred.ndim > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.squeeze(-1)
    if y_true.ndim > 1 and y_true.shape[1] == 1:
        y_true = y_true.squeeze(-1)

    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch: predictions {y_pred.shape} vs labels {y_true.shape}"
        )

    return float(np.mean(np.abs(y_pred - y_true)))


# ---------------------------------------------------------------------------
# Rouge-1
# ---------------------------------------------------------------------------

def rouge1(predictions, labels) -> float:
    """
    ROUGE-1 F-measure using rouge_score library.
    Used for ELI5.
    """
    from rouge_score import rouge_scorer, scoring

    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    agg = scoring.BootstrapAggregator()
    for pred, ref in zip(predictions, labels):
        # rouge_score expects (reference, prediction)
        agg.add_scores(scorer.score(str(ref), str(pred)))
    result = agg.aggregate()
    return float(result["rouge1"].mid.fmeasure)


# ---------------------------------------------------------------------------
# Spearman Correlation
# ---------------------------------------------------------------------------

def spearman_correlation(predictions, labels) -> float:
    """
    Spearman rank correlation coefficient.
    Used for SICK Textual Similarity.
    """
    preds = np.asarray(predictions, dtype=float)
    lbls = np.asarray(labels, dtype=float)
    return float(spearmanr(preds, lbls).correlation)


# ---------------------------------------------------------------------------
# Mean Reciprocal Rank (MRR)
# ---------------------------------------------------------------------------

def mrr(predictions_df, labels_ds) -> float:
    """
    Mean Reciprocal Rank for code retrieval.
    predictions_df: DataFrame with columns 'query' and 'rankings' (JSON list of IDs)
    labels_ds: HF Dataset with columns 'query' and 'id'
    """
    # Build ground truth mapping: query -> correct id
    gt = {}
    for q, correct_id in zip(labels_ds["query"], labels_ds["id"]):
        gt[q] = correct_id

    reciprocal_ranks = []
    for _, row in predictions_df.iterrows():
        query = row["query"]
        rankings = json.loads(row["rankings"]) if isinstance(row["rankings"], str) else row["rankings"]
        correct_id = gt.get(query)
        if correct_id is None:
            reciprocal_ranks.append(0.0)
            continue
        try:
            rank = rankings.index(correct_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)

    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


# ---------------------------------------------------------------------------
# MASE (Mean Absolute Scaled Error)
# ---------------------------------------------------------------------------

def mase(predictions, label_targets, train_targets) -> float:
    """
    Mean Absolute Scaled Error for time series forecasting.
    Re-implements sktime's mean_absolute_scaled_error with numpy.

    For each sample:
      MASE = mean(|y_true - y_pred|) / mean(|y_train[t] - y_train[t-1]|)

    predictions: list of prediction strings (JSON arrays)
    label_targets: list of full target sequences (train + forecast)
    train_targets: list of training portion sequences
    """
    mases = []
    for pred_str, label, train_target in zip(predictions, label_targets, train_targets):
        # Parse prediction string
        try:
            pred_str_clean = str(pred_str).replace("NaN", "None").replace("nan", "None")
            pred = np.array(ast.literal_eval(pred_str_clean), dtype=float)
        except Exception as e:
            raise ValueError(f"Error parsing prediction: {pred_str}, error: {e}") from e

        label = np.array(label, dtype=float)
        train_target = np.array(train_target, dtype=float)
        train_size = train_target.shape[0]

        # Remove leading NaNs from train_target
        train_target = train_target[~np.isnan(train_target)]

        # Extract forecast portion
        pred = pred[train_size:]
        label = label[train_size:]

        # Mask NaN values in labels
        mask = ~np.isnan(label)
        pred = pred[mask]
        label = label[mask]

        # Skip samples with all NaN labels
        if label.shape[0] == 0:
            continue

        # Compute MASE: MAE of forecast / MAE of naive forecast on training data
        forecast_mae = np.mean(np.abs(label - pred))
        naive_mae = np.mean(np.abs(np.diff(train_target)))

        if naive_mae == 0:
            # Avoid division by zero; constant series
            mases.append(0.0 if forecast_mae == 0 else float("inf"))
        else:
            mases.append(forecast_mae / naive_mae)

    return float(np.mean(mases)) if mases else 0.0


# ---------------------------------------------------------------------------
# Time Series MAE (Rideshare, Solar)
# ---------------------------------------------------------------------------

def time_series_mae(predictions, labels, train_targets=None, forecast_horizon=None) -> float:
    """
    MAE for time series forecasting tasks (Rideshare, Solar).
    Predictions are JSON strings of forecast arrays.
    Labels may be full sequences (train + forecast) or just forecast portions.

    If train_targets is provided, extracts forecast portion from labels.
    """
    all_preds = []
    all_labels = []

    for i, (pred_str, label) in enumerate(zip(predictions, labels)):
        # Parse prediction
        pred_clean = str(pred_str).replace("NaN", "null").replace("nan", "null")
        try:
            pred_list = json.loads(pred_clean)
            pred_list = [np.nan if x is None else x for x in pred_list]
            pred = np.array(pred_list, dtype=float)
        except (json.JSONDecodeError, TypeError):
            pred = np.array(ast.literal_eval(str(pred_str)), dtype=float)

        label = np.array(label, dtype=float)

        # Extract forecast portion if train_targets provided
        if train_targets is not None:
            train_size = np.array(train_targets[i]).shape[0]
            label = label[train_size:]

        all_preds.append(pred)
        all_labels.append(label)

    all_preds_flat = np.concatenate(all_preds)
    all_labels_flat = np.concatenate(all_labels)

    # Remove NaN values
    valid_mask = ~(np.isnan(all_preds_flat) | np.isnan(all_labels_flat))
    if not np.any(valid_mask):
        raise ValueError("No valid (non-NaN) data points found for evaluation")

    return float(np.mean(np.abs(all_preds_flat[valid_mask] - all_labels_flat[valid_mask])))


# ---------------------------------------------------------------------------
# Metric dispatch
# ---------------------------------------------------------------------------

# Maps metric names from task_config to evaluation functions.
# Some metrics need special handling (DuoRC, MRR, MASE, time series)
# and are called directly in the environment class.
METRIC_FUNCTIONS = {
    "Accuracy": accuracy_int,
    "AccuracyEq": accuracy_eq,
    "ExactMatch": exact_match,
    "MAE": mae,
    "Rouge1": rouge1,
    "SpearmanCorrelation": spearman_correlation,
}
