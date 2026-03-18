#!/usr/bin/env python3
"""
AIRS-Bench data preparation script.

Downloads the AIRS-Bench repo, downloads all HuggingFace datasets,
runs each task's prepare.py to create train/test/test_with_labels splits,
and organizes data for:
- Sandbox bucket: sandbox_data/{task_name}/data/train/ and data/test/
- Server /orwd_data: server_data/{task_name}/test_with_labels/,
  project_description.md, metadata.yaml

Usage:
    python prepare_data.py [--skip-clone] [--skip-download] [--tasks TASK1 TASK2 ...]
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
REPO_DIR = BASE_DIR / "airs-bench"
RAW_DATA_DIR = BASE_DIR / "raw_hf_data"
BUCKET_DIR = BASE_DIR / "sandbox_data"
SERVER_DIR = BASE_DIR / "server_data"

# HuggingFace datasets to download (from datasets/hf_datasets.csv)
HF_DATASETS = [
    ("ChilleD/SVAMP", "default"),
    ("Monash-University/monash_tsf", "kaggle_web_traffic"),
    ("Monash-University/monash_tsf", "rideshare"),
    ("Monash-University/monash_tsf", "solar_weekly"),
    ("Pavithree/eli5", "default"),
    ("RobZamp/sick", "default"),
    ("Yelp/yelp_review_full", "yelp_review_full"),
    ("allenai/winogrande", "winogrande_xl"),
    ("aps/super_glue", "wsc"),
    ("codeparrot/apps", "all"),
    ("dreamerdeo/finqa", "default"),
    ("google/code_x_glue_tc_nl_code_search_adv", "default"),
    ("graphs-datasets/ZINC", "default"),
    ("ibm-research/duorc", "ParaphraseRC"),
    ("nimashoghi/qm9", "default"),
    ("rajpurkar/squad", "plain_text"),
]

# Task name -> dataset path within raw_hf_data
TASK_DATASET_MAP = {
    "MathQuestionAnsweringSVAMPAccuracy": "ChilleD/SVAMP/default",
    "CoreferenceResolutionWinograndeAccuracy": "allenai/winogrande/winogrande_xl",
    "CoreferenceResolutionSuperGLUEWSCAccuracy": "aps/super_glue/wsc",
    "SentimentAnalysisYelpReviewFullAccuracy": "Yelp/yelp_review_full/yelp_review_full",
    "TextualClassificationSickAccuracy": "RobZamp/sick/default",
    "TextualSimilaritySickSpearmanCorrelation": "RobZamp/sick/default",
    "QuestionAnsweringFinqaAccuracy": "dreamerdeo/finqa/default",
    "QuestionAnsweringDuoRCAccuracy": "ibm-research/duorc/ParaphraseRC",
    "QuestionAnsweringEli5Rouge1": "Pavithree/eli5/default",
    "ReadingComprehensionSquadExactMatch": "rajpurkar/squad/plain_text",
    "CodeRetrievalCodeXGlueMRR": "google/code_x_glue_tc_nl_code_search_adv/default",
    "CodeGenerationAPPSPassAt5": "codeparrot/apps/all",
    "CvMolecularPropertyPredictionQm9MeanAbsoluteError": "nimashoghi/qm9/default",
    "GMolecularPropertyPredictionQm9MeanAbsoluteError": "nimashoghi/qm9/default",
    "R2AbsMolecularPropertyPredictionQm9MeanAbsoluteError": "nimashoghi/qm9/default",
    "U0MolecularPropertyPredictionQm9MeanAbsoluteError": "nimashoghi/qm9/default",
    "GraphRegressionZincMae": "graphs-datasets/ZINC/default",
    "TimeSeriesForecastingKaggleWebTrafficMASE": "Monash-University/monash_tsf/kaggle_web_traffic",
    "TimeSeriesForecastingRideshareMAE": "Monash-University/monash_tsf/rideshare",
    "TimeSeriesForecastingSolarWeeklyMAE": "Monash-University/monash_tsf/solar_weekly",
}


def clone_repo():
    """Clone the AIRS-Bench repository."""
    if REPO_DIR.exists():
        logger.info("AIRS-Bench repo already exists, pulling latest...")
        subprocess.run(["git", "pull"], cwd=REPO_DIR, check=True)
    else:
        logger.info("Cloning AIRS-Bench repo...")
        subprocess.run(
            ["git", "clone", "https://github.com/facebookresearch/airs-bench.git",
             str(REPO_DIR)],
            check=True,
        )


def download_datasets():
    """Download all HuggingFace datasets."""
    from datasets import load_dataset

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name, config in HF_DATASETS:
        save_path = RAW_DATA_DIR / dataset_name / config
        if save_path.exists():
            logger.info(f"Dataset {dataset_name}/{config} already exists, skipping")
            continue

        logger.info(f"Downloading {dataset_name} (config={config})...")
        try:
            ds = load_dataset(dataset_name, config, trust_remote_code=True)
            save_path.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(save_path))
            logger.info(f"  Saved to {save_path}")
        except Exception as e:
            logger.error(f"  Failed to download {dataset_name}/{config}: {e}")


def prepare_task(task_name: str):
    """
    Run the task's prepare.py to create train/test splits,
    then organize into sandbox_data and server_data.
    """
    from datasets import load_from_disk

    from task_config import TASKS

    config = TASKS[task_name]
    dataset_path = TASK_DATASET_MAP[task_name]
    raw_ds_path = RAW_DATA_DIR / dataset_path

    if not raw_ds_path.exists():
        logger.error(f"Raw dataset not found for {task_name} at {raw_ds_path}")
        return

    # Paths for output
    bucket_task_dir = BUCKET_DIR / task_name / "data"
    server_task_dir = SERVER_DIR / task_name

    # Check if task's prepare.py exists in the cloned repo
    repo_task_dir = REPO_DIR / "airsbench" / "tasks" / "rad" / task_name
    prepare_py = repo_task_dir / "prepare.py"

    prepared_via_script = False
    if prepare_py.exists():
        # Try the original prepare.py first
        logger.info(f"Running {task_name}/prepare.py from repo...")
        agent_data_dir = bucket_task_dir
        agent_data_dir.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            [
                sys.executable, str(prepare_py),
                "--global-shared-data-dir", str(RAW_DATA_DIR),
                "--agent-data-mount-dir", str(agent_data_dir),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            prepared_via_script = True
        else:
            logger.warning(
                f"  Original prepare.py failed for {task_name}, "
                f"falling back to generic prepare. Error: {result.stderr[-500:] if result.stderr else 'unknown'}"
            )

    if not prepared_via_script:
        # Generic prepare based on task_config
        logger.info(f"Using generic prepare for {task_name}...")
        bucket_task_dir.mkdir(parents=True, exist_ok=True)

        ds = load_from_disk(str(raw_ds_path))

        train_split = config.train_split
        test_split = config.test_split

        train = ds[train_split]
        test = ds[test_split]

        # Save full test with labels for server-side evaluation
        server_task_dir.mkdir(parents=True, exist_ok=True)
        test.save_to_disk(str(server_task_dir / "test_with_labels"))

        # Remove scoring columns from test set
        if config.columns_to_remove:
            # Only remove columns that actually exist
            cols_to_remove = [c for c in config.columns_to_remove if c in test.column_names]
            if cols_to_remove:
                test = test.remove_columns(cols_to_remove)

        # Save splits for agent (bucket)
        train.save_to_disk(str(bucket_task_dir / "train"))
        test.save_to_disk(str(bucket_task_dir / "test"))

    # Ensure test_with_labels exists in server_data
    if not (server_task_dir / "test_with_labels").exists():
        server_task_dir.mkdir(parents=True, exist_ok=True)
        ds = load_from_disk(str(raw_ds_path))
        test = ds[config.test_split]
        test.save_to_disk(str(server_task_dir / "test_with_labels"))

    # Copy project_description.md and metadata.yaml from repo
    if repo_task_dir.exists():
        for fname in ["project_description.md", "metadata.yaml"]:
            src = repo_task_dir / fname
            dst = server_task_dir / fname
            if src.exists() and not dst.exists():
                server_task_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))
                logger.info(f"  Copied {fname} for {task_name}")

    # Copy evaluate.py and utils.py from repo (needed for tasks with custom evaluation)
    for fname in ["evaluate.py", "utils.py", "custom_labels.py", "evaluate_prepare.py"]:
        src = repo_task_dir / fname
        dst = server_task_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(str(src), str(dst))

    logger.info(f"  Prepared {task_name}")


def main():
    parser = argparse.ArgumentParser(description="Prepare AIRS-Bench data")
    parser.add_argument("--skip-clone", action="store_true", help="Skip cloning the repo")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading datasets")
    parser.add_argument("--tasks", nargs="*", help="Only prepare these tasks (default: all)")
    args = parser.parse_args()

    # Step 1: Clone repo
    if not args.skip_clone:
        clone_repo()

    # Step 2: Download datasets
    if not args.skip_download:
        download_datasets()

    # Step 3: Prepare each task
    tasks_to_prepare = args.tasks or list(TASK_DATASET_MAP.keys())

    for task_name in tasks_to_prepare:
        if task_name not in TASK_DATASET_MAP:
            logger.warning(f"Unknown task: {task_name}, skipping")
            continue
        try:
            prepare_task(task_name)
        except Exception as e:
            logger.error(f"Failed to prepare {task_name}: {e}")

    logger.info("Data preparation complete!")
    logger.info(f"  Bucket data: {BUCKET_DIR}")
    logger.info(f"  Server data: {SERVER_DIR}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Upload sandbox_data/ to OpenReward bucket under 'airs-bench/'")
    logger.info("  2. Deploy server with server_data/ mounted at /orwd_data")


if __name__ == "__main__":
    main()
