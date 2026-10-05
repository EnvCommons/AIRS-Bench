# AIRS-Bench

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/AIRS-Bench)

## Description

**AIRS-Bench** is an environment for evaluating LLM agents' ability to perform end-to-end AI research. Given a problem description and dataset, the agent must build a model or solution and produce predictions (submission.csv). Tasks span NLP, code generation, math, molecular property prediction, graph ML, and time series forecasting.

## Capabilities

- End-to-end AI research: problem understanding, data exploration, model building, and prediction
- Multi-domain evaluation across 6 categories and 20 tasks
- Server-side evaluation with task-specific metrics preventing label leakage

## Compute Requirements

Each agent sandbox runs with 1 CPU and 2GB RAM. Network access is enabled for package installation. No GPU required for the sandbox (agents work with CPU-friendly approaches or pre-trained models).

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

## Tasks

20 tasks across 6 categories, available in both `train` and `test` splits:

| Category | Task | Metric |
|----------|------|--------|
| Math | MathQuestionAnsweringSVAMPAccuracy | Accuracy |
| NLP | CoreferenceResolutionWinograndeAccuracy | Accuracy |
| NLP | CoreferenceResolutionSuperGLUEWSCAccuracy | Accuracy |
| NLP | SentimentAnalysisYelpReviewFullAccuracy | Accuracy |
| NLP | TextualClassificationSickAccuracy | Accuracy |
| NLP | TextualSimilaritySickSpearmanCorrelation | Spearman |
| NLP | QuestionAnsweringFinqaAccuracy | Accuracy |
| NLP | QuestionAnsweringDuoRCAccuracy | DuoRC Accuracy |
| NLP | QuestionAnsweringEli5Rouge1 | Rouge-1 |
| NLP | ReadingComprehensionSquadExactMatch | ExactMatch |
| Code | CodeRetrievalCodeXGlueMRR | MRR |
| Code | CodeGenerationAPPSPassAt5 | Pass@5 |
| Molecules | CvMolecularPropertyPredictionQm9MeanAbsoluteError | MAE |
| Molecules | GMolecularPropertyPredictionQm9MeanAbsoluteError | MAE |
| Molecules | R2AbsMolecularPropertyPredictionQm9MeanAbsoluteError | MAE |
| Molecules | U0MolecularPropertyPredictionQm9MeanAbsoluteError | MAE |
| Graph | GraphRegressionZincMae | MAE |
| TimeSeries | TimeSeriesForecastingKaggleWebTrafficMASE | MASE |
| TimeSeries | TimeSeriesForecastingRideshareMAE | MAE |
| TimeSeries | TimeSeriesForecastingSolarWeeklyMAE | MAE |

TimeSeriesForecastingKaggleWebTrafficMASE is not served: a submission is 145,063 full-length series (about 1 GB of CSV), too large to grade in the environment server. `list_tasks` returns the other 19.

`check_ground_truth.py` checks each task's ground truth against its grader: the columns it reads exist, a submission reproducing the labels scores the optimum, the agent's test split has one row per label, and each regression task's worst score is the score of its best trivial model.

## Reward Structure

The raw metric is normalised to a reward in [0, 1]: `(worst - raw) / (worst - optimal)`, clipped, where `optimal` is the metric's best value and `worst` is the task's `estimated_worst_score`. For the regression tasks `worst` is the score of the best trivial model, so a trivial model earns 0:

- QM9 G and U_0 (MAE, meV): the per-element reference model, a sum of per-element energies over each molecule's H, C, N, O and F counts, fitted on train by least squares. Total energies are almost linear in composition, so a constant is not a meaningful baseline for them; the reward measures how much of the per-element model's error a submission removes.
- QM9 c_v and R_2_Abs, and ZINC (MAE): a constant prediction of the train-set median, the constant that minimises absolute error.
- Rideshare and Solar (MAE): the stronger of the naive forecast (each series' last observed value) and each series' historical mean.
- SICK relatedness (Spearman): 0, the correlation of a constant prediction.

Predictions must be finite: a submission with NaN or infinite predictions is not graded and can be fixed and resubmitted. The raw metric, the reward, `metric` and `lower_is_better` are returned in the metadata.

Pass@5 (APPS) runs the submitted programs against the hidden test cases in a separate, network-blocked grading sandbox that mounts the task's ground truth; the agent's sandbox never holds the hidden tests.

## Data

- **Source**: 16 HuggingFace datasets
- **Format**: HuggingFace datasets format, mounted at `/home/ubuntu/data/{train,test}/`
- **Test labels**: Stripped from agent-visible data; held server-side for evaluation

## Tools

| Tool | Description |
|------|-------------|
| `bash` | Execute commands in the sandbox |
| `list_files` | List directory contents |
| `read_file` | Read file content (50KB limit) |
| `write_file` | Write content to a file |
| `submit` | Submit predictions for evaluation (terminal) |
| `todo_write` | Plan and track progress |

## Time Horizon

Multi-turn. Agents typically need 20-100+ tool calls to explore data, write code, train models, and produce predictions.

## Environment Difficulty

Varies by task. NLP tasks with pre-trained models are easier; molecular property prediction and time series forecasting are harder. Each task's reference SOTA score, from its AIRS-Bench task metadata, is `sota_score` in `task_config.py`; the reward does not use it.

## Other Environment Requirements

- OpenReward API key (for sandbox access)
- No other external API keys required

## Safety

Sandboxed execution environment. Network access is enabled for package installation but agents cannot access external services beyond PyPI/conda.

## Citations

```bibtex
@article{lupidi2026airsbenchsuitetasksfrontier,
      title={AIRS-Bench: a Suite of Tasks for Frontier AI Research Science Agents},
      author={Alisia Lupidi and Bhavul Gauri and Thomas Simon Foster and Bassel Al Omari and Despoina Magka and Alberto Pepe and Alexis Audran-Reiss and Muna Aghamelu and Nicolas Baldwin and Lucia Cipolina-Kun and Jean-Christophe Gagnon-Audet and Chee Hau Leow and Sandra Lefdal and Hossam Mossalam and Abhinav Moudgil and Saba Nazir and Emanuel Tewolde and Isabel Urrego and Jordi Armengol Estape and Amar Budhiraja and Gaurav Chaurasia and Abhishek Charnalia and Derek Dunfield and Karen Hambardzumyan and Daniel Izcovich and Martin Josifoski and Ishita Mediratta and Kelvin Niu and Parth Pathak and Michael Shvartsman and Edan Toledo and Anton Protopopov and Roberta Raileanu and Alexander Miller and Tatiana Shavrina and Jakob Foerster and Yoram Bachrach},
      year={2026},
      eprint={2602.06855},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.06855},
}
```
