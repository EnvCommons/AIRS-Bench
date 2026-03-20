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

## Reward Structure

Raw metric values are returned as rewards. Metadata includes `lower_is_better` and `metric` fields so the platform can interpret the score correctly. Accuracy-type metrics range 0-1 (higher is better); MAE/MASE metrics are unbounded (lower is better).

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

Varies by task. NLP tasks with pre-trained models are easier; molecular property prediction and time series forecasting are harder. SOTA scores range from 0.059 (ZINC MAE) to 0.962 (SuperGLUE WSC Accuracy).

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
