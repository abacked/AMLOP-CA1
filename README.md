# AMLOP CA1 — Fraud Detection

Team workspace for ST1519 Agile MLOps CA1 (AY2026/27 Semester 2). The project studies fraudulent orders on the fictional SP-buy platform and develops reproducible data, model, and analytics pipelines.

## Assignment milestones

| Milestone | Deadline (Singapore time) |
| --- | --- |
| Part 1 group product backlog | 16 Nov 2026, 9:00 am |
| Part 2 group and individual submissions | 30 Nov 2026, 9:00 am |
| Interview and live demo | Week 7; tutor to confirm slot |

Submission is through Brightspace. A GitHub repository or Project board is a work area, not the final submission by itself.

## Repository layout

| Path | Purpose |
| --- | --- |
| `notebooks/` | Group data exploration and cleaning notebook |
| `src/amlop_ca1/` | Reusable data, feature, training, and ETL code |
| `tests/` | Checks for pipeline behavior and schema assumptions |
| `infra/` | Deployment configuration for MLflow and database setup |
| `docs/` | Decisions, runbooks, and demo notes |
| `.github/` | GitHub issue and pull request templates |

The three supplied base CSVs are stored in `data/` with Git LFS. Install Git LFS before cloning or pulling so you receive the CSV contents rather than pointer files. The assessment PDF is not committed. Future batch files remain ignored by default; record their source URL, batch suffix, and processing outcome in ETL logs.

## Python setup

Use Python 3.12 (the tested version). From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pytest -q
```

On Linux or macOS, from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH=src
python -m pytest -q
```

`requirements.txt` is the shared environment for cleaning, EDA, feature preprocessing and tests. It includes the notebook dependencies from main and pins the tested direct dependencies. Pip resolves transitive dependencies. The separate feature dependency manifests have been consolidated into this file.

## Feature engineering and cleaning

Import `OrderFeatures` and `make_feature_pipeline` from `amlop_ca1.features`. Use `cleaning.load_batch` and `cleaning.clean_and_merge` to prepare labelled batches with a `RejectLog`. The cleaner uses `(country_code, order_id)` and `(country_code, customer_id)` keys and reports rejected/unmatched rows. For label-free inference, use `cleaning.load_csv` and `cleaning.clean_orders` on the incoming orders file. Labels and customers are not required by the feature pipeline.

After cleaning, split rows by order date before fitting preprocessing. Investigate or quarantine rows with unknown dates before assigning temporal folds. Fit the pipeline on training rows and use only `transform` for validation, test and inference. Retain the cleaned row order and keys outside the predictor matrix to align labels and predictions. Save the fitted preprocessing pipeline with the model.

The explicit predictor allowlist contains `order_value`, `num_items_ordered`, `country_code`, `collect_type`, `payment_method`, `value_per_item` and `order_weekday`. The shared `cleaning.parse_dates` parser supports the same date formats in cleaning and feature building. Feature-level numeric checks retain the cleaner's nonnegative/whole-count rules and add type/overflow guards for direct inference callers.

`order_month` is excluded. The base batch spans 12 November 2022 to 11 May 2023 and has no observations for June–October. Month can identify the collection period; we have no temporal validation showing that it improves future-order prediction. Weekday retains a recurring within-week signal. Existing models/pipelines need refitting if they used the previous month-inclusive feature schema.

### Customer feature eligibility

No customer columns enter this baseline. The three order categories above are eligible, including the order's `country_code`. The customer snapshot does not establish what was known before shipment:

| Customer information | Eligibility decision |
| --- | --- |
| `mobile_verified` | Excluded until verification status at the prediction time can be established. A current snapshot could include later verification. |
| `first_order_datetime`, derived account age, `first_order_missing` and `first_order_invalid` | Excluded until source timing and backfill behaviour are verified. Deriving age from two dates alone does not establish historical availability. |
| Last-50-day activity/payment fields and `num_associated_customers` | Excluded until an as-of timestamp and a point-in-time calculation are available. |
| Order/customer identifiers, `is_fraud` and `refund_value` | Excluded from predictors. Keep identifiers for joins/alignment, the target for training/evaluation, and refunds for retrospective analysis. |

Changes to this allowlist require a verified prediction-time source contract and temporal validation before adding model versions. Current model development can use the documented order-only baseline.

The feature builder preserves row order. Pass canonical category values from cleaning. Skip empty batches after cleaning. Invalid numeric types become missing; finite numbers exceeding `1e100` raise an error before imputation or scaling. This ceiling protects numerical operations and does not define a business maximum for order values.

This implements the order-feature preprocessing portion of PBI-02 with the merged cleaner. Skew-transform experiments, model class weighting, temporal model evaluation and full ETL/retraining integration remain later work; PBI-02 should stay open.

### Cleaning integration tests

`tests/test_cleaning_features_integration.py` imports the merged `amlop_ca1.cleaning` module directly and runs automatically with the full suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -W error
```

The five integration tests cover a synthetic three-file CSV batch, composite keys, nullable values, dates, rejected rows, row alignment, orders-only inference, predictor exclusions and training-only fitted state. There is no external-cleaner path or integration skip. The cleaning tests also read the base CSVs from `data/`; download their Git LFS contents before running the suite.

## Team workflow

1. Refine the Product Backlog Items in GitHub Issues during the Week 3 requirements meeting. Set estimates with the team.
2. Add each issue to the GitHub Project. During Week 4 sprint planning, break PBIs into assigned tasks.
3. Create a branch for each task (`feature/<issue-number>-short-name`), open a pull request, and request a teammate review.
4. Record model experiment runs, metrics, and registry decisions in MLflow. Keep infrastructure endpoints and secrets in environment variables, never in Git.
5. Before the interview, rehearse the new-batch ETL and individual analytics flows against permitted test data. Only process tutor-provided interview URLs when the tutor supplies them at the interview.

See [docs/project-plan.md](docs/project-plan.md) for the initial backlog and acceptance criteria.

GitHub Project: [AMLOP CA1 - Fraud Detection](https://github.com/users/abacked/projects/4). Its `Product backlog` table holds the PBIs and estimates; `Sprint board` tracks their progress.
