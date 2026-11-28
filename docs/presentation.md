# Interview presentation plan (#59)

Ten minutes maximum, then the live demo. Numbers marked **TBD** are filled in once the task behind them is done; everything else is from actual results.

## Slides and timing

| # | Slide | Time | Content |
| --- | --- | --- | --- |
| 1 | Team and problem | 0:30 | SP-buy fraud orders; Trust & Safety reviews flagged orders by hand. Goal: flag risky orders before shipment |
| 2 | Data | 0:30 | 3 base files, 949,871 merged orders, 12 Nov 2022 to 11 May 2023. 11% fraud (8.1 non-fraud per fraud) |
| 3 | Cleaning findings | 1:00 | Lowercase country codes, "yes" in `mobile_verified`, 21,501 missing `payment_method` (all MY), spelling variants, 3 date formats, `order_id` repeated across countries (join on country + id), IDs with leading zeros, labels with no order/customer (rejected and logged). All in `cleaning.py` |
| 4 | Features and transformations | 1:00 | Order-only allowlist: value, items, value per item, country, collect type, payment method, weekday. Excluded: customer history (no as-of time), `refund_value` (leakage), `order_month` (only Nov to May seen). Median impute, one-hot, scaling, all fitted on training rows only |
| 5 | Model iterations | 1:30 | Split by order date 70/15/15, scored on later orders. r-z23 v1 (XGBoost baseline): precision 0.68 but recall 0.07, misses 93% of fraud. Diagnosis: class imbalance plus 0.5 threshold. v2: fraud class weighted, shallower trees, threshold tuned on validation F1: recall 0.61, F1 0.50, PR-AUC 0.455. abacked v1/v2: **TBD** (#29, #30) |
| 6 | **Live: MLflow** | 1:30 | Switch to the MLflow UI. Experiment `amlop-ca1-fraud`, compare runs by `valid_pr_auc`, show the top two models in the registry and the champion the team chose (#34, #35) |
| 7 | RDS design | 1:00 | Tables `customers`, `orders`, `labels` keyed on country + id; labels link orders to customers. `ingestion_batches`, `batch_sources`, `rejected_rows` record every batch and every rejected row. RDS is private, reached through an SSH bastion |
| 8 | MLflow service and robustness | 1:00 | Docker on Elastic Beanstalk. Original: SQLite and local folders in the container, lost on redeploy. Design from the GenAI conversation (#41): records in RDS database `mlflow`, model files in S3, Beanstalk keeps nothing. **TBD**: applied (#42) and restart test result (#43) |
| 9 | Automated pipeline | 1:00 | URLs in, clean, load to RDS, retrain every model on old + new data, runs named `<batch>-<contributor>-<version>`, compare and promote. Configuration from environment variables only. Timing: **TBD** (#37) |
| 10 | Contributions | 0:30 | Who owned what, linked to issues and PRs. Percentages agreed by both members: **TBD** |

Total 9:30, leaving 30 s of buffer.

## Who presents (proposal, to agree)

| Slides | Presenter |
| --- | --- |
| 1 to 6 | r-z23 (EDA, cleaning, models, comparison) |
| 7 to 9 | abacked (RDS, MLflow service, ETL) |
| 10 | Both |

Each member must be able to answer questions on every slide.

## Live demo after the presentation

Each pipeline must finish within 3 minutes. Do not open the tutor's URLs before the slot.

1. Before the slot: SSH tunnel running, `.env` filled, MLflow UI open in a browser tab, Prefect server running on each laptop.
2. Paste the three tutor URLs into the ETL command (#36). Show the row counts and rejected rows.
3. In MySQL, show the new batch in `ingestion_batches` and the new rows.
4. Refresh MLflow and filter by the new batch tag; show the new runs, then run `compare_models --batch <batch>` and discuss whether the champion should change.
5. Each member runs their own Prefect flow and shows the query results and charts.

## Likely questions

| Question | Short answer |
| --- | --- |
| Why not use the customer fields? | The snapshot has no as-of time, so we can't prove the values were known before the order shipped. Using them risks leakage |
| Why is `refund_value` excluded? | Refunds are usually recorded after the order and their timing isn't verified; using them could predict fraud from its consequence |
| Why split by date, not randomly? | The model scores future orders, so it is tested on later orders than it was trained on |
| Why PR-AUC, not accuracy? | With 11% fraud, predicting "not fraud" for everything is 89% accurate. PR-AUC measures how well fraud is ranked above non-fraud |
| What does v2 cost the analysts? | Precision 0.43: about 4 in 10 flagged orders are fraud, versus 7 in 10 for v1, but v2 catches 61% of fraud instead of 7% |
| What happens if the same batch is loaded twice? | Planned in #24: the batch is identified by the hash of its three files, and a batch that already succeeded is skipped. **TBD**: confirm once built |
| What if MLflow is redeployed? | Once #42 is applied, records are in RDS and files in S3, so nothing is lost. If `BACKEND_STORE_URI` is missing it falls back to SQLite, which the smoke test catches |
