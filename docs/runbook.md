# Deployment and troubleshooting runbook (#58)

How to set up, run and check the team services. For IT managers and both team members. Sections marked **TBD** are filled in once the linked task is deployed; until then they describe the agreed design, not a verified setup.

## Services

| Service | Where | Owner task |
| --- | --- | --- |
| RDS MySQL `amlop-team7-db`, database `fraud_detection` | us-east-1, private, reached through an SSH bastion | #22, #23 |
| MLflow tracking server and model registry | Docker on Elastic Beanstalk; records in RDS database `mlflow`, model files in S3 | #26, #42 |
| ETL and retraining | Python scripts run from a team laptop | #24, #36 |
| Individual analytics | Each student's Prefect flow on their own laptop | #47, #52 |

## Configuration

All resource settings come from environment variables; nothing is hard-coded or committed. Copy `.env.example` to `.env` (ignored by Git) and fill it in privately.

| Variable | Used by | Example |
| --- | --- | --- |
| `DB_HOST`, `DB_PORT` | ETL, analytics | `127.0.0.1`, `3307` when using the tunnel |
| `DB_NAME`, `DB_USER`, `DB_PASSWORD` | ETL, analytics | `fraud_detection`; the analytics user is read-only |
| `DB_SSL_CA` | ETL, analytics | Path to the AWS RDS CA bundle |
| `MLFLOW_TRACKING_URI` | Training, comparison, smoke test | The Elastic Beanstalk URL |
| `MLFLOW_EXPERIMENT_NAME` | Training, comparison | `amlop-ca1-fraud` |

Through the SSH tunnel, verify the certificate with `VERIFY_CA`, not `VERIFY_IDENTITY`: the client connects to `127.0.0.1`, which does not match the RDS hostname on the certificate.

## RDS

### Connect through the bastion

RDS has no public access. Open a tunnel in its own terminal and leave it running:

```bash
ssh -i ~/.ssh/amlop_bastion -N -L 3307:<rds-endpoint>:3306 <bastion-user>@<bastion-address>
```

Then connect to `127.0.0.1:3307`. Health check in the MySQL client:

```sql
SELECT VERSION(), DATABASE(), CURRENT_USER();
SHOW SESSION STATUS LIKE 'Ssl_cipher';
SELECT COUNT(*) FROM orders;
```

### Create the tables

Run `infra/schema.sql` once against `fraud_detection` (safe to repeat; it does not migrate existing tables). Full setup and table design: [database.md](database.md).

## MLflow on Elastic Beanstalk (TBD: #26, #42)

1. Build the source bundle zip from `infra/mlflow/` (Dockerfile plus config). Keep the zip in the repo for the submission.
2. Create the Elastic Beanstalk environment in us-east-1 with service role `LabRole` and instance profile `LabInstanceProfile`.
3. Set environment properties, never in the image:
   - `BACKEND_STORE_URI`: `mysql+pymysql://<user>:<password>@<rds-endpoint>:3306/mlflow`
   - `ARTIFACTS_DESTINATION`: `s3://<bucket>/mlflow-artifacts`
4. The `mlflow` database must be created with `COLLATE utf8mb4_bin` so names are case-sensitive.
5. Run the smoke test from a laptop (below).

### Health check

```bash
MLFLOW_TRACKING_URI=<eb-url> PYTHONPATH=src python -m amlop_ca1.mlflow_smoke
```

It checks the server is reachable, logs a run, confirms the artifacts are stored on the server, reloads the model, registers it and loads it through the registry, then cleans up. To check persistence, run it with `--keep`, restart or redeploy the environment, then run it with `--check-kept`.

## New data batch (TBD: #36)

When the tutor gives the three URLs:

1. Run the ETL entry point (#36) with the customers, orders and labels URLs. It cleans and loads the batch into RDS, then retrains every team model on old plus new data.
2. Each run is named `<batch>-<contributor>-<version>` and tagged with the batch, so one ETL run's results can be filtered in MLflow.
3. Compare the new runs:

```bash
PYTHONPATH=src python -m amlop_ca1.compare_models --batch <batch> --out comparison.csv
```

This ranks the runs by validation PR-AUC (F1 breaks ties). The ranking is a recommendation: the team agrees on the top two and the champion at the review (#35), then registers them and sets the champion in MLflow (#34).

To retrain one model by hand from local files:

```bash
PYTHONPATH=src python -m amlop_ca1.train_rz23 --data-dir data --batch base --version v2
```

## Where to find outputs

| Output | Location |
| --- | --- |
| Loaded data, batch history, rejected rows | RDS tables `customers`, `orders`, `labels`, `ingestion_batches`, `batch_sources`, `rejected_rows` |
| Cleaning log | `logs/cleaning_<batch>_<time>.log` |
| Model runs and metrics | MLflow experiment `amlop-ca1-fraud` (set by `MLFLOW_EXPERIMENT_NAME`) |
| Top two models and champion | MLflow model registry (TBD: #34) |
| Individual analytics | Each student's Prefect UI (`http://127.0.0.1:4200`) and flow logs |

## Persistence limits and backups

- MLflow records live in RDS and model files in S3, so restarting or redeploying the Elastic Beanstalk environment does not lose them. If `BACKEND_STORE_URI` is missing, MLflow falls back to SQLite inside the container and new runs disappear on the next redeploy. After each deploy, run the smoke test and confirm the run appears.
- Everything is in one AWS Academy Learner Lab account. If the account is shut down (over budget or limits) or expires, RDS, S3 and Elastic Beanstalk are deleted together. Keep a local copy before the interview: `mysqldump` of `fraud_detection` and `mlflow`, and `aws s3 sync` of the artifact bucket.
- A stopped RDS instance restarts by itself after 7 days.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ssh` times out | Bastion stopped, Learner Lab session ended, or its public IP changed | Start the lab, start the instance, check the current address (an Elastic IP avoids this) |
| `ssh` says `Permission denied (publickey)` | Wrong key file or key not added on the bastion | Use the key whose public half is in the bastion's `authorized_keys` |
| MySQL `Can't connect ... 3307` | Tunnel not running | Start the `ssh -L` command in another terminal |
| SSL error about hostname mismatch | `VERIFY_IDENTITY` used through the tunnel | Use `VERIFY_CA` with the RDS CA bundle |
| `Access denied` on insert | Using the read-only analytics user for the ETL | Use the ETL user |
| Smoke test: `artifacts are not stored on the server` | Server not started with an artifact destination | Set `ARTIFACTS_DESTINATION` and redeploy |
| Runs disappear after a redeploy | MLflow fell back to SQLite in the container | Check `BACKEND_STORE_URI` in the environment properties |
| Elastic Beanstalk health is Severe | Container crashed or wrong port | Check the environment logs; the container must listen on the port Elastic Beanstalk expects |
| `TypeError: boolean value of NA is ambiguous` in training | Old `features.py` before beb225a with pyarrow installed | Update to main |
| Pipeline over 3 minutes | Slow RDS load or training | Training takes about 15 s per model on CPU on the base data; time the load separately and load the base batch before the demo |
