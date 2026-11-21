# MySQL schema and setup — #22 / #23

## Setup progress

On 22 November 2026, Joshua confirmed provisioning and a successful Workbench connection:

| Setting | Value |
| --- | --- |
| RDS instance / region | `amlop-team7-db` / `us-east-1` |
| Engine / class | MySQL 8.4.11 / `db.t3.small` |
| Storage | 20 GiB gp3 |
| Database / port | `fraud_detection` / 3306 |
| Access | Security-group inbound rules restricted to intended clients |

This records the reported setup, not an independent AWS inspection. Python connectivity from the intended ETL environment, its TLS verification and schema deployment to RDS are still pending. #22 stays open. No endpoint or credentials belong in this document.

## Verified input and cleaning contract

The Git LFS base files contain actual CSV data: customers 901,312 rows / 10 columns; orders 949,889 / 8; labels 950,318 / 4. Source IDs are ten-character strings. Each raw file has no duplicate country-plus-ID key. The merged cleaning module in main, from PR #13, defines these relationships:

| Table | Primary key | References / indexes |
| --- | --- | --- |
| `customers` | `(country_code, customer_id)` | First accepted `batch_id`; batch index |
| `orders` | `(country_code, order_id)` | First accepted `batch_id`; date/country and batch indexes |
| `labels` | `(country_code, order_id)` | Composite order and customer foreign keys; customer and batch indexes |
| `ingestion_batches` | `batch_id` | Unique content-manifest hash; status/creation index |
| `batch_sources` | `(batch_id, source_table)` | Three source files, hashes, sanitized locations and row outcomes |
| `rejected_rows` | Auto-generated `reject_id` | Source foreign key; batch/table/row lookup index |

`orders` has no `customer_id` in the supplied schema; do not invent a direct customer relationship or require labels for storing inference orders. Labels establish that relationship when available. Foreign keys prevent cross-country matches and orphan labels. Parent deletion is restricted rather than cascaded. No country enum prevents accepting a legitimate new country.

`infra/schema.sql` preserves all source fields plus cleaning's date-quality flags. Text IDs use case-sensitive ASCII strings, preserving zeros; 128 characters is an explicit future-batch limit, not a verified upstream guarantee. Category text uses Unicode. Validate lengths and characters before loading; reject oversized/unsupported inputs, never truncate. Invalid-key details in rejection logs are diagnostic and are not foreign keys to accepted records.

Amounts use `DOUBLE` to match the CSV/cleaner float representation without inventing a currency, exchange rate or decimal rounding policy. Count fields also accept the cleaner's float representation, with nonnegative integer checks. Optional customer/refund measurements remain nullable. Required order value/item count are non-null; zero items is valid. Refunds exceeding order value are deliberately permitted, matching cleaning. These analytical types are not a financial-ledger contract.

Missing/invalid dates remain SQL `NULL`, with flags distinguishing failed parsing from missing input. Dates use `DATE`; first-order timestamps use `DATETIME(6)` without inventing a source timezone. Convert pandas `NA`/`NaT`/`NaN` to Python `None` before parameterized inserts. Never write imputed training values to these source tables.

Customer history and refunds are retained for storage/retrospective analysis. A batch timestamp does not establish prediction-time availability: these fields must remain outside the feature allowlist until verified. Schema work does not change PR #14.

## Repeatable setup (Windows and Linux)

1. Use the MySQL 8.4 client and an account with schema-creation privileges on the existing `fraud_detection` database. Keep the application account limited to the ETL/query privileges it needs; it should not have global/admin privileges.
2. Copy `.env.example` to local `.env` and fill it privately. `.env` is ignored by Git. It is not loaded automatically: export the selected `DB_*` values in your shell or use an explicit dotenv/config loader in the future ETL. `DATABASE_URL` is an alternative for future SQLAlchemy callers, not a TLS configuration by itself. If using it, URL-encode username/password and configure verified TLS separately. Do not mix fallback credentials between styles.
3. Download the current regional/global RDS CA bundle from [AWS's connection guide](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_ConnectToInstanceSSL.CLI.html). Set `DB_SSL_CA` to its local absolute path. Keep both verification flags true; connect by the actual endpoint hostname. Network restriction and certificate verification serve different purposes.

PowerShell (from the repository root):

```powershell
if (-not (Test-Path -LiteralPath .env)) { Copy-Item .env.example .env }
mysql --host=$env:DB_HOST --port=$env:DB_PORT --user=$env:DB_USER --password --ssl-ca=$env:DB_SSL_CA --ssl-mode=VERIFY_IDENTITY --database=fraud_detection
```

Linux (from the repository root):

```bash
test -e .env || cp .env.example .env
chmod 600 .env
mysql --host="$DB_HOST" --port="$DB_PORT" --user="$DB_USER" --password --ssl-ca="$DB_SSL_CA" --ssl-mode=VERIFY_IDENTITY --database=fraud_detection
```

The client prompts for a password; never put it directly in a command or screenshot. In the connected client:

```sql
SELECT VERSION(), DATABASE(), CURRENT_USER();
SHOW SESSION STATUS LIKE 'Ssl_cipher';
SELECT @@SESSION.sql_mode;
SOURCE infra/schema.sql;
SOURCE infra/schema.sql;
SHOW TABLES;
SHOW CREATE TABLE customers;
SHOW CREATE TABLE orders;
SHOW CREATE TABLE labels;
SHOW CREATE TABLE ingestion_batches;
SHOW CREATE TABLE batch_sources;
SHOW CREATE TABLE rejected_rows;
```

Confirm a nonempty TLS cipher and strict SQL mode (`STRICT_TRANS_TABLES` or `STRICT_ALL_TABLES`, plus `NO_ZERO_DATE` and `NO_ZERO_IN_DATE`). Do not disable strict mode, constraints or foreign-key checks to force a load. In Workbench, open `infra/schema.sql`, explicitly select `fraud_detection`, execute it, then execute it again and inspect the definitions.

`CREATE TABLE IF NOT EXISTS` safely repeats a fresh bootstrap but does **not** detect or migrate incompatible existing tables. Compare `SHOW CREATE TABLE` against this version before loading. For a future change, add a numbered migration under `infra/migrations/`, record its version/checksum and review the upgrade against a backup. Never drop data or assume MySQL DDL rolls back; DDL implicitly commits. No migration runner is implemented in this task.

## Batch identity, counts and transactions

The ETL in #24/#25 must implement this protocol; SQL uniqueness/FKs alone do not implement orchestration:

1. Hash the raw bytes of all three CSV files. Compute `manifest_sha256` from UTF-8 bytes of exactly `customers:<hash>\norders:<hash>\nlabels:<hash>\n`, in that order, lowercase hex and a final newline. URLs and suffixes are metadata, not identities. Retain immutable raw inputs in controlled storage for replay.
2. Insert/claim one UUID `batch_id` under the unique manifest hash. Use a transaction and `SELECT ... FOR UPDATE` on an existing hash. A succeeded batch is a no-op; an already-running batch is not started again. Increment `attempt_count` and set the running start time atomically. Recover a stale running claim only after confirming the previous worker stopped.
3. Register exactly three `batch_sources` rows before reject logging. Store sanitized locations without credentials, query tokens or signed URLs. Counts can be null while running; on success, each file must have `input_rows = accepted_rows + rejected_rows`. The ETL must check all three summaries before success because a SQL row check cannot enforce that cross-table condition.
4. Load using `load_batch` and the merged cleaner; preserve original zero-based row positions. Count each rejected source row once, even if it has multiple reason records. Exact duplicates removed by cleaning count as rejected/duplicate outcomes, not accepted rows. Keep every reason in `rejected_rows`. Existing identical database rows count as accepted no-op outcomes, not new inserts.
5. Compare existing canonical business records without the lineage `batch_id`. Identical records are no-ops; differing orders/labels are quarantined. For now, differing customer snapshots are also quarantined rather than overwritten: legitimate activity changes need an agreed, timestamped versioning policy. This conservative policy may reduce future-batch coverage and needs team review. Current tables are not customer history snapshots. Keep per-batch source evidence even for no-ops; the business row's batch ID remains its first accepted batch.
6. In one data transaction, insert customers/orders before labels, then write reject records, final per-source counts and the succeeded state with completion time. On failure roll back data and summaries, and persist a sanitized failure status separately. Retry a failed hash using the same batch ID after clearing prior attempt summaries/reasons, with no silent `REPLACE`, `INSERT IGNORE` or last-write-wins. Error text must not contain passwords, connection URLs or raw sensitive rows.

This protocol is designed here; loading, atomic claims, retries, state transitions and customer versioning are not implemented or verified by the schema tests. Labels may arrive in a later batch and reference existing parents; all three parent rows need not originate from the same batch.

## Local validation and remaining live checks

Validation on 22 November 2026: **51 tests passed** with warnings treated as errors, comprising 37 existing cleaning tests and 14 schema tests against an isolated local MySQL **8.0.44** server. The schema was applied twice, and cleaned synthetic orders were inserted and read back without losing keys or nullable values. This does not establish compatibility with every MySQL 8.4 setting or verify Python connectivity to RDS 8.4.11. No RDS data or configuration was changed.

Install `requirements.txt` in a virtual environment. `tests/test_schema.py` uses PyMySQL and runs only when `MYSQL_TEST_DATABASE` is set. It permits only a loopback server and a database named `test_amlop_*`, never the application's `DB_*` or `DATABASE_URL` settings. Create an empty disposable test database with a local MySQL administrator, then configure `MYSQL_TEST_HOST`, `MYSQL_TEST_PORT`, `MYSQL_TEST_USER` and `MYSQL_TEST_DATABASE`. Do not point these tests at RDS or production data.

```sql
CREATE DATABASE test_amlop_schema CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_bin;
```

PowerShell example (local server already running):

```powershell
$env:MYSQL_TEST_HOST = '127.0.0.1'
$env:MYSQL_TEST_PORT = '3306'
$env:MYSQL_TEST_USER = 'YOUR_LOCAL_TEST_USER'
$env:MYSQL_TEST_DATABASE = 'test_amlop_schema'
python -c "import getpass,os,pytest; os.environ['MYSQL_TEST_PASSWORD']=getpass.getpass('Local test password: '); raise SystemExit(pytest.main(['tests','-q','-W','error']))"
```

Linux uses the same Python command after exporting those four variables. Test data writes are rolled back; the disposable database and bootstrapped tables remain. The tests cover repeatability, duplicate keys, cross-country IDs, orphan labels, target/numeric constraints, counts, source lineage, lookup indexes and actual `clean_orders` output. Local MySQL validation is distinct from the intended Python-to-RDS verification in #22. Re-run the bootstrap/constraint checks against MySQL 8.4 before deployment approval; keep #23 open pending review and RDS verification.

MySQL's enforced [CHECK constraints](https://dev.mysql.com/doc/refman/8.4/en/create-table-check-constraints.html) and InnoDB foreign keys are required; SQLite is not a substitute for this validation.
