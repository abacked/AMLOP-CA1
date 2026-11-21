"""Opt-in MySQL contract tests, restricted to a disposable loopback database.

See docs/database.md. With no MYSQL_TEST_DATABASE these tests skip explicitly;
they never use DB_* or DATABASE_URL, so they cannot contact RDS accidentally.
"""

import os
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amlop_ca1 import cleaning

ROOT = Path(__file__).resolve().parents[1]
BATCH = "00000000-0000-0000-0000-000000000001"


@pytest.fixture(scope="module")
def database():
    name = os.environ.get("MYSQL_TEST_DATABASE")
    if not name:
        pytest.skip("Set MYSQL_TEST_DATABASE for disposable local MySQL validation")
    if not name.startswith("test_amlop_") or not name.replace("_", "").isalnum():
        pytest.fail("MYSQL_TEST_DATABASE must be named test_amlop_* using letters/digits/underscores")
    host = os.environ.get("MYSQL_TEST_HOST", "127.0.0.1")
    if host not in ("127.0.0.1", "localhost", "::1"):
        pytest.fail("Schema tests only connect to a disposable loopback server")
    import pymysql
    from pymysql.constants import CLIENT

    connection = pymysql.connect(
        host=host, port=int(os.environ.get("MYSQL_TEST_PORT", "3306")),
        user=os.environ["MYSQL_TEST_USER"], password=os.environ.get("MYSQL_TEST_PASSWORD", ""),
        database=name, charset="utf8mb4", autocommit=False,
        client_flag=CLIENT.MULTI_STATEMENTS,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT @@sql_mode")
            assert "STRICT" in cursor.fetchone()[0], "Strict SQL mode is required"
            cursor.execute((ROOT / "infra/schema.sql").read_text(encoding="utf-8"))
            while cursor.nextset():
                pass
        yield connection
    finally:
        connection.rollback()
        connection.close()


@pytest.fixture
def db(database):
    with database.cursor() as cursor:
        cursor.execute(
            "INSERT INTO ingestion_batches (batch_id, manifest_sha256, batch_suffix) VALUES (%s,%s,%s)",
            (BATCH, "a" * 64, "synthetic"),
        )
    try:
        yield database
    finally:
        database.rollback()


def parents(cursor, country="MY"):
    cursor.execute("INSERT INTO customers (country_code, customer_id, batch_id) VALUES (%s,'0000000001',%s)", (country, BATCH))
    cursor.execute("INSERT INTO orders (country_code, order_id, order_value, num_items_ordered, batch_id) VALUES (%s,'0000000002',10,0,%s)", (country, BATCH))


def test_bootstrap_is_repeatable(database):
    import warnings
    import pymysql

    with database.cursor() as cursor:
        cursor.execute("SHOW CREATE TABLE orders")
        before = cursor.fetchone()[1]
        # Expected IF NOT EXISTS notices are not failed constraints.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*already exists.*", category=pymysql.Warning)
            cursor.execute((ROOT / "infra/schema.sql").read_text(encoding="utf-8"))
            while cursor.nextset():
                pass
        cursor.execute("SHOW CREATE TABLE orders")
        assert cursor.fetchone()[1] == before


def test_composite_keys_allow_same_ids_in_different_countries(db):
    with db.cursor() as cursor:
        for country in ("MY", "PH"):
            parents(cursor, country)
            cursor.execute("INSERT INTO labels VALUES (%s,'0000000002','0000000001',1,%s)", (country, BATCH))
        cursor.execute("SELECT country_code, order_id FROM labels ORDER BY country_code")
        assert cursor.fetchall() == (("MY", "0000000002"), ("PH", "0000000002"))


@pytest.mark.parametrize("statement, code", [
    ("INSERT INTO orders (country_code,order_id,order_value,num_items_ordered,batch_id) VALUES ('MY','0000000002',10,1,%s)", 1062),
    ("INSERT INTO labels VALUES ('PH','0000000002','0000000001',1,%s)", 1452),
    ("INSERT INTO labels VALUES ('MY','0000000002','missing',1,%s)", 1452),
    ("INSERT INTO labels VALUES ('MY','0000000002','0000000001',2,%s)", 3819),
    ("INSERT INTO orders (country_code,order_id,order_value,num_items_ordered,batch_id) VALUES ('MY','bad',-1,1,%s)", 3819),
    ("INSERT INTO orders (country_code,order_id,order_value,num_items_ordered,batch_id) VALUES ('MY','bad',10,1.5,%s)", 3819),
    ("INSERT INTO customers (country_code,customer_id,num_orders_last_50days,batch_id) VALUES ('MY','bad',-1,%s)", 3819),
    ("INSERT INTO customers (country_code,customer_id,batch_id) VALUES ('my','bad',%s)", 3819),
])
def test_invalid_rows_fail_explicitly(db, statement, code):
    import pymysql

    with db.cursor() as cursor:
        parents(cursor)
        with pytest.raises(pymysql.MySQLError) as error:
            cursor.execute(statement, (BATCH,))
        assert error.value.args[0] == code


def test_zero_items_missing_dates_and_large_refunds_are_preserved(db):
    with db.cursor() as cursor:
        parents(cursor)
        cursor.execute("UPDATE orders SET refund_value=999,order_date_invalid=1")
        cursor.execute("SELECT num_items_ordered,refund_value,order_date,order_date_invalid FROM orders")
        assert cursor.fetchone() == (0, 999, None, 1)


def test_batch_identity_counts_and_reject_lineage(db):
    import pymysql

    with db.cursor() as cursor:
        with pytest.raises(pymysql.IntegrityError) as duplicate:
            cursor.execute("INSERT INTO ingestion_batches (batch_id,manifest_sha256,batch_suffix) VALUES (%s,%s,'other')", ("00000000-0000-0000-0000-000000000002", "a" * 64))
        assert duplicate.value.args[0] == 1062
        cursor.execute("INSERT INTO batch_sources VALUES (%s,'orders','orders_synthetic.csv',%s,3,2,1)", (BATCH, "b" * 64))
        with pytest.raises(pymysql.MySQLError) as counts:
            cursor.execute("UPDATE batch_sources SET accepted_rows=3")
        assert counts.value.args[0] == 3819
        cursor.execute("INSERT INTO rejected_rows (batch_id,source_table,source_row,reason) VALUES (%s,'orders',2,'invalid numeric value')", (BATCH,))
        with pytest.raises(pymysql.IntegrityError) as orphan:
            cursor.execute("INSERT INTO rejected_rows (batch_id,source_table,source_row,reason) VALUES (%s,'labels',2,'orphan')", (BATCH,))
        assert orphan.value.args[0] == 1452


def test_indexes_support_customer_date_and_batch_lookups(db):
    with db.cursor() as cursor:
        for table, expected in (("labels", "ix_labels_customer"), ("orders", "ix_orders_date"), ("customers", "ix_customers_batch")):
            cursor.execute(f"SHOW INDEX FROM {table}")
            assert expected in {row[2] for row in cursor.fetchall()}


def test_cleaner_output_maps_to_sql_without_losing_nulls_or_keys(db):
    raw = pd.DataFrame({
        "country_code": ["my", "PH"], "order_id": ["0000000002"] * 2,
        "collect_type": ["SELF-PICK-UP", None], "payment_method": [None, "Credit_Card"],
        "order_value": [10, 20], "num_items_ordered": [0, 2],
        "refund_value": [999, None], "order_date": ["2023/Jan/02", "bad"],
    })
    cleaned = cleaning.clean_orders(raw)
    columns = list(cleaned.columns) + ["batch_id"]
    records = cleaned.astype(object).where(cleaned.notna(), None).to_dict("records")
    with db.cursor() as cursor:
        for row in records:
            row["batch_id"] = BATCH
            if row["order_date"] is not None:
                row["order_date"] = row["order_date"].date()
            cursor.execute(
                "INSERT INTO orders (" + ",".join(columns) + ") VALUES (" + ",".join(["%s"] * len(columns)) + ")",
                tuple(row[column] for column in columns),
            )
        cursor.execute("SELECT country_code,order_id,collect_type,payment_method,refund_value,order_date_invalid FROM orders ORDER BY country_code")
        assert cursor.fetchall() == (
            ("MY", "0000000002", "pickup", "unknown", 999, 0),
            ("PH", "0000000002", None, "credit card", None, 1),
        )
