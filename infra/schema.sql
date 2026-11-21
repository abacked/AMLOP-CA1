-- Bootstrap v1 for MySQL 8.4 (also compatible with 8.0.16+).
-- Select the application database first. Never run this file in mysql/sys.
-- IF NOT EXISTS is repeatable bootstrap, not a migration of existing tables.
-- Use strict SQL mode; DDL auto-commits in MySQL.

CREATE TABLE IF NOT EXISTS ingestion_batches (
    batch_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin PRIMARY KEY,
    manifest_sha256 CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    batch_suffix VARCHAR(128) NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'pending',
    attempt_count INT UNSIGNED NOT NULL DEFAULT 0,
    started_at DATETIME(6) NULL,
    completed_at DATETIME(6) NULL,
    error_summary VARCHAR(512) NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    UNIQUE KEY uq_batches_manifest (manifest_sha256),
    INDEX ix_batches_status_created (status, created_at),
    CONSTRAINT ck_batch_hash CHECK (REGEXP_LIKE(manifest_sha256, '^[0-9a-f]{64}$', 'c')),
    CONSTRAINT ck_batch_suffix CHECK (CHAR_LENGTH(TRIM(batch_suffix)) > 0),
    CONSTRAINT ck_batch_status CHECK (status IN ('pending', 'running', 'succeeded', 'failed')),
    CONSTRAINT ck_batch_times CHECK (completed_at IS NULL OR
        (started_at IS NOT NULL AND completed_at >= started_at)),
    CONSTRAINT ck_batch_terminal CHECK (
        (status IN ('pending', 'running') AND completed_at IS NULL) OR
        (status IN ('succeeded', 'failed') AND completed_at IS NOT NULL))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin;

CREATE TABLE IF NOT EXISTS batch_sources (
    batch_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    source_table VARCHAR(16) NOT NULL,
    source_uri TEXT NOT NULL COMMENT 'Sanitized location, never signed URLs or credentials',
    content_sha256 CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    input_rows BIGINT UNSIGNED NULL,
    accepted_rows BIGINT UNSIGNED NULL,
    rejected_rows BIGINT UNSIGNED NULL,
    PRIMARY KEY (batch_id, source_table),
    CONSTRAINT fk_sources_batch FOREIGN KEY (batch_id) REFERENCES ingestion_batches(batch_id),
    CONSTRAINT ck_source_table CHECK (source_table IN ('customers', 'orders', 'labels')),
    CONSTRAINT ck_source_hash CHECK (REGEXP_LIKE(content_sha256, '^[0-9a-f]{64}$', 'c')),
    CONSTRAINT ck_source_counts CHECK (
        (input_rows IS NULL AND accepted_rows IS NULL AND rejected_rows IS NULL) OR
        (input_rows IS NOT NULL AND accepted_rows IS NOT NULL AND rejected_rows IS NOT NULL
         AND input_rows = accepted_rows + rejected_rows))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin;

CREATE TABLE IF NOT EXISTS customers (
    country_code CHAR(2) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    customer_id VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    mobile_verified BOOLEAN NULL,
    num_orders_last_50days DOUBLE NULL,
    num_cancelled_orders_last_50days DOUBLE NULL,
    num_refund_orders_last_50days DOUBLE NULL,
    total_payment_last_50days DOUBLE NULL,
    num_associated_customers DOUBLE NULL,
    first_order_datetime DATETIME(6) NULL,
    first_order_missing BOOLEAN NOT NULL DEFAULT 0,
    first_order_invalid BOOLEAN NOT NULL DEFAULT 0,
    batch_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT 'First accepted batch',
    PRIMARY KEY (country_code, customer_id),
    INDEX ix_customers_batch (batch_id),
    CONSTRAINT fk_customers_batch FOREIGN KEY (batch_id) REFERENCES ingestion_batches(batch_id),
    CONSTRAINT ck_customer_country CHECK (REGEXP_LIKE(country_code, '^[A-Z]{2}$', 'c')),
    CONSTRAINT ck_customer_id CHECK (CHAR_LENGTH(TRIM(customer_id)) > 0 AND customer_id = TRIM(customer_id)),
    CONSTRAINT ck_mobile_verified CHECK (mobile_verified IN (0, 1)),
    CONSTRAINT ck_first_missing CHECK (first_order_missing IN (0, 1)),
    CONSTRAINT ck_first_invalid CHECK (first_order_invalid IN (0, 1)),
    CONSTRAINT ck_customer_orders CHECK (num_orders_last_50days >= 0 AND num_orders_last_50days = FLOOR(num_orders_last_50days)),
    CONSTRAINT ck_customer_cancelled CHECK (num_cancelled_orders_last_50days >= 0 AND num_cancelled_orders_last_50days = FLOOR(num_cancelled_orders_last_50days)),
    CONSTRAINT ck_customer_refunds CHECK (num_refund_orders_last_50days >= 0 AND num_refund_orders_last_50days = FLOOR(num_refund_orders_last_50days)),
    CONSTRAINT ck_customer_payment CHECK (total_payment_last_50days >= 0),
    CONSTRAINT ck_customer_associated CHECK (num_associated_customers >= 0 AND num_associated_customers = FLOOR(num_associated_customers))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin;

CREATE TABLE IF NOT EXISTS orders (
    country_code CHAR(2) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    order_id VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    collect_type VARCHAR(128) NULL,
    payment_method VARCHAR(128) NULL,
    order_value DOUBLE NOT NULL,
    num_items_ordered DOUBLE NOT NULL,
    refund_value DOUBLE NULL,
    order_date DATE NULL,
    order_date_invalid BOOLEAN NOT NULL DEFAULT 0,
    batch_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT 'First accepted batch',
    PRIMARY KEY (country_code, order_id),
    INDEX ix_orders_batch (batch_id),
    INDEX ix_orders_date (order_date, country_code),
    CONSTRAINT fk_orders_batch FOREIGN KEY (batch_id) REFERENCES ingestion_batches(batch_id),
    CONSTRAINT ck_order_country CHECK (REGEXP_LIKE(country_code, '^[A-Z]{2}$', 'c')),
    CONSTRAINT ck_order_id CHECK (CHAR_LENGTH(TRIM(order_id)) > 0 AND order_id = TRIM(order_id)),
    CONSTRAINT ck_order_value CHECK (order_value >= 0),
    CONSTRAINT ck_order_items CHECK (num_items_ordered >= 0 AND num_items_ordered = FLOOR(num_items_ordered)),
    CONSTRAINT ck_order_refund CHECK (refund_value >= 0),
    CONSTRAINT ck_order_date_flag CHECK (order_date_invalid IN (0, 1))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin;

-- Orders do not contain customer_id. Labels supply the customer relationship.
CREATE TABLE IF NOT EXISTS labels (
    country_code CHAR(2) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    order_id VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    customer_id VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    is_fraud TINYINT NOT NULL,
    batch_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT 'First accepted batch',
    PRIMARY KEY (country_code, order_id),
    INDEX ix_labels_customer (country_code, customer_id),
    INDEX ix_labels_batch (batch_id),
    CONSTRAINT fk_labels_order FOREIGN KEY (country_code, order_id) REFERENCES orders(country_code, order_id),
    CONSTRAINT fk_labels_customer FOREIGN KEY (country_code, customer_id) REFERENCES customers(country_code, customer_id),
    CONSTRAINT fk_labels_batch FOREIGN KEY (batch_id) REFERENCES ingestion_batches(batch_id),
    CONSTRAINT ck_label_binary CHECK (is_fraud IN (0, 1))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin;

CREATE TABLE IF NOT EXISTS rejected_rows (
    reject_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    batch_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    source_table VARCHAR(16) NOT NULL,
    source_row BIGINT UNSIGNED NOT NULL COMMENT 'Zero-based input DataFrame position',
    reason VARCHAR(512) NOT NULL,
    country_code VARCHAR(128) NULL,
    order_id VARCHAR(128) NULL,
    customer_id VARCHAR(128) NULL,
    INDEX ix_rejects_source (batch_id, source_table, source_row),
    CONSTRAINT fk_rejects_source FOREIGN KEY (batch_id, source_table) REFERENCES batch_sources(batch_id, source_table),
    CONSTRAINT ck_reject_reason CHECK (CHAR_LENGTH(TRIM(reason)) > 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin;
