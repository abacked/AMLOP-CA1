# Base dataset

The supplied CA1 data is in this directory:

- `customers_base.csv`: customer information and recent activity
- `orders_base.csv`: order details
- `labels_base.csv`: customer-order links and fraud labels

These files are tracked with Git LFS because the combined dataset is large. Run `git lfs install` before pulling the repository. The files are assignment inputs; do not edit them in place. Keep new tutor-provided batches in this directory locally under their supplied filenames, and let the ETL process read those files or their URLs. New batch CSVs are ignored by Git unless the team explicitly decides to commit them.
