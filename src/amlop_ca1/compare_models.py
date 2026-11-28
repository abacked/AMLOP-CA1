"""Compare every team model trained on the same batch, as input for the team review.

Each training module must log one run per model version to the shared
experiment with:

- tags ``contributor``, ``version`` and ``batch``;
- validation metrics ``valid_precision``, ``valid_recall``, ``valid_f1``,
  ``valid_pr_auc`` and ``valid_roc_auc`` from the date split in train_rz23;
- the fitted pipeline logged under the artifact name ``model``.

Runs are ranked by validation PR-AUC, which does not depend on a threshold and
suits the rare fraud class; validation F1 breaks ties. Test metrics are shown
but never used for ranking. Only runs from the same batch are compared, so all
models were trained and scored on the same rows.

The ranking is a recommendation. The team picks the top two and the champion
at the scrum review (#35) and registers them in MLflow (#34).

    PYTHONPATH=src python -m amlop_ca1.compare_models --batch base --out comparison.csv
"""

import argparse
import os

import mlflow


EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT_NAME", "amlop-ca1-fraud")
METRICS = ["precision", "recall", "f1", "pr_auc", "roc_auc"]


def load_runs(batch, experiment=EXPERIMENT):
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=f"tags.batch = '{batch}' and attributes.status = 'FINISHED'",
    )
    if runs.empty:
        raise RuntimeError(f"no finished runs for batch '{batch}' in '{experiment}'")
    return runs


def compare(runs):
    table = runs[["run_id", "tags.mlflow.runName", "tags.contributor", "tags.version"]].copy()
    table.columns = ["run_id", "run_name", "contributor", "version"]
    for split in ["valid", "test"]:
        for metric in METRICS:
            column = f"metrics.{split}_{metric}"
            table[f"{split}_{metric}"] = runs[column] if column in runs else None
    # A run without validation metrics cannot be ranked fairly.
    table = table.dropna(subset=["valid_pr_auc", "valid_f1"])
    table = table.sort_values(["valid_pr_auc", "valid_f1"], ascending=False)
    return table.reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="base")
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--out", help="optional CSV path for the comparison table")
    args = parser.parse_args()

    table = compare(load_runs(args.batch, args.experiment))
    columns = ["run_name", "valid_pr_auc", "valid_precision", "valid_recall", "valid_f1", "test_pr_auc"]
    print(table[columns].round(4).to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
