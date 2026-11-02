"""Compute the published F1, precision, and recall from results/.

The label boundaries are those of the paper: rows 0 to 4000 are normal in the
10,801-sample runs, and the first 100 rows are normal in the 200-sample runs.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def evaluate(results, normal_rows):
    y_pred = pd.read_csv(results)["anomaly"].astype(int).to_numpy()
    y_true = np.concatenate((np.zeros(normal_rows, dtype=int), np.ones(len(y_pred) - normal_rows, dtype=int)))
    return (
        f1_score(y_true, y_pred, zero_division=0),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
    )


def report(pattern, normal_rows):
    rows = {}
    for file in sorted(RESULTS_DIR.glob(pattern)):
        model = file.stem.split("_")[5]
        rows[model] = evaluate(file, normal_rows)
    return pd.DataFrame.from_dict(rows, orient="index", columns=["F1", "Precision", "Recall"]).round(3)


if __name__ == "__main__":
    print("200 test samples (3900 to 4099)")
    print(report("results_from_3900_to_4100_*.csv", 100))
    print("\nAll 10,801 test samples")
    print(report("results_from_0_to_10801_*.csv", 4001))
