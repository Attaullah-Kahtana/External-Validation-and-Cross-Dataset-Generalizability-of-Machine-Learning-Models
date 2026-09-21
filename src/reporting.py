
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)


def plot_roc_curves(
    predictions_dir,
    output_file,
    title,
    pattern="*_test_predictions.csv",
):
    """Plot ROC curves from saved prediction CSV files."""
    predictions_dir = Path(
        predictions_dir
    )

    output_file = Path(
        output_file
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_files = sorted(
        predictions_dir.glob(pattern)
    )

    if not prediction_files:
        raise FileNotFoundError(
            f"No prediction files found in "
            f"{predictions_dir} matching '{pattern}'."
        )

    plt.figure(
        figsize=(8, 6)
    )

    for file in prediction_files:
        df = pd.read_csv(file)

        fpr, tpr, _ = roc_curve(
            df["y_true"],
            df["probability"],
        )

        auc = roc_auc_score(
            df["y_true"],
            df["probability"],
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{file.stem} AUC={auc:.3f}",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(title)

    plt.legend(
        fontsize=8
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
    )

    plt.close()
