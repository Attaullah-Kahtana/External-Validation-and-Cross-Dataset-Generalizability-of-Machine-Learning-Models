import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)


def calculate_metrics(y_true, probability, threshold=0.50):

    y_true = np.asarray(y_true)
    probability = np.asarray(probability)

    prediction = (
        probability >= threshold
    ).astype(int)

    sensitivity = recall_score(
        y_true,
        prediction,
        zero_division=0
    )

    specificity = recall_score(
        1 - y_true,
        1 - prediction,
        zero_division=0
    )

    return {

        "roc_auc":
            roc_auc_score(
                y_true,
                probability
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                probability
            ),

        "brier_score":
            brier_score_loss(
                y_true,
                probability
            ),

        "accuracy":
            accuracy_score(
                y_true,
                prediction
            ),

        "precision":
            precision_score(
                y_true,
                prediction,
                zero_division=0
            ),

        "sensitivity":
            sensitivity,

        "specificity":
            specificity,

        "f1":
            f1_score(
                y_true,
                prediction,
                zero_division=0
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                prediction
            )
    }


def bootstrap_ci(
    y_true,
    probability,
    metric_function,
    n_bootstrap=2000,
    seed=42,
    confidence=0.95
):

    rng = np.random.default_rng(seed)

    y_true = np.asarray(y_true)
    probability = np.asarray(probability)

    values = []

    n = len(y_true)

    for _ in range(n_bootstrap):

        indices = rng.integers(
            0,
            n,
            size=n
        )

        y_sample = y_true[indices]
        p_sample = probability[indices]

        if len(np.unique(y_sample)) < 2:
            continue

        values.append(
            metric_function(
                y_sample,
                p_sample
            )
        )

    alpha = 1 - confidence

    return (
        np.quantile(
            values,
            alpha / 2
        ),

        np.quantile(
            values,
            1 - alpha / 2
        )
    )