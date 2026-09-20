
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


def bootstrap_auc(
    y_true,
    probability,
    n_resamples=1000,
    seed=42
):

    rng = np.random.default_rng(seed)

    y_true = np.asarray(y_true)
    probability = np.asarray(probability)

    scores = []

    n = len(y_true)

    for _ in range(n_resamples):

        indices = rng.integers(
            0,
            n,
            n
        )

        y_sample = y_true[indices]
        p_sample = probability[indices]

        if len(np.unique(y_sample)) < 2:
            continue

        scores.append(
            roc_auc_score(
                y_sample,
                p_sample
            )
        )

    return np.array(scores)


def bootstrap_delta_auc(
    y_a,
    p_a,
    y_b,
    p_b,
    n_resamples=1000,
    seed=42
):

    auc_a = roc_auc_score(
        y_a,
        p_a
    )

    auc_b = roc_auc_score(
        y_b,
        p_b
    )

    observed_delta = auc_a - auc_b

    rng = np.random.default_rng(seed)

    deltas = []

    auc_boot_a = bootstrap_auc(
        y_a,
        p_a,
        n_resamples,
        seed
    )

    auc_boot_b = bootstrap_auc(
        y_b,
        p_b,
        n_resamples,
        seed + 1
    )

    n = min(
        len(auc_boot_a),
        len(auc_boot_b)
    )

    deltas = (
        auc_boot_a[:n] -
        auc_boot_b[:n]
    )

    lower = np.percentile(
        deltas,
        2.5
    )

    upper = np.percentile(
        deltas,
        97.5
    )

    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "delta_auc": observed_delta,
        "ci_lower": lower,
        "ci_upper": upper,
    }


def build_delta_auc_table(
    internal_predictions,
    external_predictions
):

    rows = []

    models = sorted(
        set(internal_predictions["model"])
        &
        set(external_predictions["model"])
    )

    for model in models:

        internal = internal_predictions[
            internal_predictions["model"] == model
        ]

        external = external_predictions[
            external_predictions["model"] == model
        ]

        result = bootstrap_delta_auc(
            internal["y_true"],
            internal["probability"],
            external["y_true"],
            external["probability"],
        )

        result["model"] = model

        rows.append(result)

    return pd.DataFrame(rows)