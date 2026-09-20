

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp, chi2_contingency


CONTINUOUS = [
    "age",
    "bmi",
]

CATEGORICAL = [
    "gender",
    "smoking",
    "alcohol",
    "physical_activity",
    "high_bp",
    "high_chol",
    "high_gluc",
]


def calculate_psi(
    expected,
    actual,
    bins=10
):

    expected = np.asarray(expected)
    actual = np.asarray(actual)

    quantiles = np.linspace(
        0,
        1,
        bins + 1
    )

    edges = np.quantile(
        expected,
        quantiles
    )

    edges = np.unique(edges)

    if len(edges) < 3:
        return np.nan

    edges[0] = -np.inf
    edges[-1] = np.inf

    expected_counts, _ = np.histogram(
        expected,
        bins=edges
    )

    actual_counts, _ = np.histogram(
        actual,
        bins=edges
    )

    expected_pct = (
        expected_counts /
        max(expected_counts.sum(), 1)
    )

    actual_pct = (
        actual_counts /
        max(actual_counts.sum(), 1)
    )

    expected_pct = np.clip(
        expected_pct,
        1e-6,
        None
    )

    actual_pct = np.clip(
        actual_pct,
        1e-6,
        None
    )

    return np.sum(
        (actual_pct - expected_pct)
        *
        np.log(
            actual_pct /
            expected_pct
        )
    )


def continuous_shift(
    expected,
    actual,
    column
):

    e = pd.to_numeric(
        expected[column],
        errors="coerce"
    ).dropna()

    a = pd.to_numeric(
        actual[column],
        errors="coerce"
    ).dropna()

    ks_stat, ks_p = ks_2samp(
        e,
        a
    )

    psi = calculate_psi(
        e,
        a
    )

    return {
        "variable": column,
        "type": "continuous",
        "statistic": ks_stat,
        "p_value": ks_p,
        "psi": psi,
    }


def categorical_shift(
    expected,
    actual,
    column
):

    e = expected[column].value_counts()
    a = actual[column].value_counts()

    categories = sorted(
        set(e.index) |
        set(a.index)
    )

    observed = np.array([
        [
            e.get(category, 0)
            for category in categories
        ],
        [
            a.get(category, 0)
            for category in categories
        ],
    ])

    chi2, p, _, _ = chi2_contingency(
        observed
    )

    return {
        "variable": column,
        "type": "categorical",
        "statistic": chi2,
        "p_value": p,
        "psi": calculate_psi(
            expected[column],
            actual[column]
        ),
    }


def compare_distributions(
    dataset_a,
    dataset_b
):

    rows = []

    for column in CONTINUOUS:

        rows.append(
            continuous_shift(
                dataset_a,
                dataset_b,
                column
            )
        )

    for column in CATEGORICAL:

        rows.append(
            categorical_shift(
                dataset_a,
                dataset_b,
                column
            )
        )

    return pd.DataFrame(rows)