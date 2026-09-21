import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency, ks_2samp


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


def calculate_psi(expected, actual, bins=10):
    """Calculate PSI for a continuous/multi-level numeric variable.

    Quantile bins are learned from the expected/reference dataset.
    For binary variables, use binary_proportion_shift instead.
    """
    expected = pd.to_numeric(
        pd.Series(expected),
        errors="coerce",
    ).dropna().to_numpy()

    actual = pd.to_numeric(
        pd.Series(actual),
        errors="coerce",
    ).dropna().to_numpy()

    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    quantiles = np.linspace(
        0,
        1,
        bins + 1,
    )

    edges = np.quantile(
        expected,
        quantiles,
    )

    edges = np.unique(edges)

    if len(edges) < 3:
        return np.nan

    edges[0] = -np.inf
    edges[-1] = np.inf

    expected_counts, _ = np.histogram(
        expected,
        bins=edges,
    )

    actual_counts, _ = np.histogram(
        actual,
        bins=edges,
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
        None,
    )

    actual_pct = np.clip(
        actual_pct,
        1e-6,
        None,
    )

    return float(
        np.sum(
            (actual_pct - expected_pct)
            * np.log(
                actual_pct / expected_pct
            )
        )
    )


def binary_proportion_shift(expected, actual, column):
    """Absolute difference in the proportion coded as 1.

    0 = identical proportions; larger values = greater shift.
    This is reported separately from continuous-variable PSI because
    ordinary quantile-binned PSI is not appropriate for binary variables.
    """
    e = pd.to_numeric(
        expected[column],
        errors="coerce",
    ).dropna()

    a = pd.to_numeric(
        actual[column],
        errors="coerce",
    ).dropna()

    if len(e) == 0 or len(a) == 0:
        return np.nan

    return float(abs(a.mean() - e.mean()))


def continuous_shift(expected, actual, column):
    e = pd.to_numeric(
        expected[column],
        errors="coerce",
    ).dropna()

    a = pd.to_numeric(
        actual[column],
        errors="coerce",
    ).dropna()

    if len(e) == 0 or len(a) == 0:
        return {
            "variable": column,
            "type": "continuous",
            "statistic": np.nan,
            "p_value": np.nan,
            "psi": np.nan,
        }

    ks_stat, ks_p = ks_2samp(
        e,
        a,
    )

    psi = calculate_psi(
        e,
        a,
    )

    return {
        "variable": column,
        "type": "continuous",
        "statistic": ks_stat,
        "p_value": ks_p,
        "psi": psi,
    }


def categorical_shift(expected, actual, column):
    e = pd.to_numeric(
        expected[column],
        errors="coerce",
    ).dropna()

    a = pd.to_numeric(
        actual[column],
        errors="coerce",
    ).dropna()

    categories = sorted(
        set(e.unique()) |
        set(a.unique())
    )

    observed = np.array([
        [
            (e == category).sum()
            for category in categories
        ],
        [
            (a == category).sum()
            for category in categories
        ],
    ])

    # Chi-square requires a non-degenerate contingency table.
    if observed.shape[1] < 2:
        chi2 = np.nan
        p = np.nan
    else:
        chi2, p, _, _ = chi2_contingency(
            observed
        )

    return {
        "variable": column,
        "type": "categorical",
        "statistic": chi2,
        "p_value": p,
        "psi": binary_proportion_shift(
            expected,
            actual,
            column,
        ),
    }


def compare_distributions(dataset_a, dataset_b):
    rows = []

    for column in CONTINUOUS:
        rows.append(
            continuous_shift(
                dataset_a,
                dataset_b,
                column,
            )
        )

    for column in CATEGORICAL:
        rows.append(
            categorical_shift(
                dataset_a,
                dataset_b,
                column,
            )
        )

    return pd.DataFrame(rows)
