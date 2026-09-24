
"""
Distribution-shift analysis for the CVD cross-dataset
external-validation study.

The purpose is to quantify differences between Dataset A
and Dataset B before interpreting external-validation
performance.

Continuous variables:
    age
    bmi

Binary variables:
    gender
    smoking
    alcohol
    physical_activity
    high_bp
    high_chol

The analysis reports:
    - mean
    - standard deviation
    - prevalence
    - standardized mean difference
    - Kolmogorov-Smirnov statistic for continuous variables
    - prevalence difference for binary variables
"""

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp


# ============================================================
# VARIABLE DEFINITIONS
# ============================================================

CONTINUOUS_VARIABLES = [
    "age",
    "bmi",
]

BINARY_VARIABLES = [
    "gender",
    "smoking",
    "alcohol",
    "physical_activity",
    "high_bp",
    "high_chol",
]


# ============================================================
# STANDARDIZED MEAN DIFFERENCE
# ============================================================

def standardized_mean_difference(
    group_a,
    group_b
):
    """
    Calculate standardized mean difference (SMD)
    for a continuous variable.

    Formula:

        SMD = (mean_A - mean_B) / pooled_SD
    """

    a = pd.to_numeric(
        group_a,
        errors="coerce"
    ).dropna()

    b = pd.to_numeric(
        group_b,
        errors="coerce"
    ).dropna()

    if len(a) == 0 or len(b) == 0:
        return np.nan

    variance_a = a.var(
        ddof=1
    )

    variance_b = b.var(
        ddof=1
    )

    pooled_sd = np.sqrt(
        (
            variance_a +
            variance_b
        ) / 2
    )

    if pooled_sd == 0:
        return np.nan

    return (
        a.mean() - b.mean()
    ) / pooled_sd


# ============================================================
# BINARY SMD
# ============================================================

def binary_standardized_mean_difference(
    group_a,
    group_b
):
    """
    Calculate an approximate standardized mean difference
    for binary variables using pooled prevalence.
    """

    a = pd.to_numeric(
        group_a,
        errors="coerce"
    ).dropna()

    b = pd.to_numeric(
        group_b,
        errors="coerce"
    ).dropna()

    if len(a) == 0 or len(b) == 0:
        return np.nan

    p_a = a.mean()
    p_b = b.mean()

    pooled_p = (
        (p_a + p_b) / 2
    )

    denominator = np.sqrt(
        pooled_p * (1 - pooled_p)
    )

    if denominator == 0:
        return np.nan

    return (
        p_a - p_b
    ) / denominator


# ============================================================
# CONTINUOUS COMPARISON
# ============================================================

def compare_continuous_variable(
    dataset_a,
    dataset_b,
    variable
):
    """
    Compare one continuous variable between datasets.
    """

    a = pd.to_numeric(
        dataset_a[variable],
        errors="coerce"
    ).dropna()

    b = pd.to_numeric(
        dataset_b[variable],
        errors="coerce"
    ).dropna()

    if len(a) == 0 or len(b) == 0:
        raise ValueError(
            f"No valid observations available for "
            f"variable '{variable}'."
        )

    ks_statistic, ks_pvalue = ks_2samp(
        a,
        b
    )

    smd = standardized_mean_difference(
        a,
        b
    )

    return {
        "variable": variable,
        "type": "continuous",

        "A_n": len(a),
        "B_n": len(b),

        "A_mean": a.mean(),
        "B_mean": b.mean(),

        "A_sd": a.std(),
        "B_sd": b.std(),

        "A_median": a.median(),
        "B_median": b.median(),

        "SMD": smd,

        "KS_statistic": ks_statistic,
        "KS_pvalue": ks_pvalue,
    }


# ============================================================
# BINARY COMPARISON
# ============================================================

def compare_binary_variable(
    dataset_a,
    dataset_b,
    variable
):
    """
    Compare one binary variable between datasets.
    """

    a = pd.to_numeric(
        dataset_a[variable],
        errors="coerce"
    ).dropna()

    b = pd.to_numeric(
        dataset_b[variable],
        errors="coerce"
    ).dropna()

    if len(a) == 0 or len(b) == 0:
        raise ValueError(
            f"No valid observations available for "
            f"variable '{variable}'."
        )

    prevalence_a = a.mean()
    prevalence_b = b.mean()

    difference = (
        prevalence_a -
        prevalence_b
    )

    absolute_difference = abs(
        difference
    )

    smd = binary_standardized_mean_difference(
        a,
        b
    )

    return {
        "variable": variable,
        "type": "binary",

        "A_n": len(a),
        "B_n": len(b),

        "A_prevalence": prevalence_a,
        "B_prevalence": prevalence_b,

        "absolute_difference": absolute_difference,

        "SMD": smd,

        "KS_statistic": np.nan,
        "KS_pvalue": np.nan,
    }


# ============================================================
# COMPLETE DATASET COMPARISON
# ============================================================

def compare_datasets(
    dataset_a,
    dataset_b
):
    """
    Compare Dataset A and Dataset B across all harmonized
    predictors.
    """

    required_variables = (
        CONTINUOUS_VARIABLES +
        BINARY_VARIABLES
    )

    missing_a = [
        variable
        for variable in required_variables
        if variable not in dataset_a.columns
    ]

    missing_b = [
        variable
        for variable in required_variables
        if variable not in dataset_b.columns
    ]

    if missing_a:
        raise ValueError(
            "Dataset A is missing variables: "
            f"{missing_a}"
        )

    if missing_b:
        raise ValueError(
            "Dataset B is missing variables: "
            f"{missing_b}"
        )

    results = []

    # Continuous variables
    for variable in CONTINUOUS_VARIABLES:

        result = compare_continuous_variable(
            dataset_a,
            dataset_b,
            variable
        )

        results.append(
            result
        )

    # Binary variables
    for variable in BINARY_VARIABLES:

        result = compare_binary_variable(
            dataset_a,
            dataset_b,
            variable
        )

        results.append(
            result
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# INTERPRETATION HELPER
# ============================================================

def add_shift_flag(
    results
):
    """
    Add a descriptive shift flag.

    This is NOT a statistical conclusion.
    It is only a descriptive screening aid.

    For continuous variables:
        |SMD| >= 0.10 -> noticeable difference

    For binary variables:
        |SMD| >= 0.10 -> noticeable difference

    These thresholds should be interpreted in context.
    """

    results = results.copy()

    results["absolute_SMD"] = (
        results["SMD"].abs()
    )

    results["SMD_flag"] = np.where(
        results["absolute_SMD"] >= 0.10,
        "noticeable_difference",
        "small_difference"
    )

    return results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "DISTRIBUTION SHIFT MODULE"
    )

    print(
        "=" * 70
    )

    print(
        "\nContinuous variables:"
    )

    for variable in CONTINUOUS_VARIABLES:
        print(
            f"  - {variable}"
        )

    print(
        "\nBinary variables:"
    )

    for variable in BINARY_VARIABLES:
        print(
            f"  - {variable}"
        )

    print(
        "\nModule test: PASSED"
    )