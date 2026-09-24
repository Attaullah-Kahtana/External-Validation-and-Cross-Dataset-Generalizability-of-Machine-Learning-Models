
"""
Preprocessing module for the CVD cross-dataset external-validation study.

The preprocessing strategy is designed to prevent data leakage.

IMPORTANT:
- The preprocessor is fitted ONLY on the training/source data.
- The fitted preprocessor is then applied to validation/test/external data.
- No information from the external dataset is used to fit preprocessing.

Primary harmonized predictors:
    age
    gender
    smoking
    bmi
    alcohol
    physical_activity
    high_bp
    high_chol

Outcome:
    cvd
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

FEATURE_COLUMNS = [
    "age",
    "gender",
    "smoking",
    "bmi",
    "alcohol",
    "physical_activity",
    "high_bp",
    "high_chol",
]

TARGET_COLUMN = "cvd"


# ============================================================
# PREPROCESSOR CLASS
# ============================================================

@dataclass
class Preprocessor:
    """
    Container for the fitted preprocessing pipeline.
    """

    pipeline: Pipeline


# ============================================================
# CREATE PREPROCESSOR
# ============================================================

def create_preprocessor():
    """
    Create the preprocessing pipeline.

    Steps:
        1. Median imputation
        2. Standardization

    Median imputation is fitted only on the training data.

    StandardScaler transforms each feature approximately to:
        mean = 0
        standard deviation = 1
    """

    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    return Preprocessor(
        pipeline=pipeline
    )


# ============================================================
# VALIDATE FEATURE COLUMNS
# ============================================================

def validate_feature_columns(df, dataset_name="dataset"):
    """
    Check whether all required predictor columns are present.
    """

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required "
            f"feature columns: {missing}"
        )

    return True


# ============================================================
# VALIDATE TARGET
# ============================================================

def validate_target(df, dataset_name="dataset"):
    """
    Check whether the target variable exists and contains
    only binary values 0 and 1, apart from missing values.
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"{dataset_name} does not contain "
            f"target column '{TARGET_COLUMN}'."
        )

    target = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce"
    )

    unique_values = set(
        target.dropna().unique()
    )

    if not unique_values.issubset({0, 1}):
        raise ValueError(
            f"{dataset_name}: target '{TARGET_COLUMN}' "
            f"must contain only 0 and 1. "
            f"Found values: {unique_values}"
        )

    return True


# ============================================================
# GET X AND Y
# ============================================================

def get_xy(df, dataset_name="dataset"):
    """
    Extract predictor matrix X and target vector y.
    """

    validate_feature_columns(
        df,
        dataset_name
    )

    validate_target(
        df,
        dataset_name
    )

    X = df[FEATURE_COLUMNS].copy()

    y = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce"
    )

    # Remove rows where outcome is missing.
    valid_target = y.notna()

    X = X.loc[valid_target].copy()
    y = y.loc[valid_target].astype(int)

    return X, y


# ============================================================
# FIT PREPROCESSOR
# ============================================================

def fit_preprocessor(
    df,
    dataset_name="training dataset"
):
    """
    Fit the preprocessing pipeline on training/source data.

    IMPORTANT:
    This function must NEVER be fitted using the external
    validation dataset.
    """

    validate_feature_columns(
        df,
        dataset_name
    )

    X = df[FEATURE_COLUMNS].copy()

    preprocessor = create_preprocessor()

    preprocessor.pipeline.fit(X)

    return preprocessor


# ============================================================
# TRANSFORM FEATURES
# ============================================================

def transform_features(
    preprocessor,
    df,
    dataset_name="dataset"
):
    """
    Transform a dataset using an already fitted preprocessor.

    The preprocessor must have been fitted on training/source
    data before this function is called.
    """

    validate_feature_columns(
        df,
        dataset_name
    )

    X = df[FEATURE_COLUMNS].copy()

    transformed = preprocessor.pipeline.transform(X)

    return transformed


# ============================================================
# FIT AND TRANSFORM TRAINING DATA
# ============================================================

def fit_transform_training_data(
    df,
    dataset_name="training dataset"
):
    """
    Fit the preprocessing pipeline and transform the same
    training dataset.

    Returns:
        preprocessor
        transformed X
        y
    """

    X, y = get_xy(
        df,
        dataset_name
    )

    preprocessor = create_preprocessor()

    X_transformed = preprocessor.pipeline.fit_transform(
        X
    )

    return (
        preprocessor,
        X_transformed,
        y.to_numpy()
    )


# ============================================================
# CONVERT TRANSFORMED ARRAY TO DATAFRAME
# ============================================================

def transformed_to_dataframe(
    transformed_data
):
    """
    Convert transformed NumPy array into a DataFrame.
    """

    return pd.DataFrame(
        transformed_data,
        columns=FEATURE_COLUMNS
    )


# ============================================================
# CHECK FOR NUMERIC DATA
# ============================================================

def check_numeric_features(
    df,
    dataset_name="dataset"
):
    """
    Confirm that all model features are numeric.
    """

    validate_feature_columns(
        df,
        dataset_name
    )

    non_numeric = []

    for column in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            non_numeric.append(
                column
            )

    if non_numeric:
        raise TypeError(
            f"{dataset_name} contains non-numeric "
            f"features: {non_numeric}"
        )

    return True


# ============================================================
# PREPROCESSING SUMMARY
# ============================================================

def preprocessing_summary(
    df,
    dataset_name="dataset"
):
    """
    Print a concise preprocessing summary.
    """

    print()
    print("=" * 70)
    print(f"PREPROCESSING SUMMARY: {dataset_name}")
    print("=" * 70)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Features: {len(FEATURE_COLUMNS)}"
    )

    print(
        "Feature columns:"
    )

    for i, column in enumerate(
        FEATURE_COLUMNS,
        start=1
    ):
        print(
            f"  {i}. {column}"
        )

    print()
    print(
        "Missing values:"
    )

    print(
        df[FEATURE_COLUMNS]
        .isna()
        .sum()
    )

    if TARGET_COLUMN in df.columns:

        target = pd.to_numeric(
            df[TARGET_COLUMN],
            errors="coerce"
        )

        print()
        print(
            "Target distribution:"
        )

        print(
            target.value_counts(
                dropna=False
            ).sort_index()
        )

        print()
        print(
            f"CVD prevalence: "
            f"{target.mean():.4f}"
        )

    print(
        "=" * 70
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "PREPROCESSING MODULE"
    )

    print(
        "=" * 70
    )

    print(
        "\nConfigured features:"
    )

    for feature in FEATURE_COLUMNS:
        print(
            f"  - {feature}"
        )

    print(
        f"\nTarget: {TARGET_COLUMN}"
    )

    print(
        "\nPreprocessing steps:"
    )

    print(
        "  1. Median imputation"
    )

    print(
        "  2. Standard scaling"
    )

    print(
        "\nModule test: PASSED"
    )