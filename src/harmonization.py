import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "age",
    "gender",
    "smoking",
    "bmi",
    "alcohol",
    "physical_activity",
    "high_bp",
    "high_chol",
    "high_gluc",
]

TARGET_COLUMN = "cvd"

BINARY_COLUMNS = [
    "gender",
    "smoking",
    "alcohol",
    "physical_activity",
    "high_bp",
    "high_chol",
    "high_gluc",
]

# Dataset B Age is the CDC 13-level five-year age-group code:
# 1=18-24, 2=25-29, ..., 12=75-79, 13=80+.
# Midpoints are used because Dataset A contains approximate continuous age.
AGE_GROUP_MIDPOINTS = {
    1: 21,
    2: 27,
    3: 32,
    4: 37,
    5: 42,
    6: 47,
    7: 52,
    8: 57,
    9: 62,
    10: 67,
    11: 72,
    12: 77,
    13: 80,
}


def _numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def harmonize_dataset_a(df):
    out = pd.DataFrame(index=df.index)

    age = _numeric(df["age"])

    # Dataset A commonly stores age in days.
    if age.max(skipna=True) > 120:
        age = age / 365.25

    out["age"] = age

    # Dataset A: 1=female, 2=male.
    # Dataset B: 0=female, 1=male.
    # Convert A to the same convention as B.
    out["gender"] = _numeric(
        df["gender"]
    ).map({
        1: 0,
        2: 1,
    })

    out["smoking"] = _numeric(
        df["smoke"]
    )

    height_m = (
        _numeric(df["height"]) / 100.0
    )

    weight = _numeric(
        df["weight"]
    )

    out["bmi"] = (
        weight /
        (height_m ** 2)
    )

    out["alcohol"] = _numeric(
        df["alco"]
    )

    out["physical_activity"] = _numeric(
        df["active"]
    )

    ap_hi = _numeric(
        df["ap_hi"]
    )

    ap_lo = _numeric(
        df["ap_lo"]
    )

    out["high_bp"] = (
        (ap_hi >= 140) |
        (ap_lo >= 90)
    ).astype(float)

    out["high_chol"] = (
        _numeric(df["cholesterol"]) >= 2
    ).astype(float)

    out["high_gluc"] = (
        _numeric(df["gluc"]) >= 2
    ).astype(float)

    out["cvd"] = _numeric(
        df["cardio"]
    )

    return clean_harmonized_dataset(out)


def find_diabetes_column(df):
    candidates = [
        "Diabetes",
        "Diabetes_binary",
        "Diabetes_012",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        "No diabetes column found. "
        f"Expected one of: {candidates}"
    )


def harmonize_dataset_b(df):
    out = pd.DataFrame(index=df.index)

    diabetes_column = find_diabetes_column(
        df
    )

    # FIX: Dataset B Age is the CDC 13-level five-year
    # age-group code, not raw age in years. Convert each
    # group to its approximate midpoint so it is comparable
    # with Dataset A's age in years.
    out["age"] = _numeric(
        df["Age"]
    ).map(
        AGE_GROUP_MIDPOINTS
    )

    # Dataset B already uses 0=female, 1=male.
    out["gender"] = _numeric(
        df["Sex"]
    )

    out["smoking"] = _numeric(
        df["Smoker"]
    )

    out["bmi"] = _numeric(
        df["BMI"]
    )

    out["alcohol"] = _numeric(
        df["HvyAlcoholConsump"]
    )

    out["physical_activity"] = _numeric(
        df["PhysActivity"]
    )

    out["high_bp"] = _numeric(
        df["HighBP"]
    )

    out["high_chol"] = _numeric(
        df["HighChol"]
    )

    diabetes = _numeric(
        df[diabetes_column]
    )

    # 0 = no diabetes/prediabetes; >0 = prediabetes or diabetes
    out["high_gluc"] = (
        diabetes > 0
    ).astype(float)

    out["cvd"] = _numeric(
        df["HeartDiseaseorAttack"]
    )

    return clean_harmonized_dataset(out)


def clean_harmonized_dataset(df):
    df = df.copy()

    for column in FEATURE_COLUMNS + [
        TARGET_COLUMN
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    for column in BINARY_COLUMNS:
        invalid = (
            (df[column] < 0) |
            (df[column] > 1)
        )

        df.loc[
            invalid,
            column
        ] = np.nan

    df = df.dropna(
        subset=[TARGET_COLUMN]
    )

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(int)
    )

    return df[
        FEATURE_COLUMNS + [TARGET_COLUMN]
    ].reset_index(
        drop=True
    )


def harmonize_all(df_a, df_b):
    processed_a = harmonize_dataset_a(
        df_a
    )

    processed_b = harmonize_dataset_b(
        df_b
    )

    return (
        processed_a,
        processed_b,
    )
