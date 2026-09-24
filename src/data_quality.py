import pandas as pd
import numpy as np


def calculate_bmi(height_cm, weight_kg):
    height_m = height_cm / 100.0

    with np.errstate(divide="ignore", invalid="ignore"):
        bmi = weight_kg / (height_m ** 2)

    return bmi


def quality_control_dataset_a(df):

    df = df.copy()

    initial_n = len(df)

    # Remove exact duplicate rows
    df = df.drop_duplicates().copy()

    # Age: Dataset A is commonly stored in days
    if "age" in df.columns:
        if df["age"].median() > 1000:
            df["age_years"] = df["age"] / 365.25
        else:
            df["age_years"] = df["age"]

    # BMI
    if "height" in df.columns and "weight" in df.columns:
        df["bmi"] = calculate_bmi(
            pd.to_numeric(df["height"], errors="coerce"),
            pd.to_numeric(df["weight"], errors="coerce")
        )

    # BP validity
    systolic_valid = (
        df["ap_hi"].between(70, 250, inclusive="both")
    )

    diastolic_valid = (
        df["ap_lo"].between(40, 150, inclusive="both")
    )

    bp_valid = systolic_valid & diastolic_valid

    df.loc[~bp_valid, ["ap_hi", "ap_lo"]] = np.nan

    # Age validity
    df.loc[
        ~df["age_years"].between(18, 100),
        "age_years"
    ] = np.nan

    # BMI validity
    df.loc[
        ~df["bmi"].between(10, 80),
        "bmi"
    ] = np.nan

    report = {
        "initial_rows": initial_n,
        "after_duplicate_removal": len(df),
        "invalid_bp": int((~bp_valid).sum()),
        "invalid_age": int(df["age_years"].isna().sum()),
        "invalid_bmi": int(df["bmi"].isna().sum()),
    }

    return df, report