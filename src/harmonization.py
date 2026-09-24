
"""
Dataset harmonization module.

Converts Dataset A and Dataset B into a common analytical representation.

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

import pandas as pd
import numpy as np


# ============================================================
# COMMON FEATURE SET
# ============================================================

FEATURES = [
    "age",
    "gender",
    "smoking",
    "bmi",
    "alcohol",
    "physical_activity",
    "high_bp",
    "high_chol",
]

TARGET = "cvd"


# ============================================================
# DATASET A
# ============================================================

def harmonize_dataset_a(df):
    """
    Harmonize Dataset A into the common representation.

    Expected Dataset A columns include:
        age
        gender
        height
        weight
        ap_hi
        ap_lo
        cholesterol
        smoke
        alco
        active
        cardio

    Age is expected to have already been converted to years
    during data-quality processing.
    """

    required_columns = [
        "age_years",
        "gender",
        "smoke",
        "bmi",
        "alco",
        "active",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "cardio",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset A is missing required columns: {missing}"
        )

    out = pd.DataFrame(index=df.index)

    # Age in years
    out["age"] = pd.to_numeric(
        df["age_years"],
        errors="coerce"
    )

    # Sex / gender
    out["gender"] = pd.to_numeric(
        df["gender"],
        errors="coerce"
    )

    # Smoking
    out["smoking"] = pd.to_numeric(
        df["smoke"],
        errors="coerce"
    )

    # BMI
    out["bmi"] = pd.to_numeric(
        df["bmi"],
        errors="coerce"
    )

    # Alcohol consumption
    out["alcohol"] = pd.to_numeric(
        df["alco"],
        errors="coerce"
    )

    # Physical activity
    out["physical_activity"] = pd.to_numeric(
        df["active"],
        errors="coerce"
    )

    # Hypertension indicator
    #
    # Dataset A uses measured blood pressure.
    # High BP is defined here as:
    # systolic >= 140 OR diastolic >= 90
    out["high_bp"] = np.where(
        (df["ap_hi"] >= 140) |
        (df["ap_lo"] >= 90),
        1,
        0
    )

    # High cholesterol
    #
    # Dataset A cholesterol coding:
    # 1 = normal
    # 2 = above normal
    # 3 = well above normal
    out["high_chol"] = np.where(
        df["cholesterol"] >= 2,
        1,
        0
    )

    # Cardiovascular disease outcome
    out["cvd"] = pd.to_numeric(
        df["cardio"],
        errors="coerce"
    )

    return out


# ============================================================
# DATASET B
# ============================================================

def harmonize_dataset_b(df):
    """
    Harmonize Dataset B (BRFSS-style dataset).

    Actual Dataset B columns:

        Diabetes_binary
        HighBP
        HighChol
        CholCheck
        BMI
        Smoker
        Stroke
        HeartDiseaseorAttack
        PhysActivity
        Fruits
        Veggies
        HvyAlcoholConsump
        AnyHealthcare
        NoDocbcCost
        GenHlth
        MentHlth
        PhysHlth
        DiffWalk
        Sex
        Age
        Education
        Income

    Important:
        Dataset B does not contain a direct high-glucose variable.
        Therefore Diabetes_binary is NOT renamed to high_gluc.
    """

    required_columns = [
        "HighBP",
        "HighChol",
        "BMI",
        "Smoker",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "HvyAlcoholConsump",
        "Sex",
        "Age",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset B is missing required columns: {missing}"
        )

    out = pd.DataFrame(index=df.index)

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------
    #
    # BRFSS Age is an ordinal age-category variable.
    #
    # Standard BRFSS coding:
    #
    # 1 = 18-24
    # 2 = 25-29
    # 3 = 30-34
    # 4 = 35-39
    # 5 = 40-44
    # 6 = 45-49
    # 7 = 50-54
    # 8 = 55-59
    # 9 = 60-64
    # 10 = 65-69
    # 11 = 70-74
    # 12 = 75-79
    # 13 = 80+
    #
    # We use approximate midpoints.
    #
    age_mapping = {
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
        13: 82,
    }

    out["age"] = pd.to_numeric(
        df["Age"],
        errors="coerce"
    ).map(age_mapping)

    # --------------------------------------------------------
    # GENDER
    # --------------------------------------------------------

    out["gender"] = pd.to_numeric(
        df["Sex"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # SMOKING
    # --------------------------------------------------------

    out["smoking"] = pd.to_numeric(
        df["Smoker"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # BMI
    # --------------------------------------------------------

    out["bmi"] = pd.to_numeric(
        df["BMI"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # ALCOHOL
    # --------------------------------------------------------
    #
    # HvyAlcoholConsump:
    # 1 = heavy alcohol consumption
    # 0 = otherwise
    #
    # This is not identical to Dataset A's "alco".
    # The difference must be reported as a harmonization limitation.
    #

    out["alcohol"] = pd.to_numeric(
        df["HvyAlcoholConsump"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # PHYSICAL ACTIVITY
    # --------------------------------------------------------

    out["physical_activity"] = pd.to_numeric(
        df["PhysActivity"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # HIGH BLOOD PRESSURE
    # --------------------------------------------------------
    #
    # Dataset B provides HighBP directly.
    #

    out["high_bp"] = pd.to_numeric(
        df["HighBP"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # HIGH CHOLESTEROL
    # --------------------------------------------------------

    out["high_chol"] = pd.to_numeric(
        df["HighChol"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # CVD OUTCOME
    # --------------------------------------------------------

    out["cvd"] = pd.to_numeric(
        df["HeartDiseaseorAttack"],
        errors="coerce"
    )

    return out


# ============================================================
# VALIDATION
# ============================================================

def validate_harmonized_data(df, dataset_name="dataset"):
    """
    Check that harmonized data contains the expected
    variables and binary outcome.
    """

    required = FEATURES + [TARGET]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing harmonized columns: "
            f"{missing}"
        )

    # Check target values
    target_values = set(
        df[TARGET].dropna().unique()
    )

    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"{dataset_name}: target 'cvd' must contain "
            f"only 0 and 1. Found: {target_values}"
        )

    print(
        f"{dataset_name} harmonization validation: PASSED"
    )

    print(
        f"{dataset_name} shape: {df.shape}"
    )

    print(
        f"{dataset_name} missing values:"
    )

    print(
        df[required].isna().sum()
    )

    print(
        f"{dataset_name} CVD prevalence: "
        f"{df[TARGET].mean():.4f}"
    )

    return True