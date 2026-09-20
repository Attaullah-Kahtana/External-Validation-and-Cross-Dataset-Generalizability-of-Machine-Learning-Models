
import numpy as np
import pandas as pd
 
 
FEATURE_COLUMNS = [
    "age", "gender", "smoking", "bmi", "alcohol",
    "physical_activity", "high_bp", "high_chol", "high_gluc",
]
 
TARGET_COLUMN = "cvd"
 
BINARY_COLUMNS = [
    "gender", "smoking", "alcohol", "physical_activity",
    "high_bp", "high_chol", "high_gluc",
]
 
 
def _numeric(series):
    return pd.to_numeric(series, errors="coerce")
 
 
def harmonize_dataset_a(df):
    out = pd.DataFrame(index=df.index)
 
    age = _numeric(df["age"])
    # Kaggle cardiovascular dataset commonly stores age in days.
    if age.max(skipna=True) > 120:
        age = age / 365.25
    out["age"] = age
 
    # FIX: Dataset A documents gender as 1=female, 2=male. Dataset B's Sex field
    # is already 0=female, 1=male and is passed through unchanged below, so
    # Dataset A must be mapped to the SAME convention (1=male, 0=female) here.
    out["gender"] = _numeric(df["gender"]).map({1: 0, 2: 1})
 
    out["smoking"] = _numeric(df["smoke"])
 
    height_m = _numeric(df["height"]) / 100.0
    weight = _numeric(df["weight"])
    out["bmi"] = weight / (height_m ** 2)
 
    out["alcohol"] = _numeric(df["alco"])
 
    active = _numeric(df["active"])
    if active.max(skipna=True) > 1:
        out["physical_activity"] = (active >= 30).astype(float)
    else:
        out["physical_activity"] = active
 
    ap_hi = _numeric(df["ap_hi"])
    ap_lo = _numeric(df["ap_lo"])
    out["high_bp"] = ((ap_hi >= 140) | (ap_lo >= 90)).astype(float)
 
    out["high_chol"] = (_numeric(df["cholesterol"]) >= 2).astype(float)
    out["high_gluc"] = (_numeric(df["gluc"]) >= 2).astype(float)
    out["cvd"] = _numeric(df["cardio"])
 
    return clean_harmonized_dataset(out)
 
 
def find_diabetes_column(df):
    candidates = ["Diabetes", "Diabetes_binary", "Diabetes_012"]
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError("No diabetes column found.")
 
 
def harmonize_dataset_b(df):
    out = pd.DataFrame(index=df.index)
    diabetes_column = find_diabetes_column(df)
 
    out["age"] = _numeric(df["Age"])
    out["gender"] = _numeric(df["Sex"])          # already 0=female, 1=male
    out["smoking"] = _numeric(df["Smoker"])
    out["bmi"] = _numeric(df["BMI"])
    out["alcohol"] = _numeric(df["HvyAlcoholConsump"])
    out["physical_activity"] = _numeric(df["PhysActivity"])
    out["high_bp"] = _numeric(df["HighBP"])
    out["high_chol"] = _numeric(df["HighChol"])
 
    diabetes = _numeric(df[diabetes_column])
    out["high_gluc"] = (diabetes > 0).astype(float)   # folds prediabetes+diabetes into 1
 
    out["cvd"] = _numeric(df["HeartDiseaseorAttack"])
 
    return clean_harmonized_dataset(out)
 
 
def clean_harmonized_dataset(df):
    df = df.copy()
 
    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
 
    for column in BINARY_COLUMNS:
        invalid = (df[column] < 0) | (df[column] > 1)
        df.loc[invalid, column] = np.nan
 
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
 
    return df[FEATURE_COLUMNS + [TARGET_COLUMN]].reset_index(drop=True)
 
 
def harmonize_all(df_a, df_b):
    processed_a = harmonize_dataset_a(df_a)
    processed_b = harmonize_dataset_b(df_b)
    return processed_a, processed_b