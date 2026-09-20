
 
import json
 
import numpy as np
import pandas as pd
 
from imblearn.over_sampling import SMOTE
 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
 
 
FEATURE_COLUMNS = [
    "age", "gender", "smoking", "bmi", "alcohol",
    "physical_activity", "high_bp", "high_chol", "high_gluc",
]
 
TARGET_COLUMN = "cvd"
 
 
def calculate_iqr_bounds(series, multiplier=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return float(lower), float(upper)
 
 
def split_dataset(df, seed=42):
    train_df, temp_df = train_test_split(
        df, test_size=0.20, stratify=df[TARGET_COLUMN], random_state=seed,
    )
    validation_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df[TARGET_COLUMN], random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
 
 
def preprocess_dataset(df, seed=42):
    train_df, validation_df, test_df = split_dataset(df, seed)
 
    # Fit outlier boundaries ONLY on training data ...
    bmi_lower, bmi_upper = calculate_iqr_bounds(train_df["bmi"])
 
    # ... then FIX: actually apply them to all three splits (this was missing).
    train_df = train_df[train_df["bmi"].between(bmi_lower, bmi_upper)].reset_index(drop=True)
    validation_df = validation_df[validation_df["bmi"].between(bmi_lower, bmi_upper)].reset_index(drop=True)
    test_df = test_df[test_df["bmi"].between(bmi_lower, bmi_upper)].reset_index(drop=True)
 
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
 
    X_train_raw = train_df[FEATURE_COLUMNS]
    X_val_raw = validation_df[FEATURE_COLUMNS]
    X_test_raw = test_df[FEATURE_COLUMNS]
 
    y_train = train_df[TARGET_COLUMN]
    y_val = validation_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]
 
    X_train = imputer.fit_transform(X_train_raw)
    X_val = imputer.transform(X_val_raw)
    X_test = imputer.transform(X_test_raw)
 
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
 
    smote = SMOTE(random_state=seed)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
 
    metadata = {
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "bmi_lower": bmi_lower,
        "bmi_upper": bmi_upper,
        "seed": seed,
    }
 
    return {
        "X_train": X_train_smote,
        "y_train": y_train_smote,
        "X_validation": X_val,
        "y_validation": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "imputer": imputer,
        "scaler": scaler,
        "metadata": metadata,
    }
 
 
def transform_external(df, imputer, scaler):
    X = df[FEATURE_COLUMNS]
    X = imputer.transform(X)
    X = scaler.transform(X)
    y = df[TARGET_COLUMN].astype(int)
    return X, y
 
 
def save_metadata(metadata, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)