
"""
Main analysis pipeline for:

Cross-Dataset External Validation and Generalizability
of Machine Learning Models for Cardiovascular Disease.

Study design:

    Dataset A
       |
       +--> Internal 5-fold CV
       |
       +--> Train final models
       |
       +--> External validation --> Dataset B
       |
       |
    Dataset B
       |
       +--> Internal 5-fold CV
       |
       +--> Train final models
       |
       +--> External validation --> Dataset A

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

IMPORTANT:
    Dataset B Diabetes_binary is NOT used as high_gluc because
    it is not equivalent to Dataset A's glucose measurement.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from src.data_acquisition import (
    load_dataset
)

from src.data_quality import (
    quality_control_dataset_a
)

from src.harmonization import (
    harmonize_dataset_a,
    harmonize_dataset_b,
    validate_harmonized_data
)

from src.train_internal import (
    run_internal_cv
)

from src.external_validation import (
    run_external_validation
)

from src.distribution_shift import (
    compare_datasets,
    add_shift_flag
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

N_SPLITS = 5


# ============================================================
# DIRECTORIES
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent

DATA_DIR = (
    PROJECT_ROOT /
    "data"
)

RAW_DIR = (
    DATA_DIR /
    "raw"
)

PROCESSED_DIR = (
    DATA_DIR /
    "processed"
)

RESULTS_DIR = (
    PROJECT_ROOT /
    "results"
)

FIGURES_DIR = (
    RESULTS_DIR /
    "figures"
)

PREDICTIONS_DIR = (
    RESULTS_DIR /
    "predictions"
)

MODELS_DIR = (
    PROJECT_ROOT /
    "models"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():

    directories = [
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        PREDICTIONS_DIR,
        MODELS_DIR,
    ]

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# FILE PATHS
# ============================================================

DATASET_A_FILE = (
    RAW_DIR /
    "dataset_a_cvd.csv"
)

DATASET_B_FILE = (
    RAW_DIR /
    "dataset_b_brfss.csv"
)

DATASET_A_PROCESSED = (
    PROCESSED_DIR /
    "dataset_a_processed.csv"
)

DATASET_B_PROCESSED = (
    PROCESSED_DIR /
    "dataset_b_processed.csv"
)


# ============================================================
# LOAD DATASETS
# ============================================================

def load_raw_datasets():

    print()
    print("=" * 70)
    print("STEP 1: LOADING RAW DATASETS")
    print("=" * 70)

    print()
    print("Loading Dataset A...")

    dataset_a = load_dataset(
        DATASET_A_FILE
    )

    print()
    print("Loading Dataset B...")

    dataset_b = load_dataset(
        DATASET_B_FILE
    )

    return (
        dataset_a,
        dataset_b
    )


# ============================================================
# PREPARE DATASETS
# ============================================================

def prepare_datasets():

    print()
    print("=" * 70)
    print("STEP 2: DATA QUALITY CONTROL AND HARMONIZATION")
    print("=" * 70)

    dataset_a_raw, dataset_b_raw = (
        load_raw_datasets()
    )

    # --------------------------------------------------------
    # DATASET A QUALITY CONTROL
    # --------------------------------------------------------

    print()
    print("Running Dataset A quality control...")

    dataset_a_clean, quality_report_a = (
        quality_control_dataset_a(
            dataset_a_raw
        )
    )

    print()
    print("Dataset A QC report:")

    for key, value in quality_report_a.items():

        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------
    # DATASET A HARMONIZATION
    # --------------------------------------------------------

    print()
    print("Harmonizing Dataset A...")

    dataset_a = harmonize_dataset_a(
        dataset_a_clean
    )

    # --------------------------------------------------------
    # DATASET B HARMONIZATION
    # --------------------------------------------------------

    print()
    print("Harmonizing Dataset B...")

    dataset_b = harmonize_dataset_b(
        dataset_b_raw
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print()
    print("Validating Dataset A...")

    validate_harmonized_data(
        dataset_a,
        "Dataset A"
    )

    print()
    print("Validating Dataset B...")

    validate_harmonized_data(
        dataset_b,
        "Dataset B"
    )

    # --------------------------------------------------------
    # SAVE QUALITY REPORT
    # --------------------------------------------------------

    quality_report = pd.DataFrame(
        [
            {
                "dataset": "A",
                **quality_report_a
            }
        ]
    )

    quality_report.to_csv(
        RESULTS_DIR /
        "data_quality_report.csv",
        index=False
    )

    # --------------------------------------------------------
    # SAVE PROCESSED DATASETS
    # --------------------------------------------------------

    dataset_a.to_csv(
        DATASET_A_PROCESSED,
        index=False
    )

    dataset_b.to_csv(
        DATASET_B_PROCESSED,
        index=False
    )

    print()
    print(
        f"Saved Dataset A: "
        f"{DATASET_A_PROCESSED}"
    )

    print(
        f"Saved Dataset B: "
        f"{DATASET_B_PROCESSED}"
    )

    return (
        dataset_a,
        dataset_b
    )


# ============================================================
# INTERNAL VALIDATION
# ============================================================

def run_internal_validation(
    dataset_a,
    dataset_b
):

    print()
    print("=" * 70)
    print("STEP 3: INTERNAL 5-FOLD CROSS-VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # DATASET A
    # --------------------------------------------------------

    print()
    print(
        "Running internal CV for Dataset A..."
    )

    results_a = run_internal_cv(
        dataset_a,
        "A",
        SEED,
        N_SPLITS
    )

    results_a.to_csv(
        RESULTS_DIR /
        "internal_cv_A.csv",
        index=False
    )

    print()
    print(
        "Dataset A internal CV results:"
    )

    print(
        results_a.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # DATASET B
    # --------------------------------------------------------

    print()
    print(
        "Running internal CV for Dataset B..."
    )

    results_b = run_internal_cv(
        dataset_b,
        "B",
        SEED,
        N_SPLITS
    )

    results_b.to_csv(
        RESULTS_DIR /
        "internal_cv_B.csv",
        index=False
    )

    print()
    print(
        "Dataset B internal CV results:"
    )

    print(
        results_b.to_string(
            index=False
        )
    )

    return (
        results_a,
        results_b
    )


# ============================================================
# EXTERNAL VALIDATION
# ============================================================

def run_external_validation_analysis(
    dataset_a,
    dataset_b
):

    print()
    print("=" * 70)
    print("STEP 4: BIDIRECTIONAL EXTERNAL VALIDATION")
    print("=" * 70)

    # ========================================================
    # A -> B
    # ========================================================

    print()
    print(
        "A → B"
    )

    print(
        "Training on Dataset A..."
    )

    print(
        "Testing on completely unseen Dataset B..."
    )

    results_a_to_b, predictions_a_to_b = (
        run_external_validation(
            dataset_a,
            dataset_b,
            "A",
            "B",
            SEED
        )
    )

    results_a_to_b_df = pd.DataFrame(
        results_a_to_b
    )

    results_a_to_b_df.to_csv(
        RESULTS_DIR /
        "external_A_to_B.csv",
        index=False
    )

    # Save predictions

    for model_name, predictions in (
        predictions_a_to_b.items()
    ):

        prediction_file = (
            PREDICTIONS_DIR /
            f"A_to_B_{model_name}.csv"
        )

        pd.DataFrame(
            {
                "y_true": dataset_b["cvd"].to_numpy(),
                "predicted_probability": predictions,
            }
        ).to_csv(
            prediction_file,
            index=False
        )

    print()
    print(
        "A → B results:"
    )

    print(
        results_a_to_b_df.to_string(
            index=False
        )
    )

    # ========================================================
    # B -> A
    # ========================================================

    print()
    print(
        "B → A"
    )

    print(
        "Training on Dataset B..."
    )

    print(
        "Testing on completely unseen Dataset A..."
    )

    results_b_to_a, predictions_b_to_a = (
        run_external_validation(
            dataset_b,
            dataset_a,
            "B",
            "A",
            SEED
        )
    )

    results_b_to_a_df = pd.DataFrame(
        results_b_to_a
    )

    results_b_to_a_df.to_csv(
        RESULTS_DIR /
        "external_B_to_A.csv",
        index=False
    )

    # Save predictions

    for model_name, predictions in (
        predictions_b_to_a.items()
    ):

        prediction_file = (
            PREDICTIONS_DIR /
            f"B_to_A_{model_name}.csv"
        )

        pd.DataFrame(
            {
                "y_true": dataset_a["cvd"].to_numpy(),
                "predicted_probability": predictions,
            }
        ).to_csv(
            prediction_file,
            index=False
        )

    print()
    print(
        "B → A results:"
    )

    print(
        results_b_to_a_df.to_string(
            index=False
        )
    )

    return (
        results_a_to_b_df,
        results_b_to_a_df
    )


# ============================================================
# DISTRIBUTION SHIFT
# ============================================================

def run_distribution_shift_analysis(
    dataset_a,
    dataset_b
):

    print()
    print("=" * 70)
    print("STEP 5: DISTRIBUTION-SHIFT ANALYSIS")
    print("=" * 70)

    print()
    print(
        "Comparing harmonized predictors "
        "between Dataset A and Dataset B..."
    )

    shift_results = compare_datasets(
        dataset_a,
        dataset_b
    )

    shift_results = add_shift_flag(
        shift_results
    )

    output_file = (
        RESULTS_DIR /
        "distribution_shift.csv"
    )

    shift_results.to_csv(
        output_file,
        index=False
    )

    print()
    print(
        shift_results.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved distribution-shift results to:"
    )

    print(
        output_file
    )

    return shift_results


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(
    dataset_a,
    dataset_b
):

    print()
    print("=" * 70)
    print("ANALYSIS PIPELINE FINISHED")
    print("=" * 70)

    print()
    print(
        "Dataset A:"
    )

    print(
        f"  Rows: {len(dataset_a):,}"
    )

    print(
        f"  Columns: {len(dataset_a.columns)}"
    )

    print()
    print(
        "Dataset B:"
    )

    print(
        f"  Rows: {len(dataset_b):,}"
    )

    print(
        f"  Columns: {len(dataset_b.columns)}"
    )

    print()
    print(
        "Harmonized predictors:"
    )

    for column in dataset_a.columns:

        if column != "cvd":

            print(
                f"  - {column}"
            )

    print()
    print(
        "Outcome:"
    )

    print(
        "  - cvd"
    )

    print()
    print(
        "Important output files:"
    )

    output_files = [
        RESULTS_DIR /
        "data_quality_report.csv",

        RESULTS_DIR /
        "internal_cv_A.csv",

        RESULTS_DIR /
        "internal_cv_B.csv",

        RESULTS_DIR /
        "external_A_to_B.csv",

        RESULTS_DIR /
        "external_B_to_A.csv",

        RESULTS_DIR /
        "distribution_shift.csv",
    ]

    for file in output_files:

        if file.exists():

            print(
                f"  ✓ {file}"
            )

        else:

            print(
                f"  ✗ {file} "
                "(not created)"
            )

    print()
    print(
        "=" * 70
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run():

    print()
    print("=" * 70)
    print(
        "CVD CROSS-DATASET EXTERNAL VALIDATION"
    )
    print("=" * 70)

    print()
    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Random seed: {SEED}"
    )

    print(
        f"Internal CV folds: {N_SPLITS}"
    )

    # --------------------------------------------------------
    # DIRECTORIES
    # --------------------------------------------------------

    create_directories()

    # --------------------------------------------------------
    # DATA PREPARATION
    # --------------------------------------------------------

    dataset_a, dataset_b = (
        prepare_datasets()
    )

    # --------------------------------------------------------
    # INTERNAL VALIDATION
    # --------------------------------------------------------

    run_internal_validation(
        dataset_a,
        dataset_b
    )

    # --------------------------------------------------------
    # EXTERNAL VALIDATION
    # --------------------------------------------------------

    run_external_validation_analysis(
        dataset_a,
        dataset_b
    )

    # --------------------------------------------------------
    # DISTRIBUTION SHIFT
    # --------------------------------------------------------

    run_distribution_shift_analysis(
        dataset_a,
        dataset_b
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print_final_summary(
        dataset_a,
        dataset_b
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run()