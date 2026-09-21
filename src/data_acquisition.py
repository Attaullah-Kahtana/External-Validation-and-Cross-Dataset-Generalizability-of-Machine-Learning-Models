from pathlib import Path
import datetime
import hashlib

import pandas as pd

from src.utils import (
    get_logger,
    ensure_project_directories,
    get_config_path,
)


logger = get_logger(__name__)


EXPECTED_A = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active", "cardio",
]

REQUIRED_B = [
    "Age", "Sex", "BMI", "Smoker", "HvyAlcoholConsump",
    "PhysActivity", "HighBP", "HighChol", "HeartDiseaseorAttack",
]


def load_csv(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"\nDataset file not found:\n{path}\n")

    logger.info(f"Loading: {path}")

    # The cardiovascular dataset is commonly distributed as semicolon-separated.
    # First try normal CSV parsing; if it produces one giant column, retry with ';'.
    df = pd.read_csv(path)

    if len(df.columns) == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(path, sep=";")

    df.columns = [str(col).strip() for col in df.columns]

    logger.info(
        f"Loaded {len(df):,} rows and {len(df.columns)} columns."
    )
    return df


def validate_dataset_a(df):
    missing = [col for col in EXPECTED_A if col not in df.columns]

    if missing:
        raise ValueError(
            f"Dataset A is missing columns:\n{missing}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return True


def validate_dataset_b(df):
    missing = [col for col in REQUIRED_B if col not in df.columns]

    if missing:
        raise ValueError(
            f"Dataset B is missing columns:\n{missing}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    diabetes_candidates = [
        "Diabetes",
        "Diabetes_binary",
        "Diabetes_012",
    ]

    diabetes_found = [
        col for col in diabetes_candidates
        if col in df.columns
    ]

    if not diabetes_found:
        raise ValueError(
            "Dataset B does not contain a recognized diabetes column.\n"
            f"Expected one of: {diabetes_candidates}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return True


def file_sha256(path):
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            sha.update(block)

    return sha.hexdigest()


def write_provenance(records, output_path):
    """Write provenance and use a timestamped fallback if the file is locked."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        pd.DataFrame(records).to_csv(
            output_path,
            index=False,
        )
    except PermissionError:
        fallback = output_path.with_name(
            f"{output_path.stem}_"
            f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
            f"{output_path.suffix}"
        )

        logger.warning(
            f"'{output_path}' is locked. "
            f"Writing provenance to '{fallback}' instead."
        )

        pd.DataFrame(records).to_csv(
            fallback,
            index=False,
        )


def acquire_all(config):
    ensure_project_directories(config)

    raw_dir = get_config_path(
        config,
        "raw_dir",
    )

    path_a = (
        raw_dir /
        config["datasets"]["A"]["raw_file"]
    )

    path_b = (
        raw_dir /
        config["datasets"]["B"]["raw_file"]
    )

    df_a = load_csv(path_a)
    df_b = load_csv(path_b)

    validate_dataset_a(df_a)
    validate_dataset_b(df_b)

    records = [
        {
            "dataset": "A",
            "source_type": config["datasets"]["A"]["source_type"],
            "file": str(path_a),
            "rows": len(df_a),
            "columns": len(df_a.columns),
            "sha256": file_sha256(path_a),
            "timestamp": datetime.datetime.now().isoformat(),
        },
        {
            "dataset": "B",
            "source_type": config["datasets"]["B"]["source_type"],
            "file": str(path_b),
            "rows": len(df_b),
            "columns": len(df_b.columns),
            "sha256": file_sha256(path_b),
            "timestamp": datetime.datetime.now().isoformat(),
        },
    ]

    write_provenance(
        records,
        raw_dir.parent / "provenance_log.csv",
    )

    logger.info("Dataset validation completed.")

    return df_a, df_b


if __name__ == "__main__":
    from src.utils import load_config

    config = load_config()

    df_a, df_b = acquire_all(config)

    print("\nDataset A:")
    print(df_a.shape)

    print("\nDataset B:")
    print(df_b.shape)

    print("\nDataset B columns:")
    print(list(df_b.columns))
