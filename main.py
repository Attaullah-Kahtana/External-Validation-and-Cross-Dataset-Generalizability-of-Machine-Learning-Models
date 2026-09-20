
import argparse

import pandas as pd

from src.utils import (
    load_config,
    ensure_project_directories,
)

from src.data_acquisition import acquire_all

from src.harmonization import harmonize_all

from src.train_internal import train_direction

from src.external_validation import validate_direction

from src.distribution_shift import compare_distributions


def run_acquisition(config):

    print("\n" + "=" * 60)
    print("STEP 1: DATA ACQUISITION")
    print("=" * 60)

    df_a, df_b = acquire_all(
        config
    )

    print(
        f"Dataset A: {df_a.shape}"
    )

    print(
        f"Dataset B: {df_b.shape}"
    )

    return df_a, df_b


def run_harmonization(
    config,
    df_a,
    df_b
):

    print("\n" + "=" * 60)
    print("STEP 2: DATASET HARMONIZATION")
    print("=" * 60)

    processed_a, processed_b = harmonize_all(
        df_a,
        df_b
    )

    processed_dir = config["paths"]["processed_dir"]

    processed_a.to_csv(
        f"{processed_dir}/dataset_a_processed.csv",
        index=False
    )

    processed_b.to_csv(
        f"{processed_dir}/dataset_b_processed.csv",
        index=False
    )

    print(
        f"Dataset A processed: {processed_a.shape}"
    )

    print(
        f"Dataset B processed: {processed_b.shape}"
    )

    return processed_a, processed_b


def run_direction(
    train_on,
    test_on,
    config
):

    print("\n" + "=" * 60)
    print(
        f"DIRECTION: {train_on} -> {test_on}"
    )
    print("=" * 60)

    train_direction(
        train_on,
        config
    )

    validate_direction(
        train_on,
        test_on,
        config
    )


def run_shift_analysis(
    config,
    dataset_a,
    dataset_b
):

    print("\n" + "=" * 60)
    print("DISTRIBUTION SHIFT")
    print("=" * 60)

    shift = compare_distributions(
        dataset_a,
        dataset_b
    )

    output = (
        config["paths"]["results_dir"]
        +
        "/distribution_shift_report.csv"
    )

    shift.to_csv(
        output,
        index=False
    )

    print(
        f"Saved: {output}"
    )


def main():

    parser = argparse.ArgumentParser(
        description="CVD External Validation Pipeline"
    )

    parser.add_argument(
        "--phase",
        choices=[
            "acquire",
            "train",
            "validate",
            "direction",
            "full",
        ],
        default="full"
    )

    parser.add_argument(
        "--train-on",
        choices=["A", "B"],
        default="A"
    )

    parser.add_argument(
        "--test-on",
        choices=["A", "B"],
        default="B"
    )

    parser.add_argument(
        "--reverse",
        action="store_true"
    )

    args = parser.parse_args()

    config = load_config()

    ensure_project_directories(
        config
    )

    if args.phase == "acquire":

        df_a, df_b = run_acquisition(
            config
        )

        run_harmonization(
            config,
            df_a,
            df_b
        )

    elif args.phase == "train":

        train_direction(
            args.train_on,
            config
        )

    elif args.phase == "validate":

        validate_direction(
            args.train_on,
            args.test_on,
            config
        )

    elif args.phase == "direction":

        run_direction(
            args.train_on,
            args.test_on,
            config
        )

    elif args.phase == "full":

        df_a, df_b = run_acquisition(
            config
        )

        processed_a, processed_b = run_harmonization(
            config,
            df_a,
            df_b
        )

        run_direction(
            "A",
            "B",
            config
        )

        run_shift_analysis(
            config,
            processed_a,
            processed_b
        )

        if args.reverse:

            run_direction(
                "B",
                "A",
                config
            )

    print("\n" + "=" * 60)
    print("PIPELINE FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()