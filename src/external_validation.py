
import joblib
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    brier_score_loss,
)

from src.preprocessing import transform_external
from src.utils import (
    load_config,
    get_config_path,
)


def calculate_metrics(
    y_true,
    probability
):

    prediction = (
        probability >= 0.5
    ).astype(int)

    return {
        "roc_auc": roc_auc_score(
            y_true,
            probability
        ),

        "pr_auc": average_precision_score(
            y_true,
            probability
        ),

        "accuracy": accuracy_score(
            y_true,
            prediction
        ),

        "precision": precision_score(
            y_true,
            prediction,
            zero_division=0
        ),

        "sensitivity": recall_score(
            y_true,
            prediction,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            prediction,
            zero_division=0
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            prediction
        ),

        "brier_score": brier_score_loss(
            y_true,
            probability
        ),
    }


def validate_direction(
    train_dataset,
    external_dataset,
    config
):

    models_root = get_config_path(
        config,
        "models_dir"
    )

    processed_dir = get_config_path(
        config,
        "processed_dir"
    )

    results_dir = get_config_path(
        config,
        "results_dir"
    )

    model_dir = (
        models_root /
        f"dataset_{train_dataset.lower()}"
    )

    imputer = joblib.load(
        model_dir / "imputer.joblib"
    )

    scaler = joblib.load(
        model_dir / "scaler.joblib"
    )

    external_file = (
        processed_dir /
        config["datasets"][external_dataset]["processed_file"]
    )

    external_df = pd.read_csv(
        external_file
    )

    X_external, y_external = transform_external(
        external_df,
        imputer,
        scaler
    )

    results = []

    model_files = sorted(
        model_dir.glob("*.joblib")
    )

    model_files = [
        p for p in model_files
        if p.name not in [
            "imputer.joblib",
            "scaler.joblib",
        ]
    ]

    for model_file in model_files:

        model_name = (
            model_file.stem
        )

        print(
            f"External validation: {model_name}"
        )

        model = joblib.load(
            model_file
        )

        probability = model.predict_proba(
            X_external
        )[:, 1]

        metrics = calculate_metrics(
            y_external,
            probability
        )

        metrics["model"] = model_name

        results.append(metrics)

        prediction_df = pd.DataFrame({
            "y_true": y_external,
            "probability": probability,
        })

        prediction_df.to_csv(
            model_dir /
            f"{model_name}_external_predictions.csv",
            index=False
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "roc_auc",
        ascending=False
    )

    output_file = (
        results_dir /
        f"external_performance_{train_dataset}_to_{external_dataset}.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved external results:\n{output_file}"
    )

    return results_df


if __name__ == "__main__":

    config = load_config()

    validate_direction(
        "A",
        "B",
        config
    )