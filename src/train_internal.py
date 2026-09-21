import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.model_zoo import get_models
from src.preprocessing import (
    preprocess_dataset,
    save_metadata,
)
from src.utils import (
    get_config_path,
    load_config,
    set_seed,
)


def calculate_metrics(y_true, probability):
    prediction = (
        probability >= 0.5
    ).astype(int)

    return {
        "roc_auc": roc_auc_score(
            y_true,
            probability,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probability,
        ),
        "accuracy": accuracy_score(
            y_true,
            prediction,
        ),
        "precision": precision_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "sensitivity": recall_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            prediction,
        ),
        "brier_score": brier_score_loss(
            y_true,
            probability,
        ),
    }


def train_direction(
    train_dataset,
    config,
):
    set_seed(
        config["seed"]
    )

    processed_dir = get_config_path(
        config,
        "processed_dir",
    )

    models_root = get_config_path(
        config,
        "models_dir",
    )

    results_dir = get_config_path(
        config,
        "results_dir",
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_file = (
        processed_dir /
        config["datasets"][train_dataset]["processed_file"]
    )

    if not data_file.exists():
        raise FileNotFoundError(
            f"Processed training dataset not found:\n"
            f"{data_file}\n"
            "Run the acquisition/harmonization phase first."
        )

    df = pd.read_csv(
        data_file
    )

    print(
        f"\nTraining on Dataset {train_dataset}"
    )

    print(
        f"Rows: {len(df):,}"
    )

    prep = preprocess_dataset(
        df,
        seed=config["seed"],
    )

    model_dir = (
        models_root /
        f"dataset_{train_dataset.lower()}"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        prep["imputer"],
        model_dir / "imputer.joblib",
    )

    joblib.dump(
        prep["scaler"],
        model_dir / "scaler.joblib",
    )

    save_metadata(
        prep["metadata"],
        model_dir / "preprocessing_metadata.json",
    )

    models = get_models(
        config["seed"]
    )

    results = []

    for name, model in models.items():
        print(
            f"\nTraining: {name}"
        )

        model.fit(
            prep["X_train"],
            prep["y_train"],
        )

        probability = model.predict_proba(
            prep["X_test"]
        )[:, 1]

        metrics = calculate_metrics(
            prep["y_test"],
            probability,
        )

        metrics["model"] = name
        results.append(metrics)

        joblib.dump(
            model,
            model_dir /
            f"{name}.joblib",
        )

        prediction_df = pd.DataFrame({
            "y_true": prep["y_test"],
            "probability": probability,
        })

        prediction_df.to_csv(
            model_dir /
            f"{name}_test_predictions.csv",
            index=False,
        )

        print(
            f"ROC-AUC: {metrics['roc_auc']:.4f}"
        )

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            "roc_auc",
            ascending=False,
        )
    )

    output_file = (
        results_dir /
        f"internal_performance_"
        f"{train_dataset}.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nSaved internal results:\n{output_file}"
    )

    return results_df


if __name__ == "__main__":
    config = load_config()

    train_direction(
        "A",
        config,
    )
