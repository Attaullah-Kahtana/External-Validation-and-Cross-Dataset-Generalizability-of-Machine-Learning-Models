from sklearn.base import clone

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

from src.preprocessing import (
    fit_preprocessor,
    transform_features
)

from src.model_zoo import get_models

from src.metrics import (
    calculate_metrics,
    bootstrap_ci
)


def external_validate_model(
    source_df,
    target_df,
    model,
    seed=42
):

    # ------------------------------------------------
    # STEP 1
    # Fit preprocessing ONLY on source dataset
    # ------------------------------------------------

    prep = fit_preprocessor(
        source_df
    )

    # ------------------------------------------------
    # STEP 2
    # Transform source
    # ------------------------------------------------

    X_source = transform_features(
        prep,
        source_df
    )

    # ------------------------------------------------
    # STEP 3
    # Transform target using source preprocessing
    # ------------------------------------------------

    X_target = transform_features(
        prep,
        target_df
    )

    y_source = (
        source_df["cvd"]
        .astype(int)
        .to_numpy()
    )

    y_target = (
        target_df["cvd"]
        .astype(int)
        .to_numpy()
    )

    # ------------------------------------------------
    # STEP 4
    # Train on complete source dataset
    # ------------------------------------------------

    fitted_model = clone(model)

    fitted_model.fit(
        X_source,
        y_source
    )

    # ------------------------------------------------
    # STEP 5
    # Predict completely unseen target dataset
    # ------------------------------------------------

    probability = (
        fitted_model
        .predict_proba(X_target)[:, 1]
    )

    # ------------------------------------------------
    # STEP 6
    # Metrics
    # ------------------------------------------------

    metrics = calculate_metrics(
        y_target,
        probability
    )

    # ------------------------------------------------
    # STEP 7
    # Confidence intervals
    # ------------------------------------------------

    auc_low, auc_high = bootstrap_ci(
        y_target,
        probability,
        roc_auc_score,
        seed=seed
    )

    pr_low, pr_high = bootstrap_ci(
        y_target,
        probability,
        average_precision_score,
        seed=seed + 1
    )

    metrics["roc_auc_low"] = auc_low
    metrics["roc_auc_high"] = auc_high

    metrics["pr_auc_low"] = pr_low
    metrics["pr_auc_high"] = pr_high

    return metrics, probability


def run_external_validation(
    source_df,
    target_df,
    source_name,
    target_name,
    seed=42
):

    models = get_models(seed)

    results = []

    predictions = {}

    for model_name, model in models.items():

        metrics, probability = (
            external_validate_model(
                source_df,
                target_df,
                model,
                seed
            )
        )

        metrics["train_dataset"] = (
            source_name
        )

        metrics["test_dataset"] = (
            target_name
        )

        metrics["model"] = (
            model_name
        )

        results.append(metrics)

        predictions[
            model_name
        ] = probability

    return results, predictions