import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from src.preprocessing import (
    fit_preprocessor,
    transform_features
)

from src.model_zoo import get_models
from src.metrics import calculate_metrics


def cross_validate_model(
    df,
    model,
    seed=42,
    n_splits=5
):

    X = df.drop(columns=["cvd"])
    y = df["cvd"].astype(int).to_numpy()

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    oof_probability = np.zeros(len(df))

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y),
        start=1
    ):

        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        # Fit preprocessing ONLY on training fold
        prep = fit_preprocessor(
            train_df
        )

        X_train = transform_features(
            prep,
            train_df
        )

        X_test = transform_features(
            prep,
            test_df
        )

        y_train = train_df["cvd"].astype(int)
        y_test = test_df["cvd"].astype(int)

        fitted_model = clone(model)

        fitted_model.fit(
            X_train,
            y_train
        )

        probability = (
            fitted_model
            .predict_proba(X_test)[:, 1]
        )

        oof_probability[test_idx] = probability

        metrics = calculate_metrics(
            y_test,
            probability
        )

        metrics["fold"] = fold

        fold_results.append(metrics)

    fold_df = pd.DataFrame(
        fold_results
    )

    overall = calculate_metrics(
        y,
        oof_probability
    )

    return (
        fold_df,
        overall,
        oof_probability
    )


def run_internal_cv(
    df,
    dataset_name,
    seed=42,
    n_splits=5
):

    all_results = []

    models = get_models(seed)

    for name, model in models.items():

        fold_df, overall, oof = (
            cross_validate_model(
                df,
                model,
                seed,
                n_splits
            )
        )

        result = {
            "dataset":
                dataset_name,

            "model":
                name,

            **overall,

            "roc_auc_mean":
                fold_df["roc_auc"].mean(),

            "roc_auc_sd":
                fold_df["roc_auc"].std(),

            "pr_auc_mean":
                fold_df["pr_auc"].mean(),

            "pr_auc_sd":
                fold_df["pr_auc"].std()
        }

        all_results.append(result)

    return pd.DataFrame(
        all_results
    )