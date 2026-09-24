from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
)

from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def get_models(seed=42):

    return {

        "RandomForest":
            RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1
            ),

        "ExtraTrees":
            ExtraTreesClassifier(
                n_estimators=500,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1
            ),

        "GradientBoosting":
            GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                random_state=seed
            ),

        "HistGradientBoosting":
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                random_state=seed
            ),

        "AdaBoost":
            AdaBoostClassifier(
                n_estimators=300,
                learning_rate=0.05,
                random_state=seed
            ),

        "XGBoost":
            XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1
            ),

        "LightGBM":
            LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                random_state=seed,
                verbosity=-1,
                n_jobs=-1
            ),

        "CatBoost":
            CatBoostClassifier(
                iterations=400,
                depth=6,
                learning_rate=0.05,
                random_seed=seed,
                verbose=False
            ),

        "DeepNeuralNetwork":
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                max_iter=500,
                early_stopping=True,
                random_state=seed
            )
    }