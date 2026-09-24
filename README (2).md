# CVD External Validation

## Objective

This project evaluates the transportability and external performance of machine-learning models for cardiovascular disease prediction across two independent datasets.

## Datasets

### Dataset A
Clinical cardiovascular disease dataset.

### Dataset B
BRFSS Diabetes Health Indicators dataset.

## Harmonized Features

- age
- gender
- smoking
- bmi
- alcohol
- physical_activity
- high_bp
- high_chol
- high_gluc

Target:

- cvd

## Modeling

Models include:

- Random Forest
- Gradient Boosting
- Extra Trees
- HistGradientBoosting
- AdaBoost
- XGBoost
- LightGBM
- CatBoost
- Multilayer Perceptron

## Validation Strategy

The study evaluates:

1. Internal validation
2. A -> B external validation
3. B -> A external validation
4. Distribution shift

## Metrics

- ROC-AUC
- PR-AUC
- Accuracy
- Precision
- Sensitivity
- F1
- Balanced Accuracy
- Brier Score

## Reproducibility

Random seed:

42