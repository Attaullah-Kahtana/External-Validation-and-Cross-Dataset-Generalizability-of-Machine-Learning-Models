import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve


def calibration_statistics(
    y_true,
    probability
):

    y_true = np.asarray(y_true)
    probability = np.asarray(probability)

    eps = 1e-6

    clipped = np.clip(
        probability,
        eps,
        1 - eps
    )

    logit = np.log(
        clipped / (1 - clipped)
    )

    model = LogisticRegression()

    model.fit(
        logit.reshape(-1, 1),
        y_true
    )

    intercept = model.intercept_[0]
    slope = model.coef_[0][0]

    fraction_positive, mean_prediction = (
        calibration_curve(
            y_true,
            probability,
            n_bins=10,
            strategy="quantile"
        )
    )

    return {
        "calibration_intercept":
            intercept,

        "calibration_slope":
            slope,

        "observed_mean":
            np.mean(y_true),

        "predicted_mean":
            np.mean(probability),

        "calibration_bins":
            len(fraction_positive)
    }