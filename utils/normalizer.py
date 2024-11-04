import numpy as np


def scaled_sigmoid(x: float, start: float, end: float) -> float:
    n = np.abs(start - end)
    score = 2 / (1 + np.exp(-np.log(40_000) * (x - start - n) / n + np.log(5e-3)))
    return score / 2


def rolling_scaled_sigmoid(x: float, mean: float, std: float) -> float:
    if std == 0:
        return 0.5

    z_score = (x - mean) / std
    return scaled_sigmoid(z_score, -2.5, 2.5)
