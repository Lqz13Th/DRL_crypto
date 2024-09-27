import numpy as np


def scaled_sigmoid(x: float, start: float, end: float) -> float:
    """当`x`落在`[start,end]`区间时，函数值为[0,1]且在该区间有较好的响应灵敏度"""
    n = np.abs(start - end)
    score = 2 / (1 + np.exp(-np.log(40_000) * (x - start - n) / n + np.log(5e-3)))
    return score / 2
