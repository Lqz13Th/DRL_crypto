import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def scaled_sigmoid(x: float, start: float, end: float) -> float:
    n = np.abs(start - end)
    score = 2 / (1 + np.exp(-np.log(40_000) * (x - start - n) / n + np.log(5e-3)))
    return score / 2


def rolling_scaled_sigmoid(x: float, mean: float, std: float) -> float:
    if std == 0:
        return 0.5

    z_score = (x - mean) / std
    return scaled_sigmoid(z_score, -2.5, 2.5)

# def scaled_sigmoid_dynamic(x, start, end):
#     n = np.abs(start - end)
#     if n == 0:
#         return 0.5
#     score = 2 / (1 + np.exp(-np.log(4_000_000) * (x - start - n) / n + np.log(5e-3)))
#     return score / 2
#
#
# def rolling_scaled_sigmoid_dynamic(series: pd.Series, window: int = 20) -> pd.Series:
#     rolling_mean = series.rolling(window).mean()
#     rolling_std = series.rolling(window).std()
#     z_scores = (series - rolling_mean) / rolling_std
#     rolling_min = z_scores.rolling(window).min()
#     rolling_max = z_scores.rolling(window).max()
#
#     result = [
#         scaled_sigmoid_dynamic(z, lo, hi)
#         if not (np.isnan(z) or np.isnan(lo) or np.isnan(hi))
#         else np.nan
#         for z, lo, hi in zip(z_scores, rolling_min, rolling_max)
#     ]
#     return pd.Series(result, index=series.index)
#
#
# # 示例数据
# np.random.seed(0)
# x = pd.Series(np.random.randn(20000).cumsum(), name="raw_series")
# y = rolling_scaled_sigmoid_dynamic(x, window=2000)
#
# # ✅ 双轴画图，尺度一致可读性强
# fig, ax1 = plt.subplots(figsize=(12, 5))
#
# color1 = "tab:blue"
# ax1.set_xlabel("Index")
# ax1.set_ylabel("Raw Series", color=color1)
# ax1.plot(x, color=color1, label="Raw Series")
# ax1.tick_params(axis='y', labelcolor=color1)
#
# ax2 = ax1.twinx()  # 第二个 y 轴
# color2 = "tab:red"
# ax2.set_ylabel("Scaled Sigmoid", color=color2)
# ax2.plot(y, color=color2, label="Scaled Sigmoid")
# ax2.tick_params(axis='y', labelcolor=color2)
#
# plt.title("Raw Series vs. Rolling Scaled Sigmoid (Dual Y-Axis)")
# fig.tight_layout()
# plt.show()
