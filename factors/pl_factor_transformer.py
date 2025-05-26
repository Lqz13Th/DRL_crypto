import polars as pl
from typing import List


def rolling_skew_expr(col: str, window: int) -> pl.Expr:
    mean = pl.col(col).rolling_mean(window)
    std = pl.col(col).rolling_std(window) + 1e-8
    m3 = ((pl.col(col) - mean) ** 3).rolling_mean(window)
    return (m3 / (std ** 3)).alias(f"{col}_skew")


def rolling_kurt_expr(col: str, window: int) -> pl.Expr:
    mean = pl.col(col).rolling_mean(window)
    std = pl.col(col).rolling_std(window) + 1e-8
    m4 = ((pl.col(col) - mean) ** 4).rolling_mean(window)
    return (m4 / (std ** 4)).alias(f"{col}_kurt")

def diff_expr(col: str, lag: int = 1) -> pl.Expr:
    return (pl.col(col) - pl.col(col).shift(lag)).alias(f"{col}_diff_{lag}")

def second_order_diff_expr(col: str, lag: int = 1) -> pl.Expr:
    # 二阶差分 = 一阶差分的差分
    first_diff = pl.col(col) - pl.col(col).shift(lag)
    second_diff = first_diff - first_diff.shift(lag)
    return second_diff.alias(f"{col}_second_order_diff_{lag}")

def momentum_expr(col: str, lag: int = 200) -> pl.Expr:
    # 动量 = x_t - x_{t-lag}
    return (pl.col(col) - pl.col(col).shift(lag)).alias(f"{col}_momentum_{lag}")

def momentum_ratio_expr(col: str, lag: int = 200) -> pl.Expr:
    # 动量比率 = x_t / x_{t-lag}
    return (pl.col(col) / (pl.col(col).shift(lag) + 1e-8)).alias(f"{col}_momentum_ratio_{lag}")

def lag_expr(col: str, lag: int = 200) -> pl.Expr:
    return pl.col(col).shift(lag).alias(f"{col}_lag_{lag}")

def sigmoid_expr(col: str) -> pl.Expr:
    return (1 / (1 + (-pl.col(col)).exp())).alias(f"{col}_sigmoid")

def tanh_expr(col: str) -> pl.Expr:
    return pl.col(col).tanh().alias(f"{col}_tanh")

def boxcox_expr(col: str, lam: float = 0.0) -> pl.Expr:
    # Box-Cox: 当 lam=0时，等同于 log(x + 1)
    # 这里为了安全，先加1避免0或负数
    if lam == 0:
        return (pl.col(col) + 1).log().alias(f"{col}_boxcox_{lam}")
    else:
        return (((pl.col(col) + 1) ** lam - 1) / lam).alias(f"{col}_boxcox_{lam}")

def triple_cross_expr(cols: List[str]) -> List[pl.Expr]:
    exprs = []
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                exprs.append(
                    (pl.col(cols[i]) * pl.col(cols[j]) * pl.col(cols[k]))
                    .alias(f"{cols[i]}_X_{cols[j]}_X_{cols[k]}")
                )
    return exprs

def triple_max_mid_min_expr(cols: List[str]) -> List[pl.Expr]:
    exprs = []
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                max_col = pl.max_horizontal([pl.col(cols[i]), pl.col(cols[j]), pl.col(cols[k])])
                mean_col = pl.mean_horizontal([pl.col(cols[i]), pl.col(cols[j]), pl.col(cols[k])])
                min_col = pl.min_horizontal([pl.col(cols[i]), pl.col(cols[j]), pl.col(cols[k])])
                exprs.append(
                    ((max_col - mean_col) / (mean_col - min_col + 1e-8))
                    .alias(f"{cols[i]}_{cols[j]}_{cols[k]}_max_mean_min_ratio")
                )
    return exprs

def batch_apply_transforms(df: pl.DataFrame, window: int, lag: int, exclude_cols: List[str] = None) -> pl.DataFrame:
    if exclude_cols is None:
        exclude_cols = ['price', 'timestamp', 'symbol']

    if isinstance(df, pl.LazyFrame):
        cols = df.collect_schema().names()
    else:
        cols = df.columns

    base_cols = [
        col for col in cols
        if col not in exclude_cols and not (
            col.endswith('_rolling_mean') or
            col.endswith('_rolling_std') or
            col.endswith('_scaled')
        )
    ]

    exprs = []

    # single features transformation
    for col in base_cols:
        exprs.extend([
            rolling_skew_expr(col, window),
            rolling_kurt_expr(col, window),
            diff_expr(col, lag),
            second_order_diff_expr(col, lag),
            momentum_expr(col, lag),
            momentum_ratio_expr(col, lag),
            lag_expr(col, lag),
            sigmoid_expr(col),
            tanh_expr(col),
            boxcox_expr(col, lam=0),
            (pl.col(col) ** 2).alias(f"{col}_squared"),
            (pl.col(col).sqrt()).alias(f"{col}_sqrt"),
            (pl.col(col).log1p()).alias(f"{col}_log1p"),
        ])

    # double features transformation
    n = len(base_cols)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = base_cols[i], base_cols[j]
            exprs.extend([
                (pl.col(a) * pl.col(b)).alias(f"{a}_X_{b}"),
                (pl.col(a) / (pl.col(b) + 1e-8)).alias(f"{a}_DIV_{b}"),
                ((pl.col(a) / (pl.col(b) + 1e-8)).log()).alias(f"{a}_LOGR_{b}")
            ])

    # triple features transformation
    exprs.extend(triple_cross_expr(base_cols))
    exprs.extend(triple_max_mid_min_expr(base_cols))

    return df.with_columns(exprs)

