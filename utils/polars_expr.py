import polars as pl


def cal_z_score(x: pl.Expr, mean: pl.Expr, std: pl.Expr) -> pl.Expr:
    return (x - mean) / std


def scaled_sigmoid_expr(x: pl.Expr, start: pl.Expr, end: pl.Expr) -> pl.Expr:
    n = (start - end).abs()
    score = pl.lit(2) / (
            pl.lit(1) + (pl.lit(2.71828) ** (-pl.lit(40_0000).log(10) * ((x - start - n) / n) + pl.lit(5e-3).log(10)))
    )
    return score / pl.lit(2)


def rolling_scaled_sigmoid_expr(x: str, mean: str, std: str) -> pl.Expr:
    return (
        pl.when(pl.col(std) == 0)
        .then(pl.lit(0.5))
        .otherwise(
            scaled_sigmoid_expr(cal_z_score(pl.col(x), pl.col(mean), pl.col(std)), pl.lit(-2.), pl.lit(2.))
        )
        .alias(f"scaled_{x}")  # 列命名
    )

def rolling_sum_expr(col_name: str, window: int) -> pl.Expr:
    return pl.col(col_name).rolling_sum(window).alias(f"{col_name}_sum_{window}")

def rolling_normalize_data(rollin_df: pl.DataFrame, window: int) -> pl.DataFrame:
    columns_to_normalize = [
        col for col in rollin_df.columns
        if col not in ['price', 'timestamp', 'symbol']
           and not (col.endswith('_rolling_mean') or col.endswith('_rolling_std'))
    ]

    normalized_df = rollin_df.with_columns(
        [
            pl.col(column).rolling_mean(window).alias(f"{column}_rolling_mean")
            for column in columns_to_normalize
        ] + [
            pl.col(column).rolling_std(window).alias(f"{column}_rolling_std")
            for column in columns_to_normalize
        ]
    ).with_columns(
        rolling_scaled_sigmoid_expr(column, f"{column}_rolling_mean", f"{column}_rolling_std")
        for column in columns_to_normalize
    )
    return normalized_df