import polars as pl

from tqdm import tqdm

from utils.polars_expr import rolling_scaled_sigmoid_expr


def rolling_normalize_data(rollin_df: pl.DataFrame, window: int) -> pl.DataFrame:
    rollin_df = rollin_df.with_columns(pl.col("price_pct_change").rolling_sum(100).alias(f"price_pct_change_sum_100"))
    columns_to_normalize = [col for col in rollin_df.columns if col not in ['price', 'timestamp']]
    print(columns_to_normalize)
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