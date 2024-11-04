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
