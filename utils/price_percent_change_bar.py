import polars as pl
import pandas as pd

from tqdm import tqdm

from utils.normalizer import scaled_sigmoid
from utils.polars_expr import rolling_scaled_sigmoid_expr


def rolling_normalize_data(df: pl.DataFrame, window: int) -> pl.DataFrame:
    columns_to_normalize = df.columns

    normalized_df = df.with_columns(
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
    ).select(
        f"scaled_{column}" for column in columns_to_normalize
    )

    # df.with_columns(
    #     scaled_sum_buy_size=pl.col("sum_buy_size").rolling_mean(window),
    #     scaled_sum_sell_size=pl.col("sum_sell_size").rolling_mean(window),
    #     scaled_timestamp_duration=pl.col("timestamp_duration").rolling_mean(window),
    #     scaled_price_pct_change=pl.col("price_pct_change").rolling_mean(window),
    #     scaled_buy_sell_imbalance=pl.col("buy_sell_imbalance").rolling_mean(window),
    #     scaled_change_sidee=pl.col("change_side").rolling_mean(window),
    # )
    #
    # for column in ['sum_buy_size', 'sum_sell_size', 'timestamp_duration', 'price_pct_change',
    #                'buy_sell_imbalance', 'change_side']:
    #     rolling_mean = df_normalized[column].rolling(window=window, min_periods=1).mean()
    #     rolling_std = df_normalized[column].rolling(window=window, min_periods=1).std()
    #
    #     # 使用滚动均值和标准差进行scaled sigmoid归一化
    #     df_normalized[f'scaled_{column}'] = df_normalized.apply(
    #         lambda row: rolling_scaled_sigmoid(row[column], rolling_mean[row.name], rolling_std[row.name]),
    #         axis=1
    #     )
    print(normalized_df)
    return normalized_df


def generate_px_pct_bar(
        df: pl.DataFrame,
        threshold: float,
        window: int,
) -> pl.DataFrame:
    last_px = df[0, "price"]
    last_ts = df[0, "timestamp"]

    bars = []
    sum_buy_size = 0
    sum_sell_size = 0

    print(last_px)

    for i in tqdm(range(len(df)), desc='Processing bars'):
        row = df[i]
        px = row["price"].first()
        sz = row["amount"].first()
        ts = row["timestamp"].first()

        side = -1 if row["side"].first() == 'sell' else 1  # 卖方主导为 -1，买方主导为 1
        px_pct = (px - last_px) / last_px
        if side == 1:
            sum_buy_size += sz

        else:
            sum_sell_size += sz

        if abs(px_pct) > threshold:
            ts_duration = ts - last_ts

            bar = {
                "sum_buy_size": sum_buy_size,
                "sum_sell_size": sum_sell_size,
                "timestamp_duration": ts_duration,
                "price_pct_change": px_pct,
                'buy_sell_imbalance': sum_buy_size - sum_sell_size,
                "change_side": 1 if px_pct > 0 else 0,
            }
            bars.append(bar)

            last_px = px
            last_ts = ts
            sum_buy_size = 0
            sum_sell_size = 0

    bars_df = pl.DataFrame(bars)
    # bars_df = bars_df.with_columns(
    #     (bars_df["price_pct_change"].pct_change(1)).alias('future_price_pct_change'),
    #     (bars_df["future_price_pct_change"].pct_change(1)).alias('future_price_pct_change'),
    #
    # )
    #
    # # 计算 scaled_sigmoid_future_price_pct_change
    # bars_df = bars_df.with_columns(
    #     (bars_df["future_price_pct_change"].apply(
    #         lambda x: scaled_sigmoid(x, -threshold * float(window), threshold * float(window)))).alias(
    #         'scaled_sigmoid_future_price_pct_change')
    # )

    # 删除缺失值
    bars_df = bars_df.drop_nulls()

    return bars_df


if __name__ == "__main__":
    from ppo_algo.datas.high_frequency_data_parser import ParseHFTData

    psd = ParseHFTData()
    df = psd.parse_trade_data_tardis(
        "C:/Users/trade/PycharmProjects/DataGrabber/datasets/binance-futures_trades_2024-08-05_FILUSDT.csv.gz"
    )
    print(df)

    pct_sampling = generate_px_pct_bar(df, 0.001, 10)
    print(pct_sampling)

    normalized_data = rolling_normalize_data(pct_sampling, 200)
    print(normalized_data)


