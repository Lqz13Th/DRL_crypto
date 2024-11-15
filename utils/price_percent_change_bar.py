import polars as pl

from tqdm import tqdm

from utils.polars_expr import rolling_scaled_sigmoid_expr


def rolling_normalize_data(df: pl.DataFrame, window: int) -> pl.DataFrame:
    df = df.with_columns(pl.col("price_pct_change").rolling_sum(10).alias(f"price_pct_change_sum_10"))
    columns_to_normalize = [col for col in df.columns if col not in ['price', 'timestamp']]
    print(columns_to_normalize)
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
    )
    return normalized_df


def generate_px_pct_bar(
        df: pl.DataFrame,
        threshold: float,
) -> pl.DataFrame:
    last_px = df[0, "price"]
    last_ts = df[0, "timestamp"]

    bars = []
    sum_buy_size = 0
    sum_sell_size = 0

    print(last_px)

    for row in tqdm(df.iter_rows(), desc='Processing bars', total=len(df)):
        ts = row[0]
        px = row[1]
        sz = row[2]
        side = -1 if row[3] == 'sell' else 1  # 卖方主导为 -1，买方主导为 1
        px_pct = (px - last_px) / last_px
        if side == 1:
            sum_buy_size += sz

        else:
            sum_sell_size += sz

        if abs(px_pct) > threshold:
            ts_duration = ts - last_ts

            bar = {
                "ts": ts,
                "price": px,
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
    bars_df = bars_df.drop_nulls()

    return bars_df


if __name__ == "__main__":
    from ppo_algo.datas.high_frequency_data_parser import ParseHFTData
    from utils.dates_utils import generate_dates

    start_date = "2024_07_01"
    end_date = "2024_08_24"
    dates_list = generate_dates(start_date, end_date)

    path_template = "C:/Users/trade/PycharmProjects/DataGrabber/datasets/binance-futures_trades_{date}_FILUSDT.csv.gz"
    file_paths = [path_template.format(date=date) for date in dates_list]

    for path in file_paths:
        print(path)

    psd = ParseHFTData()
    df = psd.parse_trade_data_list_path_tardis(file_paths)
    print(df)

    pct_sampling = generate_px_pct_bar(df, 0.0002)
    print(pct_sampling)

    normalized_data = rolling_normalize_data(pct_sampling, 200).drop_nulls()
    print(normalized_data)
    normalized_data.write_csv("normalized_data_0.0002.csv")


