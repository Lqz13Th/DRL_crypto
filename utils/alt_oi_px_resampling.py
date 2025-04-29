import polars as pl
import os

from tqdm import tqdm

from ppo_algo.datas.high_frequency_data_parser import ParseHFTData
from utils.polars_expr import rolling_scaled_sigmoid_expr
from utils.dates_utils import generate_dates

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

def rename_with_prefix(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    return df.rename({
        col: f"{prefix}{col}" for col in df.columns if col != "timestamp"
    })

def merge_dataframes_on_timestamp(dfs: list[pl.DataFrame], prefixes: list[str]) -> pl.DataFrame:
    assert len(dfs) == len(prefixes), "每个 DataFrame 需要一个对应的前缀"

    dfs_renamed = [rename_with_prefix(df, prefix) for df, prefix in zip(dfs, prefixes)]
    merged = dfs_renamed[0]
    for df in dfs_renamed[1:]:
        merged = (
            merged
            .join(df, on="timestamp", how="full")
            .with_columns(
                pl
                .when(pl.col("timestamp").is_null())
                .then(pl.col("timestamp_right"))
                .otherwise(pl.col("timestamp"))
                .alias("timestamp")
            )
        )

        merged = merged.drop("timestamp_right")

    return merged.sort("timestamp")

def auto_fill_dataframes_with_old_data(auto_fill_df: pl.DataFrame) -> pl.DataFrame:
    columns_to_fill = auto_fill_df.columns
    for col in columns_to_fill:
        auto_fill_df = auto_fill_df.with_columns(
            pl
            .col(col)
            .fill_null(strategy="forward")
            .alias(col)
        )

    return auto_fill_df

def generate_alt_oi_px_bar(
        input_df: pl.DataFrame,
        threshold: float,
) -> (pl.DataFrame, int):
    last_px = input_df[0, "price"]
    last_ts = input_df[0, "timestamp"]

    sampled_datas = []
    sum_buy_size = 0
    sum_sell_size = 0

    print(last_px)

    for row in tqdm(input_df.iter_rows(), desc='Processing daily sampled data', total=len(input_df)):
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
            sampled_datas.append(bar)

            last_px = px
            last_ts = ts
            sum_buy_size = 0
            sum_sell_size = 0


    sampled_df = pl.DataFrame(sampled_datas)
    sampled_df = sampled_df.drop_nulls()

    return sampled_df


def process_data_by_day_with_multiple_pairs(
        start_date: str,
        end_date: str,
        threshold: float,
        rolling_window: int,
        output_dir: str,
        target_instruments: list
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dates_list = generate_dates(start_date, end_date)
    tardis_trade_path_template = "C:/quant/data/tardis_data/datasets/binance-futures_trades_{date}_{symbol}.csv.gz"
    tardis_lob_path_template = "C:/quant/data/tardis_data/datasets/binance-futures_book_snapshot_25_{date}_{symbol}.csv.gz"
    alt_data_path_template = "C:/quant/data/binance_alt_data/alt_database/binance_futures_data/combined_alt_data/{symbol}_data.csv"

    psd = ParseHFTData()

    for ins in target_instruments:
        print(f"Processing data for instrument: {ins}...")
        alt_data_path = alt_data_path_template.format(symbol=ins)
        alt_df = (
            pl
            .read_csv(alt_data_path)
            .with_columns(
                (pl.col('timestamp') * 1000)
                .alias('timestamp')
            )
        )


        print(alt_df)

        for date in tqdm(dates_list, desc='Processing bars', total=len(dates_list)):
            tardis_trade_path = tardis_trade_path_template.format(date=date, symbol=ins)
            tardis_lob_path = tardis_lob_path_template.format(date=date, symbol=ins)

            trade_df = psd.parse_trade_data_tardis(tardis_trade_path)
            print("Parsed data for {ins} on {date}. Shape: {trade_df.shape}")

            lob_df = psd.parse_lob_data_tardis(tardis_lob_path)
            print(f"Parsed data for {ins} on {date}. Shape: {lob_df.shape}")

            ts_min = min(trade_df['timestamp'].min(), lob_df['timestamp'].min()) - 1 * 1000 * 1000
            ts_max = max(trade_df['timestamp'].max(), lob_df['timestamp'].max())

            print(ts_min, ts_max)

            alt_daily_df = alt_df.filter(
                (pl.col('timestamp') >= ts_min) &
                (pl.col('timestamp') <= ts_max)
            )
            merged_df = merge_dataframes_on_timestamp(
                [trade_df, lob_df, alt_daily_df],
                ["trades_", "lob_", "alt_"]
            )

            print(merged_df)
            merged_df = auto_fill_dataframes_with_old_data(merged_df)
            exit()



            pct_sampling = generate_alt_oi_px_bar(trade_df, lob_df, alt_df, threshold)
            print(f"Generated price percent change bars for {ins} on {date}. Shape: {pct_sampling.shape}")
            print(pct_sampling)
            pct_sampling.write_csv("a.csv")
            exit()
            # normalized_data = rolling_normalize_data(pct_sampling, rolling_window).drop_nulls()
            # print(f"Normalized data for {ins} on {date}. Shape: {normalized_data.shape}")
            #
            # symbol_folder = os.path.join(output_dir, ins)
            # if not os.path.exists(symbol_folder):
            #     os.makedirs(symbol_folder)
            #
            # date_folder = os.path.join(symbol_folder, date)
            # if not os.path.exists(date_folder):
            #     os.makedirs(date_folder)
            #
            # # 保存归一化后的数据到对应的币种和日期的文件夹
            # output_file_path = os.path.join(date_folder,
            #                                 f"normalized_data_{ins}_{date}_threshold{threshold}_rolling{rolling_window}.csv")
            # normalized_data.write_csv(output_file_path)
            # print(f"Saved normalized data for {ins} on {date} to {output_file_path}.")
            #
            # # 清理内存
            # del trade_df, lob_df, pct_sampling, normalized_data
            # print(f"Memory cleaned for {ins} on {date}.")


if __name__ == "__main__":
    from ppo_algo.datas.high_frequency_data_parser import ParseHFTData


    instruments = ["BTCUSDT", "ETHUSDT", "FILUSDT", "GUNUSDT", "JASMYUSDT"]
    output_directory = "C:/quant/data/binance_resampled_data"
    process_data_by_day_with_multiple_pairs(
        start_date="2025_04_06",
        end_date="2025_04_21",
        threshold=0.0002,
        rolling_window=500,
        output_dir=output_directory,
        target_instruments=instruments
    )


