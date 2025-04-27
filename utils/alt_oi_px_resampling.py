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
        input_trade_df: pl.DataFrame,
        input_lob_df: pl.DataFrame,
        input_alt_df: pl.DataFrame,
        threshold: float,
        simulation_latency: int = 100 * 1_000,
        i_alt = 0,
) -> (pl.DataFrame, int):

    input_trade_df = input_trade_df.sort("timestamp")
    input_lob_df = input_lob_df.sort("timestamp")
    input_alt_df = input_alt_df.sort("timestamp")


    # init
    lock_trade, lock_lob = 0, 0


    # temp
    sync_ts = 0
    i_trade, i_lob, i_alt = 0, 0, i_alt
    row_trade, row_lob, row_alt = input_trade_df[i_trade], input_lob_df[i_lob], input_alt_df[i_alt]

    # sampled data
    sampled_datas = []
    sum_buy_size = 0
    sum_sell_size = 0

    while i_trade < len(input_trade_df) and i_lob < len(input_lob_df) and i_alt < len(input_alt_df):
        sum_buy_size = 0
        sum_sell_size = 0

        ts_trade = row_trade['timestamp'].to_list()[0]
        ts_lob = row_lob['timestamp'].to_list()[0]
        ts_alt = row_alt['timestamp'].to_list()[0]
        sync_ts = min(ts_trade, ts_lob, ts_alt)

        if ts_trade == sync_ts:
            row_trade = input_trade_df[i_trade]
            sync_ts = row_trade['timestamp'].to_list()[0]
            # side = row_trade['side']
            # size = row_trade['size']
            # if side == 'buy':
            #     sum_buy_size += size
            # else:
            #     sum_sell_size += size
            i_trade += 1
            lock_trade = 1
            # sampled_data = {
            #     "ts": ts_trade,
            #     "trade": 1,
            #     "lob": 0,
            #     "alt": 0,
            # }
            # sampled_datas.append(sampled_data)
            sum_buy_size = 1


        if ts_lob == sync_ts:
            row_lob = input_lob_df[i_lob]
            sync_ts = row_lob['timestamp'].to_list()[0]

            i_lob += 1
            lock_lob = 1
            # sampled_data = {
            #     "ts": ts_lob,
            #     "trade": 0,
            #     "lob": 1,
            #     "alt": 0,
            # }
            # sampled_datas.append(sampled_data)
            sum_sell_size = 1


        if ts_alt == sync_ts:
            row_alt = input_alt_df[i_alt]
            sync_ts = row_alt['timestamp'].to_list()[0]
            i_alt += 1
            # print(row_alt)
            if i_alt >= len(input_alt_df):
                i_alt -= 1
                break

            # sampled_data = {
            #     "ts": ts_alt,
            #     "trade": 0,
            #     "lob": 0,
            #     "alt": 1,
            # }
            # sampled_datas.append(sampled_data)
            # l_ts = ts_alt
            # print(l_ts)

        if lock_trade == lock_lob == 1:
            sampled_data = {
                "ts": sync_ts,
                "price": row_trade['price'].to_list()[0],
                "ask": row_lob['asks[0].price'].to_list()[0],
                'td': sum_buy_size,
                'lob': sum_sell_size,
            }
            sampled_datas.append(sampled_data)

    sampled_df = pl.DataFrame(sampled_datas)
    sampled_df = sampled_df.drop_nulls()

    return (sampled_df, i_alt)


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
        i_alt = 0
        # sampled_row =

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
            print(alt_daily_df)
            # 假设你有三个 df：trades_df, book_df, vol_df
            merged_df = merge_dataframes_on_timestamp(
                [trade_df, lob_df, alt_daily_df],
                ["trades_", "lob_", "alt_"]
            )

            print(merged_df)
            merged_df = auto_fill_dataframes_with_old_data(merged_df)
            merged_df.head(100000).write_csv("c.csv")
            print(auto_fill_dataframes_with_old_data(merged_df))
            exit()



            (pct_sampling, i_alt) = generate_alt_oi_px_bar(trade_df, lob_df, alt_df, threshold, i_alt)
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


