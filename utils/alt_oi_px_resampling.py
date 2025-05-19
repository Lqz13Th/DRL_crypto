import polars as pl
import os

from tqdm import tqdm

from ppo_algo.datas.high_frequency_data_parser import ParseHFTData
from utils.polars_expr import rolling_scaled_sigmoid_expr
from utils.dates_utils import generate_dates
from factors.lob_factors import *
from utils.polars_expr import *


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

def on_sampling(row, leverage):
    return {
        "impact_price_pct_ask_imn": impact_price_pct_ask_row(row, imn=cal_imn_usdt(leverage)),
        "impact_price_pct_bid_imn": impact_price_pct_bid_row(row, imn=cal_imn_usdt(leverage)),
    }

def generate_alt_oi_px_bar(
        input_df: pl.DataFrame,
        threshold: float,
) -> (pl.DataFrame, int):

    last_px = input_df[0, "trades_price"]
    last_ts = input_df[0, "timestamp"]

    sampled_datas = []
    sum_buy_size = 0
    sum_sell_size = 0


    for row in input_df.iter_rows(named=True):
        ts = row['timestamp']
        px = row['trades_price']
        sz = row['trades_amount']
        side = -1 if row['trades_side'] == 'sell' else 1  # 卖方主导为 -1，买方主导为 1
        px_pct = (px - last_px) / last_px
        if side == 1:
            sum_buy_size += sz

        else:
            sum_sell_size += sz

        if abs(px_pct) > threshold:
            ts_duration = ts - last_ts

            sampled_data = {
                "ts": ts,
                "price": px,
                "sum_buy_size": sum_buy_size,
                "sum_sell_size": sum_sell_size,
                "timestamp_duration": ts_duration,
                "price_pct_change": px_pct,
                'buy_sell_imbalance': sum_buy_size - sum_sell_size,
                "change_side": 1 if px_pct > 0 else 0,
                "alt_factor_long_term_oi_trend": row['alt_factor_long_term_oi_trend'],
                "alt_top_long_short_position_ratio_data_longShortRatio_scaled": row['alt_top_long_short_position_ratio_data_longShortRatio_rolling_mean'],
                "alt_trade_taker_long_short_ratio_data_buySellRatio_scaled": row['alt_trade_taker_long_short_ratio_data_buySellRatio_scaled'],
                "alt_factor_short_term_oi_trend_scaled": row['alt_factor_short_term_oi_trend_scaled'],
                "alt_factor_long_term_oi_trend_scaled": row['alt_factor_long_term_oi_trend_scaled'],
                "alt_factor_buy_sell_vol_diff_scaled": row['alt_factor_buy_sell_vol_diff_scaled'],
                **on_sampling(row, 125),
            }

            sampled_datas.append(sampled_data)

            last_px = px
            last_ts = ts
            sum_buy_size = 0
            sum_sell_size = 0


    sampled_df = pl.DataFrame(sampled_datas)
    sampled_df = sampled_df.drop_nulls()

    return sampled_df


def cal_factors_with_sampled_data(
        input_path: str,
) -> pl.DataFrame:
    EPSILON = 1e-6

    factors_df = (
        pl
        .scan_csv(input_path)
        .with_columns(
            pl.col("price_pct_change").rolling_sum(window_size=50).alias("rolling_px_pct_sum"),
        )
        .with_columns([
            pl.col("price").pct_change().alias("ret_1"),
            pl.col("price").pct_change().shift(1).alias("ret_1_lag"),
            pl.col("price").pct_change().rolling_mean(5).alias("ret_mean_5"),
            pl.col("price").pct_change().rolling_mean(10).alias("ret_mean_10"),
            pl.col("price").rolling_std(10).alias("volatility_10"),

            ((pl.col("sum_buy_size") - pl.col("sum_sell_size")) /
             (pl.col("sum_buy_size") + pl.col("sum_sell_size") + EPSILON))
            .alias("buy_sell_ratio"),

            pl.col("sum_buy_size").rolling_mean(10).alias("avg_buy_size_10"),
            pl.col("sum_sell_size").rolling_mean(10).alias("avg_sell_size_10"),

            pl.col("price_pct_change").rolling_sum(window_size=50).alias("rolling_px_pct_sum"),

            pl.col("change_side").rolling_sum(5).alias("change_side_sum_5"),
        ])
        .with_columns([
            (pl.col("rolling_px_pct_sum") / pl.col("alt_factor_short_term_oi_trend_scaled"))
            .alias("px_short_oi_divergence"),

            (pl.col("alt_factor_short_term_oi_trend_scaled") - pl.col("alt_factor_long_term_oi_trend_scaled"))
            .alias("oi_trend_slope"),

            (pl.col("price").diff().rolling_mean(100) - pl.col("alt_factor_long_term_oi_trend"))
            .alias("factor_px_oi_divergence"),

            (pl.col("rolling_px_pct_sum") * pl.col("sum_buy_size"))
            .alias("factor_momentum_volume"),

            (pl.col("rolling_px_pct_sum") * (pl.col("sum_buy_size") + pl.col("sum_sell_size")))
            .alias("impact_momentum"),

            (pl.col("rolling_px_pct_sum") / (pl.col("sum_buy_size") + pl.col("sum_sell_size") + 1e-6))
            .alias("impact_sensitivity"),
        ])
        .with_columns([
            ((pl.col("ret_1") > 0).cast(pl.Int8) * pl.col("buy_sell_ratio"))
            .alias("momentum_confirmed_by_orderflow"),

            ((pl.col("px_short_oi_divergence") > 0).cast(pl.Int8) * pl.col("impact_momentum"))
            .alias("oi_long_breakout_signal"),

            ((pl.col("px_short_oi_divergence") < 0).cast(pl.Int8) * pl.col("impact_momentum"))
            .alias("oi_short_breakout_signal"),
        ])
        .drop_nulls()
        .collect()
    )
    return factors_df


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
    alt_data_path_template = "C:/quant/data/binance_alt_data/alt_database/binance_futures_data/alt_factors_data/{symbol}_data.csv"

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
            lob_df = psd.parse_lob_data_tardis(tardis_lob_path)
            ts_min = min(trade_df['timestamp'].min(), lob_df['timestamp'].min()) - 1 * 1000 * 1000
            ts_max = max(trade_df['timestamp'].max(), lob_df['timestamp'].max())

            alt_daily_df = alt_df.filter(
                (pl.col('timestamp') >= ts_min) &
                (pl.col('timestamp') <= ts_max)
            )
            merged_df = merge_dataframes_on_timestamp(
                [trade_df, lob_df, alt_daily_df],
                ["trades_", "lob_", "alt_"]
            )

            auto_filled_df = auto_fill_dataframes_with_old_data(merged_df).drop_nulls()

            pct_sampling = generate_alt_oi_px_bar(auto_filled_df, threshold)
            normalized_data = rolling_normalize_data(pct_sampling, rolling_window).drop_nulls()
            symbol_folder = os.path.join(output_dir, ins)
            if not os.path.exists(symbol_folder):
                os.makedirs(symbol_folder)

            output_file_path = os.path.join(symbol_folder,
                                            f"normalized_data_{ins}_{date}_threshold{threshold}_rolling{rolling_window}.csv")
            normalized_data.write_csv(output_file_path)

            del trade_df, lob_df, pct_sampling, normalized_data

        output_path = merge_all_csvs_for_symbol(
            symbol=ins,
            input_dir=output_dir,
            output_dir=output_dir
        )

        factors_df = cal_factors_with_sampled_data(
            input_path=output_path,
        )

        normalized_factors_df = rolling_normalize_data(factors_df, rolling_window).drop_nulls()
        normalized_factors_df.write_csv(os.path.join(output_dir, f"{ins}_factors.csv"))
        print(f"{ins} 因子计算完成，共 {normalized_factors_df.shape[0]} 行。保存至：{output_path}")


def merge_all_csvs_for_symbol(symbol: str, input_dir: str, output_dir: str) -> str:
    symbol_folder = os.path.join(input_dir, symbol)
    all_files = [
        os.path.join(symbol_folder, f)
        for f in os.listdir(symbol_folder)
        if f.endswith(".csv") and f.startswith("normalized_data")
    ]

    # 全部读入并垂直拼接
    df_list = [pl.read_csv(f) for f in all_files]
    merged_df = pl.concat(df_list)

    # 可选：按时间戳排序
    merged_df = merged_df.sort("ts")

    # 保存
    output_path = os.path.join(output_dir, f"{symbol}_merged.csv")
    merged_df.write_csv(output_path)
    print(f"{symbol} 合并完成，共 {merged_df.shape[0]} 行。保存至：{output_path}")
    return output_path


if __name__ == "__main__":
    from ppo_algo.datas.high_frequency_data_parser import ParseHFTData


    # instruments = ["BTCUSDT", "ETHUSDT", "FILUSDT", "GUNUSDT", "JASMYUSDT"]
    instruments = ["BTCUSDT"]
    output_directory = "C:/quant/data/binance_resampled_data"
    process_data_by_day_with_multiple_pairs(
        start_date="2025_04_06",
        end_date="2025_04_20",
        threshold=0.0002,
        rolling_window=500,
        output_dir=output_directory,
        target_instruments=instruments
    )

