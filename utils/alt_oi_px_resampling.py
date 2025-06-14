import os

from tqdm import tqdm

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
            .with_columns([
                pl.coalesce([pl.col("timestamp"), pl.col("timestamp_right")]).alias("timestamp")
            ])
            .drop("timestamp_right")
        )

    return merged.sort("timestamp")

def auto_fill_dataframes_with_old_data(auto_fill_df: pl.DataFrame) -> pl.DataFrame:
    auto_fill_df = auto_fill_df.sort("timestamp")
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
        "far_bid_price": row["lob_bids[19].price"],
        "far_ask_price": row["lob_asks[19].price"],
        "best_bid_price": row["lob_bids[0].price"],
        "best_ask_price": row["lob_asks[0].price"],
        "real_bid_amount_sum": calculate_bid_amount_sum_row(row),
        "real_ask_amount_sum": calculate_ask_amount_sum_row(row),
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
        trade_px = row['trades_price']
        bid_px = row['lob_bids[0].price']
        ask_px = row['lob_asks[0].price']

        px = np.clip(trade_px, bid_px, ask_px)

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
                # raw data
                "timestamp": ts,
                "px": px,
                "sum_buy_sz": sum_buy_size,
                "sum_sell_sz": sum_sell_size,
                "ts_duration": ts_duration,
                "px_pct": px_pct,
                "bs_imbalance": sum_buy_size - sum_sell_size,
                "top_acc_longShortRatio": row['alt_top_long_short_account_ratio_data_longShortRatio'],
                "top_pos_longShortRatio": row['alt_top_long_short_position_ratio_data_longShortRatio'],
                "acc_longShortRatio": row['alt_long_short_ratio_data_longShortRatio'],
                "trade_taker_buySellRatio": row['alt_trade_taker_long_short_ratio_data_buySellRatio'],
                "trade_taker_buy_vol": row['alt_trade_taker_long_short_ratio_data_buyVol'],
                "trade_taker_sell_vol": row['alt_trade_taker_long_short_ratio_data_sellVol'],
                "sum_open_interest": row['alt_open_interest_data_sumOpenInterest'],
                # factor raw data
                "raw_factor_oi_change_sum": row['alt_factor_oi_change_sum'],
                "raw_factor_oi_change_sum_long_term": row['alt_factor_oi_change_sum_long_term'],
                "raw_factor_short_term_oi_volatility": row['alt_factor_short_term_oi_volatility'],
                "raw_factor_long_term_oi_volatility": row['alt_factor_long_term_oi_volatility'],
                "raw_factor_short_term_oi_trend": row['alt_factor_short_term_oi_trend'],
                "raw_factor_long_term_oi_trend": row['alt_factor_long_term_oi_trend'],
                "raw_factor_buy_sell_vlm_diff": row['alt_factor_buy_sell_vlm_diff'],
                "raw_factor_sentiment_net": row['alt_factor_sentiment_net'],

                # factor data
                "z_factor_oi_change": row['alt_z_factor_oi_change_sum'],
                "z_factor_oi_change_long_term": row['alt_z_factor_oi_change_sum_long_term'],
                "z_factor_short_term_oi_volatility": row['alt_z_factor_short_term_oi_volatility'],
                "z_factor_long_term_oi_volatility": row['alt_z_factor_long_term_oi_volatility'],
                "z_factor_short_term_oi_trend": row['alt_z_factor_short_term_oi_trend'],
                "z_factor_long_term_oi_trend": row['alt_z_factor_long_term_oi_trend'],
                "z_factor_buy_sell_vlm_diff": row['alt_z_factor_buy_sell_vlm_diff'],
                "z_factor_sentiment_net": row['alt_z_factor_sentiment_net'],

                # aux data
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


def z_score_expr(col_name: str, window: int) -> pl.Expr:
    return (
            (pl.col(col_name) - pl.col(col_name)
             .rolling_mean(window, min_samples=1)
             .fill_null(strategy="zero"))
            / (pl.col(col_name)
               .rolling_std(window, min_samples=1)
               .fill_nan(0) + 1e-6)
    ).fill_null(0).alias(f"z_{col_name}")

def divergence_expr_with_sign(window: int) -> pl.Expr:
    px_sign = (
        pl.when(pl.col(f"z_px_pct_rol_sum_{window}") > 0)
          .then(1)
          .when(pl.col(f"z_px_pct_rol_sum_{window}") < 0)
          .then(-1)
          .otherwise(0)
    )

    oi_sign = (
        pl.when(pl.col("z_factor_oi_change") > 0)
        .then(1)
        .when(pl.col("z_factor_oi_change") < 0)
        .then(-1)
        .otherwise(0)
    )

    is_divergent = (px_sign * oi_sign) < 0

    return  (
        ((pl.col(f"z_px_pct_rol_sum_{window}") + pl.col("z_factor_oi_change").abs()) * is_divergent.cast(pl.Int8))
        .alias("factor_oi_px_divergence_with_sign")
    )

def divergence_expr_with_sign_long_term(window: int) -> pl.Expr:
    px_sign = (
        pl.when(pl.col(f"z_px_pct_rol_sum_{window}") > 0)
          .then(1)
          .when(pl.col(f"z_px_pct_rol_sum_{window}") < 0)
          .then(-1)
          .otherwise(0)
    )

    oi_sign = (
        pl.when(pl.col("z_factor_oi_change_long_term") > 0)
        .then(1)
        .when(pl.col("z_factor_oi_change_long_term") < 0)
        .then(-1)
        .otherwise(0)
    )

    is_divergent = (px_sign * oi_sign) < 0

    return  (
        ((pl.col(f"z_px_pct_rol_sum_{window}") + pl.col("z_factor_oi_change_long_term").abs()) * is_divergent.cast(pl.Int8))
        .alias("factor_oi_px_divergence_with_sign_long_term")
    )

def cal_factors_with_sampled_data(
        input_path: str,
        window: int,
) -> pl.DataFrame:
    EPSILON = 1e-6

    factors_df = (
        pl
        .scan_csv(input_path)
        .with_columns([
            pl.col("px_pct").rolling_sum(10).alias(f"px_pct_rol_sum_{10}"),
            pl.col("px_pct").rolling_sum(20).alias(f"px_pct_rol_sum_{20}"),
            pl.col("px_pct").rolling_sum(40).alias(f"px_pct_rol_sum_{40}"),
            pl.col("px_pct").rolling_sum(80).alias(f"px_pct_rol_sum_{80}"),
            pl.col("px_pct").rolling_sum(160).alias(f"px_pct_rol_sum_{160}"),
            pl.col("px_pct").rolling_sum(window).alias(f"px_pct_rol_sum_{window}"),

            (-pl.col("ts_duration").rolling_mean(window).add(EPSILON).log())
            .alias(f"ts_velo_rol_mean_{window}"),

            (pl.col("far_bid_price") - pl.col("best_bid_price")).abs().rolling_mean(window)
            .alias(f"bid_px_gap_rol_mean_{window}"),

            (pl.col("far_ask_price") - pl.col("best_ask_price")).abs().rolling_mean(window)
            .alias(f"ask_px_gap_rol_mean_{window}"),

            (pl.col("real_bid_amount_sum") - pl.col("real_ask_amount_sum"))
            .rolling_mean(window)
            .alias(f"lob_ratio_rol_mean_{window}"),

            pl.col("bs_imbalance").rolling_mean(window).alias(f"bs_imba_rol_mean_{window}"),

            (pl.col("sum_buy_sz") + pl.col("sum_sell_sz"))
            .rolling_mean(window)
            .alias(f"sum_sz_rol_mean_{window}"),

            ((pl.col("sum_buy_sz") - pl.col("sum_sell_sz")) /
             (pl.col("sum_buy_sz") + pl.col("sum_sell_sz") + EPSILON))
            .rolling_mean(window)
            .alias(f"bs_ratio_rol_mean_{window}"),
        ])
        .with_columns([
            (pl.col(f"sum_sz_rol_mean_{window}") * pl.col(f"px_pct_rol_sum_{window}"))
            .alias(f"sum_sz_px_pct_rol_sum_{window}"),

            (pl.col(f"ts_velo_rol_mean_{window}") * pl.col(f"px_pct_rol_sum_{window}"))
            .alias(f"px_velo_rol_mean_{window}"),

            ((pl.col(f"bid_px_gap_rol_mean_{window}") * pl.col("real_bid_amount_sum")) ** 2 -
             (pl.col(f"ask_px_gap_rol_mean_{window}") * pl.col("real_ask_amount_sum")) ** 2)
            .rolling_mean(window)
            .alias(f"lob_sz_imba_rol_mean_{window}"),

            (pl.col("sum_open_interest") - pl.col(f"px_pct_rol_sum_{window}"))
            .alias(f"oi_px_diff_{window}"),

            pl.when(
                (pl.col(f"px_pct_rol_sum_{window}") > 0) & (pl.col("raw_factor_oi_change_sum") < 0)
            ).then(
                pl.col(f"px_pct_rol_sum_{window}") * pl.col("raw_factor_oi_change_sum").abs()
            ).otherwise(0.0)
            .alias("oi_up_divergence"),

            pl.when(
                (pl.col(f"px_pct_rol_sum_{window}") < 0) & (pl.col("raw_factor_oi_change_sum") > 0)
            ).then(
                pl.col(f"px_pct_rol_sum_{window}") * pl.col("raw_factor_oi_change_sum").abs()
            ).otherwise(0.0)
            .alias("oi_down_divergence"),

            pl.when(
                (pl.col(f"px_pct_rol_sum_{window}") > 0) & (pl.col("raw_factor_oi_change_sum_long_term") < 0)
            ).then(
                pl.col(f"px_pct_rol_sum_{window}") * pl.col("raw_factor_oi_change_sum_long_term").abs()
            ).otherwise(0.0)
            .alias("oi_up_divergence_long_term"),

            pl.when(
                (pl.col(f"px_pct_rol_sum_{10}") < 0) & (pl.col("raw_factor_oi_change_sum_long_term") > 0)
            ).then(
                pl.col(f"px_pct_rol_sum_{10}") * pl.col("raw_factor_oi_change_sum_long_term").abs()
            ).otherwise(0.0)
            .alias("oi_down_divergence_long_term"),

            pl.when(
                (pl.col(f"px_pct_rol_sum_{10}") > 0) & (pl.col("raw_factor_oi_change_sum") < 0)
            ).then(
                pl.col(f"px_pct_rol_sum_{10}") * pl.col("raw_factor_oi_change_sum").abs()
            ).otherwise(0.0)
            .alias("oi_up_divergence_short_term"),

            pl.when(
                (pl.col(f"px_pct_rol_sum_{10}") < 0) & (pl.col("raw_factor_oi_change_sum") > 0)
            ).then(
                pl.col(f"px_pct_rol_sum_{10}") * pl.col("raw_factor_oi_change_sum").abs()
            ).otherwise(0.0)
            .alias("oi_down_divergence_short_term"),
        ])
        .with_columns([
            z_score_expr(f"px_pct_rol_sum_{10}", window),
            z_score_expr(f"px_pct_rol_sum_{20}", window),
            z_score_expr(f"px_pct_rol_sum_{40}", window),
            z_score_expr(f"px_pct_rol_sum_{80}", window),
            z_score_expr(f"px_pct_rol_sum_{160}", window),
            z_score_expr(f"px_pct_rol_sum_{window}", window),
            z_score_expr(f"ts_velo_rol_mean_{window}", window),
            z_score_expr(f"bid_px_gap_rol_mean_{window}", window),
            z_score_expr(f"ask_px_gap_rol_mean_{window}", window),
            z_score_expr(f"lob_ratio_rol_mean_{window}", window),
            z_score_expr(f"bs_imba_rol_mean_{window}", window),
            z_score_expr(f"sum_sz_rol_mean_{window}", window),
            z_score_expr(f"bs_ratio_rol_mean_{window}", window),
            z_score_expr(f"sum_sz_px_pct_rol_sum_{window}", window),
            z_score_expr(f"px_velo_rol_mean_{window}", window),
            z_score_expr(f"lob_sz_imba_rol_mean_{window}", window),
            z_score_expr(f"oi_px_diff_{window}", window),
            z_score_expr("oi_up_divergence", window),
            z_score_expr("oi_down_divergence", window),
            z_score_expr("oi_up_divergence_long_term", window),
            z_score_expr("oi_down_divergence_long_term", window),
            z_score_expr("oi_up_divergence_short_term", window),
            z_score_expr("oi_down_divergence_short_term", window),
        ])
        .with_columns([
            (pl.col("z_oi_up_divergence") + pl.col("z_oi_down_divergence")).alias("oi_di"),
            (pl.col("oi_up_divergence_long_term") + pl.col("oi_down_divergence_long_term")).alias("oi_di_long_term"),
            (pl.col("oi_up_divergence_short_term") + pl.col("oi_down_divergence_short_term")).alias("oi_di_short_term"),

            (pl.col(f"bs_imba_rol_mean_{window}") - pl.col(f"px_pct_rol_sum_{window}")).alias("taker_px_pct_diff"),

            (pl.col(f"z_px_pct_rol_sum_{window}") / (pl.col("z_factor_short_term_oi_trend").abs() + EPSILON))
            .alias("factor_px_oi_force"),

            (pl.col(f"z_px_pct_rol_sum_{window}") / (pl.col("z_factor_long_term_oi_trend").abs() + EPSILON))
            .alias("factor_px_oi_long_term_force"),

            (pl.col("z_factor_short_term_oi_trend") - pl.col("z_factor_long_term_oi_trend"))
            .alias("factor_oi_trend_slope"),

            (pl.col(f"z_px_pct_rol_sum_{window}") * pl.col(f"z_sum_sz_rol_mean_{window}"))
            .alias("factor_impact_momentum"),

            (pl.col(f"z_px_pct_rol_sum_{window}") / (pl.col(f"z_sum_sz_rol_mean_{window}") + EPSILON))
            .alias("factor_impact_sensitivity"),

            (pl.col(f"z_px_pct_rol_sum_{window}") * pl.col(f"z_bs_ratio_rol_mean_{window}"))
            .alias("factor_orderflow_sz_momentum"),

            divergence_expr_with_sign(window),
            divergence_expr_with_sign_long_term(window),
        ])
        .with_columns([
            z_score_expr("oi_di", window),
            z_score_expr("oi_di_long_term", window),
            z_score_expr("factor_oi_px_divergence_with_sign", window),
            z_score_expr("factor_oi_px_divergence_with_sign_long_term", window),
            z_score_expr("taker_px_pct_diff", window),
            z_score_expr("factor_px_oi_force", window),
            z_score_expr("factor_px_oi_long_term_force", window),
            z_score_expr("factor_oi_trend_slope", window),
            z_score_expr("factor_impact_momentum", window),
            z_score_expr("factor_impact_sensitivity", window),
            z_score_expr("factor_orderflow_sz_momentum", window),
        ])
        .with_columns([
            (pl.col("z_factor_px_oi_force") * pl.col("z_factor_orderflow_sz_momentum"))
            .alias("factor_oi_breakout_signal"),

            (pl.col("z_factor_impact_momentum") * pl.col("z_factor_oi_trend_slope").abs())
            .alias("factor_momentum_trend_confirm"),

            (pl.col("z_factor_orderflow_sz_momentum") * pl.col("z_factor_sentiment_net").abs())
            .alias("factor_order_sentiment_divergence"),

            (pl.col("z_factor_impact_momentum") * pl.col("z_factor_oi_change"))
            .alias("factor_oi_momentum_punch"),

            (pl.col("z_factor_impact_momentum") * pl.col("z_factor_oi_change_long_term"))
            .alias("factor_oi_momentum_long_term_punch"),

        ])
        .with_columns([
            z_score_expr("factor_oi_breakout_signal", window),
            z_score_expr("factor_momentum_trend_confirm", window),
            z_score_expr("factor_order_sentiment_divergence", window),
            z_score_expr("factor_oi_momentum_punch", window),
            z_score_expr("factor_oi_momentum_long_term_punch", window),

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
        target_instruments: list,
        resample: bool,
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

        alt_df = pl.scan_csv(alt_data_path)

        cols = [col for col in alt_df.collect_schema().names() if col != '']

        alt_df = (
            alt_df
            .select(cols)
            .with_columns([
                pl.col(col).cast(
                    pl.Int64 if col == "timestamp" else
                    pl.Utf8 if col == "symbol" else
                    pl.Float64
                ) for col in cols
            ])
            .with_columns(
                (pl.col("timestamp") * 1000).alias("timestamp")
            )
            .collect()
        )

        print(alt_df)

        if resample:
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
                del trade_df, lob_df, alt_daily_df,

                auto_filled_df = auto_fill_dataframes_with_old_data(merged_df).drop_nulls()
                print(auto_filled_df)
                pct_sampling = generate_alt_oi_px_bar(auto_filled_df, threshold)
                symbol_folder = os.path.join(output_dir, ins)
                if not os.path.exists(symbol_folder):
                    os.makedirs(symbol_folder)

                output_file_path = os.path.join(
                    symbol_folder,
                    f"resampled_data_{ins}_{date}_threshold{threshold}.csv"
                )

                pct_sampling.write_csv(output_file_path)
                del merged_df, pct_sampling

        output_path = merge_all_csvs_for_symbol(
            symbol=ins,
            dates_list=dates_list,
            input_dir=output_dir,
            output_dir=output_dir,
            threshold=threshold,
            rolling_window=rolling_window,
        )

        factors_df = cal_factors_with_sampled_data(
            input_path=output_path,
            window=rolling_window
        )

        output_filename = f"{ins}_factors_threshold{threshold}_rolling{rolling_window}.csv"
        output_path = os.path.join(output_dir, output_filename)

        factors_df.write_csv(output_path)
        print(f"{ins} 因子计算完成，共 {factors_df.shape[0]} 行。保存至：{output_path}")


def merge_all_csvs_for_symbol(
        symbol: str,
        dates_list: list,
        input_dir: str,
        output_dir: str,
        threshold: float,
        rolling_window: int,
) -> str:
    symbol_folder = os.path.join(input_dir, symbol)

    target_files = [
        f"resampled_data_{symbol}_{date}_threshold{threshold}.csv"
        for date in dates_list
    ]

    df_list = []
    for file_name in target_files:
        fpath = os.path.join(symbol_folder, file_name)
        if os.path.exists(fpath):
            df_to_merge = pl.read_csv(fpath)
            print(f"reading files: {fpath}")
            casted_df = df_to_merge.with_columns([
                pl.col(col).cast(
                    pl.Int64 if col == "timestamp" else
                    pl.Float64
                ) for col in df_to_merge.columns
            ])
            df_list.append(casted_df)
        else:
            print(f"文件不存在，跳过：{fpath}")

    if not df_list:
        raise FileNotFoundError("未找到任何匹配的 CSV 文件。")

    merged_df = pl.concat(df_list).sort("timestamp")

    # 保存
    output_filename = f"{symbol}_merged_thr{threshold}_roll{rolling_window}.csv"
    output_path = os.path.join(output_dir, output_filename)
    merged_df.write_csv(output_path)
    print(f"{symbol} 合并完成，共 {merged_df.shape[0]} 行。保存至：{output_path}")
    return output_path


if __name__ == "__main__":
    from ppo_algo.datas.high_frequency_data_parser import ParseHFTData
    import time

    # instruments = ["ETHUSDT", "SOLUSDT", "DOGEUSDT", "FILUSDT", "GUNUSDT", "JASMYUSDT"]
    instruments = ["BTCUSDT"]
    output_directory = "C:/quant/data/binance_resampled_data"

    print("start", time.strftime("%Y-%m-%d %H:%M:%S"))
    process_data_by_day_with_multiple_pairs(
        start_date="2025_04_07",
        end_date="2025_06_10",
        threshold=0.001,
        rolling_window=200,
        output_dir=output_directory,
        target_instruments=instruments,
        resample=True,
    )
    print("task 1 finished", time.strftime("%Y-%m-%d %H:%M:%S"))
    #
    # process_data_by_day_with_multiple_pairs(
    #     start_date="2025_04_07",
    #     end_date="2025_06_10",
    #     threshold=0.002,
    #     rolling_window=200,
    #     output_dir=output_directory,
    #     target_instruments=instruments,
    #     resample=True,
    # )
    # print("task 2 finished", time.strftime("%Y-%m-%d %H:%M:%S"))
    #
    # process_data_by_day_with_multiple_pairs(
    #     start_date="2025_04_07",
    #     end_date="2025_06_10",
    #     threshold=0.005,
    #     rolling_window=100,
    #     output_dir=output_directory,
    #     target_instruments=instruments,
    #     resample=True,
    # )
    # print("task 3 start", time.strftime("%Y-%m-%d %H:%M:%S"))


