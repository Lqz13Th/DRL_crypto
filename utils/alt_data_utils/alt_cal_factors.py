import pandas as pd

from pathlib import Path
from tqdm import tqdm 

from factors.alt_factors import *

base_path = Path("C:/quant/data/binance_alt_data/alt_database/binance_futures_data/combined_alt_data/")

factor_output_path = Path("C:/quant/data/binance_alt_data/alt_database/binance_futures_data/alt_factors_data/")
factor_output_path.mkdir(parents=True, exist_ok=True)



def rolling_zscore(series, window=72, clip_value=5):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std(ddof=0)
    z = (series - rolling_mean) / (rolling_std + 1e-9)  # 避免除以0
    return z.clip(-clip_value, clip_value)  # 限制极端值

def calculate_factors(alt_factor_df, zscore_window=72):
    alt_factor_df = alt_factor_df.copy()
    alt_factor_df['factor_oi_change'] = factor_oi_change(alt_factor_df)
    alt_factor_df['factor_short_term_oi_volatility'] = factor_short_term_volatility(alt_factor_df)
    alt_factor_df['factor_long_term_oi_volatility'] = factor_long_term_volatility(alt_factor_df)
    alt_factor_df['factor_short_term_oi_trend'] = factor_short_term_oi_trend(alt_factor_df)
    alt_factor_df['factor_long_term_oi_trend'] = factor_long_term_oi_trend(alt_factor_df)
    alt_factor_df['factor_buy_sell_vlm_diff'] = factor_buy_sell_vol_diff(alt_factor_df)
    alt_factor_df['factor_sentiment_net'] = factor_sentiment_net(alt_factor_df)

    factor_columns = [
        'factor_oi_change',
        'factor_short_term_oi_volatility',
        'factor_long_term_oi_volatility',
        'factor_short_term_oi_trend',
        'factor_long_term_oi_trend',
        'factor_buy_sell_vlm_diff',
        'factor_sentiment_net',
    ]

    # 滚动 z-score 归一化处理
    for col in factor_columns:
        alt_factor_df[f'z_{col}'] = rolling_zscore(alt_factor_df[col], window=zscore_window)

    return alt_factor_df

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)

for csv_file in tqdm(base_path.glob("*_data.csv"), desc="Calculating factors"):
    try:
        df = pd.read_csv(csv_file)

        factor_df = calculate_factors(df)

        output_file = factor_output_path / csv_file.name
        factor_df.to_csv(output_file)

    except Exception as e:
        print(f"Processing files {csv_file.name} failure: {e}")
        break

print(f"finished: {factor_output_path}")


