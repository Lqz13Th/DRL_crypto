import pandas as pd

from pathlib import Path
from tqdm import tqdm 

from factors.alt_factors import *
from utils.polars_expr import *

base_path = Path("C:/quant/data/binance_alt_data/alt_database/binance_futures_data/combined_alt_data/")

factor_output_path = Path("C:/quant/data/binance_alt_data/alt_database/binance_futures_data/alt_factors_data/")
factor_output_path.mkdir(parents=True, exist_ok=True)

def calculate_factors(df):
    df = df.copy()
    df['factor_short_term_oi_trend'] = factor_short_term_oi_trend(df)
    df['factor_long_term_oi_trend'] = factor_long_term_oi_trend(df)
    df['factor_buy_sell_vol_diff'] = factor_buy_sell_vol_diff(df)
    return df

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)

for csv_file in tqdm(base_path.glob("*_data.csv"), desc="Calculating factors"):
    try:
        df = pd.read_csv(csv_file)

        factor_df = calculate_factors(df)

        factor_pl_df = pl.from_pandas(factor_df)
        normalized_df = rolling_normalize_data(factor_pl_df, window=50).drop_nulls()
        output_file = factor_output_path / csv_file.name
        normalized_df.write_csv(output_file)

    except Exception as e:
        print(f"Processing files {csv_file.name} failure: {e}")
        break

print(f"finished: {factor_output_path}")


