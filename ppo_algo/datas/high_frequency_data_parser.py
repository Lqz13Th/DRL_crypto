import pandas as pd


class ParseHFTData:
    def __init__(self):
        pass

    @staticmethod
    def parse_agg_trade_data_binance(file_path: str,) -> pd.DataFrame:
        df_agg_trade = pd.read_csv(file_path)
        selected_columns = df_agg_trade[['price', 'quantity', 'transact_time', 'is_buyer_maker']]
        return selected_columns


if __name__ == '__main__':
    pd.set_option("display.max_rows", 5000)
    pd.set_option("expand_frame_repr", False)

    psd = ParseHFTData()
    df = psd.parse_agg_trade_data_binance(
        "/home/pcone/drl_crypto/DRL_crypto/binance-futures_trades_2024-08-05_FILUSDT.csv.gz"
    )

    print(df)

