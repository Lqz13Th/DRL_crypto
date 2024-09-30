import pandas as pd


class ParseHFTData:
    def __init__(self):
        pass

    @staticmethod
    def parse_agg_trade_data_binance(file_path: str,) -> pd.DataFrame:
        df_candle = pd.read_csv(file_path)
        selected_columns = df_candle.iloc[:, 2:7]
        selected_columns.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return selected_columns


if __name__ == '__main__':
    pd.set_option("display.max_rows", 5000)
    pd.set_option("expand_frame_repr", False)

    psd = ParseHFTData()
    agg_trade_data = pd.read_csv("C:/Work Files/data/backtest/aggtrade/FILUSDT/FILUSDT-aggTrades-2024-04.csv")
    print(agg_trade_data)

