import pandas as pd


class ParseCandleData:
    def __init__(self):
        pass

    @staticmethod
    def parse_candle_data_okx(file_path: str,) -> pd.DataFrame:
        df_candle = pd.read_csv(file_path)
        selected_columns = df_candle.iloc[:, 2:7]
        selected_columns.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return selected_columns


if __name__ == '__main__':
    pd.set_option("display.max_rows", 5000)
    pd.set_option("expand_frame_repr", False)

    psd = ParseCandleData()
    df = psd.parse_candle_data_okx('C:/Work Files/data/backtest/candle/candle1m/FIL-USDT1min.csv')
    print(df)

