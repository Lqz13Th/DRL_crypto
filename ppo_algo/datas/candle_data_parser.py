import polars as pl


class ParseCandleData:
    def __init__(self):
        pass

    @staticmethod
    def parse_candle_data_okx(file_path: str,) -> pl.DataFrame:
        df_candle = pl.read_csv(file_path)
        selected_columns = df_candle.iloc[:, 2:7]
        selected_columns.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return selected_columns


if __name__ == '__main__':
    psd = ParseCandleData()
    df = psd.parse_candle_data_okx('C:/Work Files/data/evaluation/candle/candle1m/FIL-USDT1min.csv')
    print(df)

