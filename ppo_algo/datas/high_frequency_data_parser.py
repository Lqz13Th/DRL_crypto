import polars as pl


class ParseHFTData:
    def __init__(self):
        pass

    @staticmethod
    def parse_agg_trade_data_binance(file_path: str,) -> pl.DataFrame:
        df_agg_trade = pl.read_csv(file_path)
        selected_columns = df_agg_trade[['price', 'quantity', 'transact_time', 'is_buyer_maker']]
        return selected_columns

    @staticmethod
    def parse_trade_data_tardis(file_path: str, ) -> pl.DataFrame:
        df_trade = pl.read_csv(file_path)
        selected_columns = df_trade[["timestamp", "price", "amount", "side"]]
        return selected_columns


if __name__ == '__main__':
    psd = ParseHFTData()
    df = psd.parse_trade_data_tardis(
        "C:/Users/trade/PycharmProjects/DataGrabber/datasets/binance-futures_trades_2024-08-05_FILUSDT.csv.gz"
    )

    print(df)

