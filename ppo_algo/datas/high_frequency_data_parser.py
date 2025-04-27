import polars as pl


class ParseHFTData:
    def __init__(self):
        pass

    @staticmethod
    def parse_agg_trade_data_binance(file_path: str) -> pl.DataFrame:
        df_agg_trade = pl.scan_csv(file_path)
        selected_columns = df_agg_trade.select(['price', 'quantity', 'transact_time', 'is_buyer_maker'])
        return selected_columns.collect()

    @staticmethod
    def parse_trade_data_tardis(file_path: str) -> pl.DataFrame:
        df_trade = pl.scan_csv(file_path)
        selected_columns = df_trade.select(["timestamp", "price", "amount", "side"])
        return selected_columns.collect()

    @staticmethod
    def parse_lob_data_tardis(file_path: str) -> pl.DataFrame:
        df_lob = pl.scan_csv(file_path)
        columns_to_exclude = ['exchange', 'symbol', 'local_timestamp']

        df_lob = df_lob.drop(columns_to_exclude)

        df_lob_collected = df_lob.collect()
        return df_lob_collected

    @staticmethod
    def parse_trade_data_list_path_tardis(file_paths: list) -> pl.DataFrame:
        lazy_dfs = [pl.scan_csv(file_path).select(["timestamp", "price", "amount", "side"]) for file_path in file_paths]

        combined_lazy_df = pl.concat(lazy_dfs)

        combined_df = combined_lazy_df.collect()

        return combined_df

    @staticmethod
    def parse_lob_data_list_path_tardis(file_paths: list) -> pl.DataFrame:
        lazy_dfs = [pl.scan_csv(file_path).drop(['exchange', 'symbol', 'local_timestamp']) for file_path in file_paths]

        combined_lazy_df = pl.concat(lazy_dfs)

        combined_df = combined_lazy_df.collect()

        return combined_df

if __name__ == '__main__':
    psd = ParseHFTData()
    df = psd.parse_trade_data_tardis(
        "C:/quant/data/tardis_data/datasets/binance-futures_trades_2025-03-05_BTCUSDT.csv.gz"
    )

    print(df)

