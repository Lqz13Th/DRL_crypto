import polars as pl


class ParseHFTData:
    def __init__(self):
        pass

    @staticmethod
    def parse_agg_trade_data_binance(file_path: str) -> pl.DataFrame:
        df_agg_trade = pl.scan_csv(
            file_path,
            infer_schema_length=10000,
        )
        selected_columns = df_agg_trade.select(['price', 'quantity', 'transact_time', 'is_buyer_maker'])
        return selected_columns.collect()

    @staticmethod
    def parse_trade_data_tardis(file_path: str) -> pl.DataFrame:
        df_trade = pl.scan_csv(
            file_path,
            infer_schema_length=10000,
            schema_overrides={
                "timestamp": pl.Int64,
                "price": pl.Float64,
                "amount": pl.Float64,
                "side": pl.Utf8,
            }
        )

        select_cols = ["timestamp", "price", "amount", "side"]
        df_trade = df_trade.select(select_cols)
        return df_trade.collect()

    @staticmethod
    def parse_lob_data_tardis(file_path: str) -> pl.DataFrame:
        columns_to_exclude = ['exchange', 'symbol', 'local_timestamp']

        tmp_df = pl.scan_csv(file_path, infer_schema_length=10000)
        col_names = [col for col in tmp_df.collect_schema().names() if col != "" and col not in columns_to_exclude]

        schema_overrides = {
            col: (
                pl.Int64 if col == "timestamp"
                else pl.Utf8 if col in {"symbol", "side"}
                else pl.Float64
            ) for col in col_names
        }

        df_lob = pl.scan_csv(
            file_path,
            infer_schema_length=10000,
            schema_overrides=schema_overrides,
        ).select(col_names)

        df_lob = df_lob.with_columns([
            pl.col(col).cast(dtype) for col, dtype in schema_overrides.items()
        ])

        return df_lob.collect()

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

