import polars as pl

# trade_df: 包含 timestamp 和 price
trade_data = {
    "timestamp": [1, 2, 3, 4, 5],
    "price": [100, 101, 102, 103, 104]
}
trade_df = pl.DataFrame(trade_data)

# lob_df: 包含 timestamp 和 volume
lob_data = {
    "timestamp": [2, 3, 5, 6],
    "volume": [10, 15, 20, 25]
}
lob_df = pl.DataFrame(lob_data)

print("trade_df:")
print(trade_df)

print("\nlob_df:")
print(lob_df)

# 合并这两个 DataFrame，使用 timestamp 列进行 full join
merged = trade_df.join(lob_df, on="timestamp", how="full")

# 查看合并后的结果
print("\nmerged:")
print(merged)

# 合并 timestamp 和 timestamp_right
merged = merged.with_columns(
    pl.when(pl.col("timestamp").is_null())
    .then(pl.col("timestamp_right"))
    .otherwise(pl.col("timestamp"))
    .alias("timestamp")
)

# 删除多余的 timestamp_right 列
merged = merged.drop("timestamp_right")

# 查看合并后的结果
print(merged)

