import numpy as np

def cal_imn_usdt(lev):
    return 200 * lev

def impact_price_pct_ask_row(row: dict, imn: float, levels: int = 2) -> float | None:
    cum_value = 0
    impact_price = row.get("lob_asks[0].price", None)

    for i in range(levels):
        price = row.get(f"lob_asks[{i}].price")
        amount = row.get(f"lob_asks[{i}].amount")
        if price is None or amount is None:
            break
        cum_value += price * amount
        impact_price = price
        if cum_value >= imn:
            break

    best_ask = row.get("lob_asks[0].price")
    if best_ask is None or best_ask <= 0:
        return None
    return (impact_price - best_ask) / best_ask


def impact_price_pct_bid_row(row: dict, imn: float, levels: int = 2) -> float | None:
    cum_value = 0
    impact_price = row.get("lob_bids[0].price", None)

    for i in range(levels):
        price = row.get(f"lob_bids[{i}].price")
        amount = row.get(f"lob_bids[{i}].amount")
        if price is None or amount is None:
            break
        cum_value += price * amount
        impact_price = price
        if cum_value >= imn:
            break

    best_bid = row.get("lob_bids[0].price")
    if best_bid is None or best_bid <= 0:
        return None
    return (best_bid - impact_price) / best_bid



def calculate_bid_amount_sum_row(row: dict, start_level: int = 1, end_level: int = 19) -> float | None:
    total_amount = 0.0

    for i in range(start_level, end_level):
        col_name = f"lob_bids[{i}].amount"
        amount = row.get(col_name)

        if amount is not None and not np.isnan(amount):
            total_amount += amount

    return total_amount

def calculate_ask_amount_sum_row(row: dict, start_level: int = 1, end_level: int = 19) -> float | None:
    total_amount = 0.0

    for i in range(start_level, end_level):
        col_name = f"lob_asks[{i}].amount"
        amount = row.get(col_name)

        if amount is not None and not np.isnan(amount):
            total_amount += amount

    return total_amount


import polars as pl

# 构造数据
df = pl.DataFrame({
    "lob_asks[0].price": [100, 101],
    "lob_asks[0].amount": [3, 5],
    "lob_asks[1].price": [102, 103],
    "lob_asks[1].amount": [5, 10],
    "lob_bids[0].price": [99, 100],
    "lob_bids[0].amount": [2, 4],
    "lob_bids[1].price": [98, 99],
    "lob_bids[1].amount": [10, 15],
})

# 使用单行 row + map 方式
imn = 50
levels = 2
df = df.with_columns([
    pl.Series(
        name=f"impact_price_pct_ask@{imn}",
        values=[impact_price_pct_ask_row(row, imn, levels) for row in df.iter_rows(named=True)]
    ),
    pl.Series(
        name=f"impact_price_pct_bid@{imn}",
        values=[impact_price_pct_bid_row(row, imn, levels) for row in df.iter_rows(named=True)]
    ),
])

print(df)
