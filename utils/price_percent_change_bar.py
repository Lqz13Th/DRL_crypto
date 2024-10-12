import pandas as pd
import numpy as np
from tqdm import tqdm


def scaled_sigmoid(x: float, start: float, end: float) -> float:
    """当`x`落在`[start,end]`区间时，函数值为[0,1]且在该区间有较好的响应灵敏度"""
    n = np.abs(start - end)
    score = 2 / (1 + np.exp(-np.log(40_000) * (x - start - n) / n + np.log(5e-3)))
    return score / 2


def rolling_scaled_sigmoid(x: float, mean: float, std: float) -> float:
    """使用Z-score标准化后进行scaled sigmoid变换"""
    if std == 0:  # 防止除以零
        return scaled_sigmoid(x, mean, mean + 1e-10)  # 如果标准差为0，则返回相对值

    z_score = (x - mean) / std
    return scaled_sigmoid(z_score, -1, 1)  # 将Z-score映射到[-1, 1]区间


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    df_normalized = df.copy()

    df_normalized['scaled_sigmoid_price'] = df_normalized['price'].apply(
        lambda x: scaled_sigmoid(x, 0, 10)
    )
    df_normalized['scaled_sigmoid_sum_buy_size'] = df_normalized['sum_buy_size'].apply(
        lambda x: scaled_sigmoid(x, 0, 100_000)
    )
    df_normalized['scaled_sigmoid_sum_sell_size'] = df_normalized['sum_sell_size'].apply(
        lambda x: scaled_sigmoid(x, 0, 100_000)
    )
    df_normalized['scaled_sigmoid_timestamp_duration'] = df_normalized['timestamp_duration'].apply(
        lambda x: scaled_sigmoid(x, 0, 600_000)
    )
    df_normalized['scaled_sigmoid_price_pct_change'] = df_normalized['price_pct_change'].apply(
        lambda x: scaled_sigmoid(x, -1., 1.)
    )
    df_normalized['scaled_sigmoid_buy_sell_imbalance'] = df_normalized['buy_sell_imbalance'].apply(
        lambda x: scaled_sigmoid(x, -100_000, 100_000)
    )
    return df_normalized


def rolling_normalize_data(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df_normalized = df.copy()

    for column in ['sum_buy_size', 'sum_sell_size', 'timestamp_duration', 'price_pct_change',
                   'buy_sell_imbalance', 'change_side']:
        rolling_mean = df_normalized[column].rolling(window=window, min_periods=1).mean()
        rolling_std = df_normalized[column].rolling(window=window, min_periods=1).std()

        # 使用滚动均值和标准差进行scaled sigmoid归一化
        df_normalized[f'scaled_{column}'] = df_normalized.apply(
            lambda row: rolling_scaled_sigmoid(row[column], rolling_mean[row.name], rolling_std[row.name]),
            axis=1
        )

    return df_normalized.iloc[:, 6:]


def generate_px_pct_bar(
        df: pd.DataFrame,
        threshold: float,
        window: int,
) -> pd.DataFrame:
    last_px = df.iloc[0]["price"]
    last_ts = df.iloc[0]["transact_time"]

    bars = []
    sum_buy_size = 0
    sum_sell_size = 0

    print(last_px)
    for i, row in tqdm(df.iterrows(), desc='Processing bars', total=len(df)):
        px = row["price"]
        sz = row["quantity"]
        ts = row["transact_time"]
        side = -1 if row["is_buyer_maker"] else 1  # 卖方主导为 -1，买方主导为 1

        px_pct = (px - last_px) / last_px

        if side == 1:
            sum_buy_size += sz

        else:
            sum_sell_size += sz

        if abs(px_pct) > threshold:
            ts_duration = ts - last_ts

            bar = {
                "price": px,
                "sum_buy_size": sum_buy_size,
                "sum_sell_size": sum_sell_size,
                "timestamp_duration": ts_duration,
                "price_pct_change": px_pct,
                'buy_sell_imbalance': sum_buy_size - sum_sell_size,
                "change_side": 1 if px_pct > 0 else 0,
            }
            bars.append(bar)

            last_px = px
            last_ts = ts
            sum_buy_size = 0
            sum_sell_size = 0

    bars_df = pd.DataFrame(bars)
    bars_df['future_price_pct_change'] = bars_df['price'].shift(-window) / bars_df['price'] - 1
    bars_df['scaled_sigmoid_future_price_pct_change'] = bars_df['future_price_pct_change'].apply(
        lambda x: scaled_sigmoid(x, -threshold * float(window), threshold * float(window))
    )
    bars_df = bars_df.dropna()
    return bars_df


if __name__ == "__main__":
    pd.set_option("display.max_rows", 5000)
    pd.set_option("expand_frame_repr", False)

    agg_trade_data = pd.read_csv("C:/Work Files/data/backtest/aggtrade/FILUSDT/FILUSDT-aggTrades-2024-04.csv")
    print(agg_trade_data)

    px_pct_bar = generate_px_pct_bar(
        df=agg_trade_data,
        threshold=0.001,
        window=3,
    )

    print(px_pct_bar)

    from sklearn.linear_model import Lasso
    from sklearn.model_selection import train_test_split

    X = px_pct_bar[[
        'price',
        'sum_buy_size',
        'sum_sell_size',
        'timestamp_duration',
        'timestamp_duration',
        'price_pct_change',
        'buy_sell_imbalance',
        'change_side'
    ]]

    y = px_pct_bar['future_price_pct_change']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)

    print("Lasso Coefficients:", lasso.coef_)
    print("Intercept:", lasso.intercept_)

    # 测试集上进行预测
    y_pred = lasso.predict(X_test)

