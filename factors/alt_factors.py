# OI factors
def factor_oi_ratio(df):
    return df['open_interest_data_sumOpenInterest'] / df['open_interest_data_sumOpenInterest'].rolling(3).mean()

def factor_oi_change_sum(df, window=12):
    return df['open_interest_data_sumOpenInterest'].pct_change(fill_method=None).rolling(window).sum()

def factor_oi_value_change(df):
    return df['open_interest_data_sumOpenInterestValue'].pct_change(fill_method=None)

def factor_short_term_volatility(df, window=12):
    # 使用过去12个5分钟的收益标准差来衡量短期波动性
    return df['open_interest_data_sumOpenInterest'].pct_change(fill_method=None).rolling(window).std()

def factor_long_term_volatility(df, window=72):  # 72个5分钟等于6小时
    # 使用过去72个5分钟的收益标准差来衡量长期波动性
    return df['open_interest_data_sumOpenInterest'].pct_change(fill_method=None).rolling(window).std()

def factor_short_term_oi_trend(df, window=12):
    return df['open_interest_data_sumOpenInterest'].rolling(window).mean()

def factor_long_term_oi_trend(df, window=72):
    return df['open_interest_data_sumOpenInterest'].rolling(window).mean()

# LS ratio
def factor_top_lsr_diff(df):
    return df['top_long_short_position_ratio_data_longAccount'] - df['top_long_short_position_ratio_data_shortAccount']

def factor_top_lsr_ratio(df):
    return df['top_long_short_position_ratio_data_longShortRatio']

def factor_account_lsr_diff(df):
    return df['top_long_short_account_ratio_data_longAccount'] - df['top_long_short_account_ratio_data_shortAccount']

def factor_account_lsr_ratio(df):
    return df['top_long_short_account_ratio_data_longShortRatio']

def factor_lsr_ratio_diff(df):
    return df['long_short_ratio_data_longAccount'] - df['long_short_ratio_data_shortAccount']

def factor_lsr_ratio(df):
    return df['long_short_ratio_data_longShortRatio']


# BS vol
def factor_buy_sell_ratio(df):
    return df['trade_taker_long_short_ratio_data_buySellRatio']

def factor_buy_sell_vol_diff(df):
    return df['trade_taker_long_short_ratio_data_buyVol'] - df['trade_taker_long_short_ratio_data_sellVol']

def factor_buy_vol_pct_change(df):
    return df['trade_taker_long_short_ratio_data_buyVol'].pct_change(fill_method=None)

def factor_sell_vol_pct_change(df):
    return df['trade_taker_long_short_ratio_data_sellVol'].pct_change(fill_method=None)


# Sentiment
def factor_bullish_sentiment(df):
    return (
        df['top_long_short_account_ratio_data_longAccount']
        + df['top_long_short_position_ratio_data_longAccount']
        + df['long_short_ratio_data_longAccount']
    ) / 3

def factor_bearish_sentiment(df):
    return (
        df['top_long_short_account_ratio_data_shortAccount']
        + df['top_long_short_position_ratio_data_shortAccount']
        + df['long_short_ratio_data_shortAccount']
    ) / 3

def factor_net_sentiment_score(df):
    return factor_bullish_sentiment(df) - factor_bearish_sentiment(df)

def factor_net_sentiment_5min(df):
    bullish = factor_bullish_sentiment(df)
    bearish = factor_bearish_sentiment(df)
    return bullish - bearish

def factor_short_term_sentiment_volatility(df, window=12):
    sentiment = factor_net_sentiment_5min(df)
    return sentiment.rolling(window).std()

def factor_sentiment_net(df):
    bull = (
        df['top_long_short_account_ratio_data_longAccount']
        + df['top_long_short_position_ratio_data_longAccount']
        + df['long_short_ratio_data_longAccount']
    ) / 3
    bear = (
        df['top_long_short_account_ratio_data_shortAccount']
        + df['top_long_short_position_ratio_data_shortAccount']
        + df['long_short_ratio_data_shortAccount']
    ) / 3
    return bull - bear  # >0 多头占优，<0 空头占优


# Momentum
def factor_oi_momentum_3(df):
    return df['open_interest_data_sumOpenInterest'].diff(3)

def factor_buy_vol_momentum_3(df):
    return df['trade_taker_long_short_ratio_data_buyVol'].diff(3)

def factor_sell_vol_momentum_3(df):
    return df['trade_taker_long_short_ratio_data_sellVol'].diff(3)

def factor_flow_rate(df):
    return (df['trade_taker_long_short_ratio_data_buyVol'] - df['trade_taker_long_short_ratio_data_sellVol']) / df['open_interest_data_sumOpenInterest']

def factor_liquidity_shock(df):
    return (df['trade_taker_long_short_ratio_data_buyVol'] + df['trade_taker_long_short_ratio_data_sellVol']) / df['open_interest_data_sumOpenInterest'].shift(1)

def factor_5min_volatility(df):
    return df['open_interest_data_sumOpenInterest'].pct_change().rolling(12).std()  # 12期表示60分钟

def factor_volatility_volume_ratio(df):
    return factor_5min_volatility(df) / df['trade_taker_long_short_ratio_data_buySellRatio'].rolling(12).mean()

def factor_price_volume_deviation(df):
    return (df['open_interest_data_sumOpenInterest'] - df['trade_taker_long_short_ratio_data_buyVol']) / df['trade_taker_long_short_ratio_data_sellVol']

def factor_market_imbalance(df):
    buy_sell_diff = df['trade_taker_long_short_ratio_data_buyVol'] - df['trade_taker_long_short_ratio_data_sellVol']
    return buy_sell_diff / (df['trade_taker_long_short_ratio_data_buyVol'] + df['trade_taker_long_short_ratio_data_sellVol'])

def factor_volume_delta(df):
    return df['trade_taker_long_short_ratio_data_buyVol'] - df['trade_taker_long_short_ratio_data_sellVol']

def factor_volatility_ratio(df):
    short_term_vol = factor_short_term_volatility(df)
    long_term_vol = factor_long_term_volatility(df)
    return short_term_vol / long_term_vol

def factor_volatility_trend_change(df):
    short_term_vol = factor_short_term_volatility(df)
    long_term_vol = factor_long_term_volatility(df)
    return short_term_vol.diff() - long_term_vol.diff()

def factor_short_term_holding_time(df):
    volatility_ratio = factor_volatility_ratio(df)
    if volatility_ratio > 1.5:
        return 'hold_for_short_term'  # 强烈震荡市场，短期持仓
    elif volatility_ratio < 0.8:
        return 'hold_for_long_term'  # 长期趋势市场，持仓时间较长
    else:
        return 'hold_for_medium_term'  # 中期震荡市场，适中持仓

def factor_dynamic_position_adjustment(df):
    volatility_trend = factor_volatility_trend_change(df)
    if volatility_trend > 0:  # 短期波动性上升，市场可能进入震荡阶段
        return 'adjust_to_short_term'
    elif volatility_trend < 0:  # 长期波动性上升，市场可能形成趋势
        return 'adjust_to_long_term'
    else:
        return 'maintain_current_position'  # 持仓状态不变


def apply_all_factors(df):
    df = df.copy()

    df['factor_oi_ratio'] = factor_oi_ratio(df)
    df['factor_oi_change'] = factor_oi_change_sum(df)
    df['factor_oi_value_change'] = factor_oi_value_change(df)

    df['factor_top_lsr_diff'] = factor_top_lsr_diff(df)
    df['factor_top_lsr_ratio'] = factor_top_lsr_ratio(df)
    df['factor_account_lsr_diff'] = factor_account_lsr_diff(df)
    df['factor_account_lsr_ratio'] = factor_account_lsr_ratio(df)
    df['factor_lsr_ratio_diff'] = factor_lsr_ratio_diff(df)
    df['factor_lsr_ratio'] = factor_lsr_ratio(df)

    df['factor_buy_sell_ratio'] = factor_buy_sell_ratio(df)
    df['factor_buy_sell_vol_diff'] = factor_buy_sell_vol_diff(df)
    df['factor_buy_vol_pct_change'] = factor_buy_vol_pct_change(df)
    df['factor_sell_vol_pct_change'] = factor_sell_vol_pct_change(df)

    df['factor_bullish_sentiment'] = factor_bullish_sentiment(df)
    df['factor_bearish_sentiment'] = factor_bearish_sentiment(df)
    df['factor_net_sentiment_score'] = factor_net_sentiment_score(df)

    df['factor_oi_momentum_3'] = factor_oi_momentum_3(df)
    df['factor_buy_vol_momentum_3'] = factor_buy_vol_momentum_3(df)
    df['factor_sell_vol_momentum_3'] = factor_sell_vol_momentum_3(df)

    return df
