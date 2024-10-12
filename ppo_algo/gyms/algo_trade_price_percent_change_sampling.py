import numpy as np
import gymnasium as gym
import pandas as pd

from gymnasium import spaces

from utils.price_percent_change_bar import rolling_normalize_data


class ObservationSpaceParser:
    def __init__(
            self,
            data_frame,
            rolling_window: int,
            rolling_normalize_window: int,
    ):
        self.raw_df = data_frame

        self.price_threshold = 0.001
        # indexes utils
        self.raw_idx = 0
        self.raw_idx_max = data_frame.index.max()
        self.raw_idx_max_train = self.raw_idx_max * 0.7
        # ===================================================================
        # generate from histo data or fetch from live data
        self.pub_trade = 0
        self.ask_fake = 0
        self.bid_fake = 0
        # ===================================================================
        self.rolling_window = rolling_window
        self.rolling_normalize_window = rolling_normalize_window

        self.features_df = pd.DataFrame()

        # for price percent change bar
        self.last_px = 0
        self.last_ts = 0
        self.sum_buy_size = 0
        self.sum_sell_size = 0

        self.enter_px = 0

    def generate_px_pct_bar_on_step(
            self,
    ):
        self.raw_idx += 1
        px = self.raw_df.iloc[self.raw_idx]["price"]
        sz = self.raw_df.iloc[self.raw_idx]["quantity"]
        ts = self.raw_df.iloc[self.raw_idx]["transact_time"]
        side = -1 if self.raw_df.iloc[self.raw_idx]["is_buyer_maker"] else 1  # 卖方主导为 -1，买方主导为 1

        px_pct = (px - self.last_px) / self.last_px

        self.pub_trade = px
        if side == 1:
            self.sum_buy_size += sz
            self.ask_fake = px

        else:
            self.sum_sell_size += sz
            self.bid_fake = px

        if abs(px_pct) > self.price_threshold:
            ts_duration = ts - self.last_ts

            bar = {
                "sum_buy_size": self.sum_buy_size,
                "sum_sell_size": self.sum_sell_size,
                "timestamp_duration": ts_duration,
                "price_pct_change": px_pct,
                'buy_sell_imbalance': self.sum_buy_size - self.sum_sell_size,
                "change_side": 1 if px_pct > 0 else 0,
            }

            self.last_px = px
            self.last_ts = ts
            self.sum_buy_size = 0
            self.sum_sell_size = 0

            bars_df = pd.DataFrame(bar)

            self.features_df = self.add_new_row(
                self.features_df,
                bars_df,
                self.rolling_window + self.rolling_normalize_window
            )

    def generate_px_pct_bar_on_reset(
            self,
    ) -> pd.DataFrame:
        self.last_px = self.raw_df.iloc[self.raw_idx]["price"]
        self.last_ts = self.raw_df.iloc[self.raw_idx]["transact_time"]

        bars = []

        print(self.last_px)
        features_idx = 0
        print(self.rolling_window, self.rolling_normalize_window, self.price_threshold)
        while features_idx < self.rolling_window + self.rolling_normalize_window:
            px = self.raw_df.iloc[self.raw_idx]["price"]
            sz = self.raw_df.iloc[self.raw_idx]["quantity"]
            ts = self.raw_df.iloc[self.raw_idx]["transact_time"]
            side = -1 if self.raw_df.iloc[self.raw_idx]["is_buyer_maker"] else 1  # 卖方主导为 -1，买方主导为 1

            self.raw_idx += 1

            px_pct = (px - self.last_px) / self.last_px
            self.pub_trade = px
            if side == 1:
                self.sum_buy_size += sz
                self.ask_fake = px

            else:
                self.sum_sell_size += sz
                self.bid_fake = px

            if abs(px_pct) > self.price_threshold:
                ts_duration = ts - self.last_ts

                bar = {
                    "sum_buy_size": self.sum_buy_size,
                    "sum_sell_size": self.sum_sell_size,
                    "timestamp_duration": ts_duration,
                    "price_pct_change": px_pct,
                    'buy_sell_imbalance': self.sum_buy_size - self.sum_sell_size,
                    "change_side": 1 if px_pct > 0 else 0,
                }
                bars.append(bar)

                self.last_px = px
                self.last_ts = ts
                self.sum_buy_size = 0
                self.sum_sell_size = 0

                features_idx += 1

        bars_df = pd.DataFrame(bars)
        print(bars_df)

        bars_df = bars_df.dropna()
        print(bars_df)
        return bars_df

    @staticmethod
    def add_new_row(df, new_row, max_len):
        df = pd.concat([df, new_row], ignore_index=True)

        if len(df) > max_len:
            df.drop(df.index[0], inplace=True)

        return df


class PricePercentChangeSamplingEnv(gym.Env):
    def __init__(
            self,
            data_frame,
            numb_features=6,
            rolling_window=10,
            rolling_normalize_window=200,
    ):
        super(PricePercentChangeSamplingEnv, self).__init__()
        self.op = ObservationSpaceParser(data_frame, rolling_window, rolling_normalize_window)

        self.raw_df = data_frame
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.position = 0  # 持仓状态：0=空仓，1=持有

        assert all(col in self.raw_df.columns for col in [
            'price',
            'quantity',
            'transact_time',
            'is_buyer_maker',
        ]), "DataFrame lack of columns"

        self.action_space = spaces.Discrete(3)

        # multi modal tensorboard --logdir=examples/train/TPPCS_tensorboard/
        self.observation_space = spaces.Dict({
            "pub_price": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "features": spaces.Box(low=0, high=1, shape=(rolling_window, numb_features), dtype=np.float32),
            "account_states": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64), # to_do
        })

    def reset(self, seed=None, options=None):  # 添加 seed 参数
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.position = 0

        self.op.features_df = self.op.generate_px_pct_bar_on_reset()
        print(self.op.features_df)

        normalized_df = rolling_normalize_data(
            self.op.features_df,
            window=self.op.rolling_normalize_window,
        ).tail(self.op.rolling_window).reset_index(drop=True)

        print(normalized_df)
        print(len(self.op.features_df), len(normalized_df), self.op.rolling_window)

        avg_px_range = 0 if self.op.enter_px == 0 else (self.op.pub_trade - self.op.enter_px) / self.op.enter_px
        init_obs = {
            "pub_price": np.array(
                [self.op.bid_fake, self.op.ask_fake, self.op.pub_trade],
                dtype=np.float64,
            ),
            "features": normalized_df.values.astype(np.float32),
            "account_states": avg_px_range
        }
        if options == "live":
            pass

        return init_obs, {}

    def step(self, action_state, options=None):
        trade_signal = self._take_action(action_state)

        reward = self._calculate_reward()
        if options == "backtest":
            terminated = self.op.raw_idx >= self.op.raw_idx_max

        else:
            terminated = self.op.raw_idx >= self.op.raw_idx_max_train

        truncated = False
        next_obs = self._next_observation()

        return next_obs, reward, terminated, truncated, {"trade_signal": trade_signal}

    def _next_observation(self):
        normalized_df = rolling_normalize_data(
            self.op.features_df,
            window=self.op.rolling_normalize_window,
        ).tail(self.op.rolling_window).reset_index(drop=True)

        avg_px_range = 0 if self.op.enter_px == 0 else (self.op.pub_trade - self.op.enter_px) / self.op.enter_px
        next_obs = {
            "pub_price": np.array(
                [self.op.bid_fake, self.op.ask_fake, self.op.pub_trade],
                dtype=np.float64,
            ),
            "features": normalized_df.values.astype(np.float32),
            "account_states": avg_px_range
        }

        return next_obs

    def _take_action(self, action):
        trade_signal = 0
        if action == 1:  # 买入
            if self.position == 0:
                self.position = self.balance / self.op.ask_fake
                self.balance = 0
                self.op.enter_px = self.op.ask_fake
                trade_signal = 1

        elif action == 2:  # 卖出
            if self.position > 0:
                self.balance = self.position * self.op.bid_fake * 0.999
                self.position = 0
                self.op.enter_px = 0
                trade_signal = -1

        return trade_signal

    def _calculate_reward(self):
        current_price = self.op.pub_trade
        total_value = self.balance + self.position * current_price
        reward = total_value - self.initial_balance
        return reward

