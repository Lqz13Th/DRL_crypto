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
        self.features_idx = 0
        self.raw_idx_max = len(data_frame) - 1
        self.raw_idx_max_train = self.raw_idx_max * 0.7
        # ===================================================================
        # generate from histo data or fetch from live data
        self.pub_trade = 0
        self.ask_fake = 0
        self.bid_fake = 0
        # ===================================================================
        self.rolling_window = rolling_window
        self.rolling_normalize_window = rolling_normalize_window

        self.features_df = pd.DataFrame

    def generate_px_pct_bar_on_reset(
            self,
    ) -> pd.DataFrame:
        last_px = self.raw_df.iloc[self.raw_idx]["price"]
        last_ts = self.raw_df.iloc[self.raw_idx]["transact_time"]

        bars = []
        sum_buy_size = 0
        sum_sell_size = 0

        print(last_px)
        while self.features_idx > self.rolling_window + self.rolling_normalize_window:
            px = self.raw_df.iloc[self.raw_idx]["price"]["price"]
            sz = self.raw_df.iloc[self.raw_idx]["quantity"]
            ts = self.raw_df.iloc[self.raw_idx]["transact_time"]
            side = -1 if self.raw_df.iloc[self.raw_idx]["is_buyer_maker"] else 1  # 卖方主导为 -1，买方主导为 1

            self.raw_idx += 1

            px_pct = (px - last_px) / last_px

            if side == 1:
                sum_buy_size += sz
                self.ask_fake = px

            else:
                sum_sell_size += sz
                self.bid_fake = px

            if abs(px_pct) > self.price_threshold:
                ts_duration = ts - last_ts

                bar = {
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

                self.features_idx += 1

        bars_df = pd.DataFrame(bars)
        bars_df = bars_df.dropna()
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
        self.current_step = 0
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

        # multi modal
        self.observation_space = spaces.Dict({
            "pub_price": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "features": spaces.Box(low=0, high=1, shape=(numb_features, rolling_window), dtype=np.float32),
            # "account_states": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64), # to_do
        })

    def reset(self, seed=None, options=None):  # 添加 seed 参数
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0

        price_percent_change_df = self.op.generate_px_pct_bar_on_reset()
        self.op.features_df = rolling_normalize_data(
            price_percent_change_df,
            window=self.op.rolling_normalize_window,
        ).tail(self.op.rolling_window).reset_index(drop=True)

        print(self.op.features_df)
        print(len(self.op.features_df), self.op.rolling_window)
        obs = {
            "pub_price": np.array(
                [self.op.bid_fake, self.op.ask_fake, self.op.pub_trade],
                dtype=np.float64,
            ),
            "features": self.op.features_df.values.astype(np.float32)
        }
        if options == "live":
            pass

        return obs

    def step(self, action_state):
        self._take_action(action_state)
        self.current_step += 1

        reward = self._calculate_reward()
        terminated = self.current_step >= len(self.raw_df) - 1
        truncated = False
        current_obs = self._next_observation()
        return current_obs, reward, terminated, truncated, {}

    def _next_observation(self):
        observation_value = self.raw_df.iloc[self.current_step][
            ['price', 'quantity', 'transact_time', 'is_buyer_maker']
        ]

        return observation_value.values

    def _take_action(self, action):
        current_price = self.raw_df.loc[self.current_step, 'price']

        if action == 1:  # 买入
            if self.position == 0:
                self.position = self.balance / current_price
                self.balance = 0

        elif action == 2:  # 卖出
            if self.position > 0:
                self.balance = self.position * current_price
                self.position = 0

    def _calculate_reward(self):
        current_price = self.raw_df.loc[self.current_step, 'price']
        total_value = self.balance + self.position * current_price
        reward = total_value - self.initial_balance
        return reward

