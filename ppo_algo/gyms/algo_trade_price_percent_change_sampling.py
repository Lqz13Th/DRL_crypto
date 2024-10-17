import numpy as np
import gymnasium as gym
import pandas as pd

from gymnasium import spaces

from engine.backtest.order_module import Order
from engine.backtest.backtest_module import BacktestEngine
from utils.price_percent_change_bar import rolling_normalize_data
from utils.price_percent_change_bar import scaled_sigmoid


class ObservationSpaceParser:
    def __init__(
            self,
            data_frame,
            rolling_window: int,
            rolling_normalize_window: int,
            single_token: str,
    ):
        self.single_token = single_token

        self.raw_df = data_frame

        self.price_threshold = 0.001
        # indexes utils
        self.raw_idx = 0
        self.raw_idx_max = data_frame.index.max()
        self.raw_idx_max_train = int(self.raw_idx_max * 0.2)

        self.features_update_idx = 0
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
        sz = self.raw_df.iloc[self.raw_idx]["amount"]
        ts = self.raw_df.iloc[self.raw_idx]["timestamp"]
        side = -1 if self.raw_df.iloc[self.raw_idx]["side"] == 'sell' else 1  # 卖方主导为 -1，买方主导为 1

        px_pct = (px - self.last_px) / self.last_px

        self.pub_trade = px
        if side == 1:
            self.sum_buy_size += sz
            self.ask_fake = px

        else:
            self.sum_sell_size += sz
            self.bid_fake = px

        if abs(px_pct) > self.price_threshold:
            self.features_update_idx = self.raw_idx

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

            bars_df = pd.DataFrame([bar])

            self.features_df = self.add_new_row(
                self.features_df,
                bars_df,
                self.rolling_window + self.rolling_normalize_window
            )

    def generate_px_pct_bar_on_reset(
            self,
    ) -> pd.DataFrame:
        self.last_px = self.raw_df.iloc[self.raw_idx]["price"]
        self.last_ts = self.raw_df.iloc[self.raw_idx]["amount"]

        bars = []

        print(self.last_px, self.raw_idx, self.raw_idx_max_train, self.raw_idx_max)
        features_idx = 0
        print(self.rolling_window, self.rolling_normalize_window, self.price_threshold)
        while features_idx < self.rolling_window + self.rolling_normalize_window:
            px = self.raw_df.iloc[self.raw_idx]["price"]
            sz = self.raw_df.iloc[self.raw_idx]["amount"]
            ts = self.raw_df.iloc[self.raw_idx]["amount"]
            side = -1 if self.raw_df.iloc[self.raw_idx]["side"] == 'sell' else 1  # 卖方主导为 -1，买方主导为 1

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
            single_token: str,
            numb_features=6,
            rolling_window=200,
            rolling_normalize_window=2000,
    ):
        super(PricePercentChangeSamplingEnv, self).__init__()
        self.op = ObservationSpaceParser(data_frame, rolling_window, rolling_normalize_window, single_token)
        self.be = BacktestEngine()
        self.be.token_default(single_token)

        self.raw_df = data_frame

        self.pnl_rate = 0
        self.pos_value_rate = 0
        self.current_pos_rate = 0
        self.price = self.op.pub_trade

        assert all(col in self.raw_df.columns for col in [
            'price',
            'quantity',
            'transact_time',
            'is_buyer_maker',
        ]), "DataFrame lack of columns"

        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # multi modal | tensorboard --logdir=examples/train/TPPCS_tensorboard/
        self.observation_space = spaces.Dict({
            "pub_price": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "features": spaces.Box(low=0, high=1, shape=(rolling_window, numb_features), dtype=np.float32),
            "account_states": spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
        })

    def reset(self, seed=None, options=None):  # 添加 seed 参数
        super().reset(seed=seed)
        self.op.raw_idx = 0
        self.pnl_rate = 0
        self.pos_value_rate = 0
        self.current_pos_rate = 0
        self.op.features_update_idx = 0

        self.be.token_default(self.op.single_token)
        
        self.op.features_df = self.op.generate_px_pct_bar_on_reset()
        print(self.op.features_df)

        normalized_df = rolling_normalize_data(
            self.op.features_df,
            window=self.op.rolling_normalize_window,
        ).tail(self.op.rolling_window).reset_index(drop=True)

        print(normalized_df)
        print(len(self.op.features_df), len(normalized_df), self.op.rolling_window)

        init_obs = {
            "pub_price": np.array(
                [self.op.bid_fake, self.op.ask_fake, self.op.pub_trade],
                dtype=np.float64,
            ),
            "features": normalized_df.values.astype(np.float32),
            "account_states": np.array(
                [
                    scaled_sigmoid(self.pnl_rate, -0.1, 0.1),
                    scaled_sigmoid(self.pos_value_rate, -1, 1),
                    scaled_sigmoid(self.current_pos_rate, -0.5, 0.5),
                    scaled_sigmoid(self.be.side[self.op.single_token], -1, 1),
                ],
                dtype=np.float64,
            ),
        }
        if options == "live":
            pass

        return init_obs, {}

    def step(self, action_state, options=None):
        trade_signal = self._take_action(action_state)

        reward = self._calculate_reward()

        terminated = False
        if options == "backtest":
            if self.op.raw_idx >= self.op.raw_idx_max:
                terminated = True
                print("dddddddddddddddddddoneeeeeeee", self.be.eval.funds)
                self.op.raw_idx = 0

        elif self.op.raw_idx >= self.op.raw_idx_max_train:
            terminated = True
            self.op.raw_idx = 0

        truncated = False
        self.op.generate_px_pct_bar_on_step()

        next_obs = self._next_observation()
        infos = {
            "price": self.op.pub_trade,
            "trade_signal": trade_signal,
        }

        self.price = self.op.pub_trade
        # print(next_obs)
        return next_obs, reward, terminated, truncated, infos

    def _next_observation(self):

        normalized_df = rolling_normalize_data(
            self.op.features_df,
            window=self.op.rolling_normalize_window,
        ).tail(self.op.rolling_window).reset_index(drop=True)

        next_obs = {
            "pub_price": np.array(
                [self.op.bid_fake, self.op.ask_fake, self.op.pub_trade],
                dtype=np.float64,
            ),
            "features": normalized_df.values.astype(np.float32),
            "account_states": np.array(
                [
                    scaled_sigmoid(self.pnl_rate, -0.1, 0.1),
                    scaled_sigmoid(self.pos_value_rate, -1, 1),
                    scaled_sigmoid(self.current_pos_rate, -0.5, 0.5),
                    scaled_sigmoid(self.be.side[self.op.single_token], -1, 1),
                ],
                dtype=np.float64,
            ),
        }

        return next_obs

    def _take_action(self, action):
        trade_signal = 0
        print(
            self.op.raw_idx,
            action[0],
            self.be.side[self.op.single_token],
            self.be.eval.funds,
            self.be.eval.cumulative_pnl,
            self.be.average_price[self.op.single_token],
            self.be.position[self.op.single_token],
            self.be.eval.total_position_value
        )
        if self.op.raw_idx == self.op.features_update_idx:
            self.be.update_pos_value()

        match action[0]:
            case action if action > 0.55:
                if self.be.side[self.op.single_token] == -1 and 0.99999 > action > 0.99:
                    order = Order(side=1, price=self.op.pub_trade, size=50, order_type="market")
                    self.be.check_orders(price=self.op.pub_trade, token=self.op.single_token, orders=[order])
                    trade_signal = 1
                    print(
                        self.op.raw_idx,
                        "buy",
                        self.price,
                        self.op.pub_trade,
                        self.be.eval.funds,
                        self.be.eval.cumulative_pnl,
                        self.be.average_price[self.op.single_token],
                        self.be.position[self.op.single_token],
                        self.be.eval.total_position_value
                    )

                elif self.op.raw_idx == self.op.features_update_idx and action < 0.99999:
                    order = Order(side=1, price=self.op.pub_trade, size=50, order_type="market")
                    self.be.check_orders(price=self.op.pub_trade, token=self.op.single_token, orders=[order])
                    trade_signal = 1
                    print(
                        self.op.raw_idx,
                        "buy==",
                        self.price,
                        self.op.pub_trade,
                        self.be.eval.funds,
                        self.be.eval.cumulative_pnl,
                        self.be.average_price[self.op.single_token],
                        self.be.position[self.op.single_token],
                        self.be.eval.total_position_value
                    )

            case action if action < 0.45:
                if self.be.side[self.op.single_token] == 1 and 0.00001 < action < 0.01:
                    order = Order(side=-1, price=self.op.pub_trade, size=50, order_type="market")
                    self.be.check_orders(price=self.op.pub_trade, token=self.op.single_token, orders=[order])
                    trade_signal = -1
                    print(
                        self.op.raw_idx,
                        "sell",
                        self.price,
                        self.op.pub_trade,
                        self.be.eval.funds,
                        self.be.eval.cumulative_pnl,
                        self.be.average_price[self.op.single_token],
                        self.be.position[self.op.single_token],
                        self.be.eval.total_position_value
                    )

                elif self.op.raw_idx == self.op.features_update_idx and action > 0.00001:
                    order = Order(side=-1, price=self.op.pub_trade, size=50, order_type="market")
                    self.be.check_orders(price=self.op.pub_trade, token=self.op.single_token, orders=[order])
                    trade_signal = -1
                    print(
                        self.op.raw_idx,
                        "sell==",
                        self.price,
                        self.op.pub_trade,
                        self.be.eval.funds,
                        self.be.eval.cumulative_pnl,
                        self.be.average_price[self.op.single_token],
                        self.be.position[self.op.single_token],
                        self.be.eval.total_position_value
                    )

            case _:
                self.be.check_orders(price=self.op.pub_trade, token=self.op.single_token)

        return trade_signal

    def _calculate_reward(self):
        pnl = self.be.eval.cumulative_pnl
        pos_value = self.be.eval.total_position_value
        avg_pos_price = self.be.average_price[self.op.single_token]
        current_price = self.op.pub_trade

        pnl_rate = pnl / self.be.eval.funds
        pos_value_rate = pos_value / self.be.eval.funds
        abs_pos_value_rate = abs(pos_value_rate)

        current_pos_rate = 0

        match self.be.side[self.op.single_token]:
            case 1:
                current_pos_rate = (current_price - avg_pos_price) / avg_pos_price

            case -1:
                current_pos_rate = (avg_pos_price - current_price) / avg_pos_price

            case _:
                pass

        self.pnl_rate = pnl_rate
        self.pos_value_rate = pos_value_rate
        self.current_pos_rate = current_pos_rate

        addition_reward = 0.2 * abs_pos_value_rate if 0.2 > abs_pos_value_rate >= 0.001 else 0
        penalty = -abs_pos_value_rate if abs_pos_value_rate > 0.6 else 0

        # [5790736 rows x 4 columns]
        # Using cpu device
        # 9.972 0 1158147 5790735
        # 200 2000 0.001
        reward = 5 * current_pos_rate - 0.2 * abs_pos_value_rate + 0.1 * pnl_rate + addition_reward + 0.5 * penalty
        return scaled_sigmoid(reward, -2, 2) - 0.5

