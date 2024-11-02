import numpy as np
import gymnasium as gym
import pandas as pd

from gymnasium import spaces

from engine.evaluation.order_module import Order
from engine.evaluation.match_module import MatchEngine
from utils.price_percent_change_bar import rolling_normalize_data
from utils.price_percent_change_bar import scaled_sigmoid


class PricePercentChangeSamplingEnv(gym.Env):
    def __init__(
            self,
            data_frame,
            single_token: str,
            mode='train',
            numb_features=6,
            rolling_window=20,
            rolling_normalize_window=200,
    ):
        super(PricePercentChangeSamplingEnv, self).__init__()
        self.op = ObservationSpaceParser(data_frame, rolling_window, rolling_normalize_window, single_token)
        self.me = MatchEngine(debug=True)
        self.me.token_default(single_token)

        self.raw_df = data_frame
        self.mode = mode

        self.pnl_rate = 0
        self.pos_value_rate = 0
        self.current_pos_rate = 0
        self.price = self.op.pub_trade

        assert all(col in self.raw_df.columns for col in [
            'price',
            'amount',
            'timestamp',
            'side',
        ]), "DataFrame lack of columns"

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # multi modal | tensorboard --logdir=examples/train/TPPCS_tensorboard/
        self.observation_space = spaces.Dict({
            "action_sig": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "features": spaces.Box(low=0, high=1, shape=(rolling_window, numb_features), dtype=np.float32),
            "account_states": spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
        })

    def reset(self, seed=None, options=None):  # 添加 seed 参数
        super().reset(seed=seed)
        self.op.raw_idx = self.op.raw_idx_max_train if self.mode == "evaluation" else 0
        self.pnl_rate = 0
        self.pos_value_rate = 0
        self.current_pos_rate = 0
        self.op.features_update_sig = 0

        self.me.evaluation_default()
        self.me.account_default()
        self.me.token_default(self.op.single_token)
        
        self.op.features_df = self.op.generate_px_pct_bar_on_reset()
        print(self.op.features_df)

        normalized_df = rolling_normalize_data(
            self.op.features_df,
            window=self.op.rolling_normalize_window,
        ).tail(self.op.rolling_window).reset_index(drop=True)

        print(normalized_df)
        print(len(self.op.features_df), len(normalized_df), self.op.rolling_window)

        init_obs = {
            "action_sig": 0,
            "features": normalized_df.values.astype(np.float32),
            "account_states": np.array(
                [
                    scaled_sigmoid(self.pnl_rate, -0.1, 0.1),
                    scaled_sigmoid(self.pos_value_rate, -1, 1),
                    scaled_sigmoid(self.current_pos_rate, -0.5, 0.5),
                    scaled_sigmoid(self.me.side[self.op.single_token], -1, 1),
                ],
                dtype=np.float64,
            ),
        }

        return init_obs, {}

    def step(self, action_state):
        self._take_action(action_state)

        reward = self._calculate_reward(action_state)

        terminated = False
        if self.mode == "evaluation":
            if self.op.raw_idx >= self.op.raw_idx_max:
                terminated = True
                print("done", self.me.funds)
                self.op.raw_idx = 0

        elif self.op.raw_idx >= self.op.raw_idx_max_train:
            terminated = True
            self.op.raw_idx = 0

        elif self.me.funds < self.me.init_funds * 0.1:
            terminated = True
            reward = - 0.5
            self.op.raw_idx = 0

        truncated = False
        self.op.generate_px_pct_bar_on_step()

        next_obs = self._next_observation()
        infos = {
            "price": self.op.pub_trade,
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
            "action_sig": self.op.features_update_sig,
            "features": normalized_df.values.astype(np.float32),
            "account_states": np.array(
                [
                    scaled_sigmoid(self.pnl_rate, -0.1, 0.1),
                    scaled_sigmoid(self.pos_value_rate, -1, 1),
                    scaled_sigmoid(self.current_pos_rate, -0.5, 0.5),
                    scaled_sigmoid(self.me.side[self.op.single_token], -1, 1),
                ],
                dtype=np.float64,
            ),
        }

        return next_obs

    def _take_action(self, action):
        print(
            self.op.raw_idx,
            action[0],
            self.me.side[self.op.single_token],
            self.me.funds,
            self.me.cumulative_pnl,
            self.me.average_price[self.op.single_token],
            self.price,
            self.me.position[self.op.single_token],
            self.me.cumulative_pos_value
        )
        if self.op.features_update_sig == 1:
            print('==================================================================================================')
            self.me.update_pos_value()

        match action[0]:
            case action if action >= 0.99:
                order = Order(side=1, price=self.op.pub_trade, size=500, order_type="market")
                self.me.check_orders(price=self.op.pub_trade, token=self.op.single_token, orders=[order])
                print(
                    self.op.raw_idx,
                    "buy5",
                    self.price,
                    self.op.pub_trade,
                    self.me.funds,
                    self.me.cumulative_pnl,
                    self.me.average_price[self.op.single_token],
                    self.me.position[self.op.single_token],
                    self.me.cumulative_pos_value
                )

            case action if 0.99 > action >= 0.8:
                order = Order(side=1, price=self.op.pub_trade, size=200, order_type="market")
                self.me.check_orders(price=self.op.pub_trade, token=self.op.single_token, orders=[order])
                print(
                    self.op.raw_idx,
                    "buy4",
                    self.price,
                    self.op.pub_trade,
                    self.me.funds,
                    self.me.cumulative_pnl,
                    self.me.average_price[self.op.single_token],
                    self.me.position[self.op.single_token],
                    self.me.cumulative_pos_value
                )

            case action if 0.8 > action >= 0.6:
                order = Order(side=1, price=self.op.pub_trade, size=100, order_type="market")
                self.me.check_orders(price=self.op.pub_trade, token=self.op.single_token, orders=[order])
                print(
                    self.op.raw_idx,
                    "buy3",
                    self.price,
                    self.op.pub_trade,
                    self.me.funds,
                    self.me.cumulative_pnl,
                    self.me.average_price[self.op.single_token],
                    self.me.position[self.op.single_token],
                    self.me.cumulative_pos_value
                )

            case action if 0.6 > action >= 0.4:
                order = Order(side=1, price=self.op.pub_trade, size=50, order_type="market")
                self.me.check_orders(price=self.op.pub_trade, token=self.op.single_token, orders=[order])
                print(
                    self.op.raw_idx,
                    "buy2",
                    self.price,
                    self.op.pub_trade,
                    self.me.funds,
                    self.me.cumulative_pnl,
                    self.me.average_price[self.op.single_token],
                    self.me.position[self.op.single_token],
                    self.me.cumulative_pos_value
                )

            case action if 0.4 > action >= 0.2:
                order = Order(side=1, price=self.op.pub_trade, size=10, order_type="market")
                self.me.check_orders(price=self.op.pub_trade, token=self.op.single_token, orders=[order])
                print(
                    self.op.raw_idx,
                    "buy1",
                    self.price,
                    self.op.pub_trade,
                    self.me.funds,
                    self.me.cumulative_pnl,
                    self.me.average_price[self.op.single_token],
                    self.me.position[self.op.single_token],
                    self.me.cumulative_pos_value
                )

            case action if action <= -0.99:
                order = Order(side=-1, price=self.op.pub_trade, size=500, order_type="market")
                self.me.check_orders(price=self.op.pub_trade, token=self.op.single_token, orders=[order])
                print(
                    self.op.raw_idx,
                    "sell5",
                    self.price,
                    self.op.pub_trade,
                    self.me.funds,
                    self.me.cumulative_pnl,
                    self.me.average_price[self.op.single_token],
                    self.me.position[self.op.single_token],
                    self.me.cumulative_pos_value
                )

            case action if -0.99 < action <= -0.8:
                order = Order(side=-1, price=self.op.pub_trade, size=200, order_type="market")
                self.me.check_orders(price=self.op.pub_trade, token=self.op.single_token, orders=[order])
                print(
                    self.op.raw_idx,
                    "sell4",
                    self.price,
                    self.op.pub_trade,
                    self.me.funds,
                    self.me.cumulative_pnl,
                    self.me.average_price[self.op.single_token],
                    self.me.position[self.op.single_token],
                    self.me.cumulative_pos_value
                )

            case action if -0.8 < action <= -0.6:
                order = Order(side=-1, price=self.op.pub_trade, size=100, order_type="market")
                self.me.check_orders(price=self.op.pub_trade, token=self.op.single_token, orders=[order])
                print(
                    self.op.raw_idx,
                    "sell3",
                    self.price,
                    self.op.pub_trade,
                    self.me.funds,
                    self.me.cumulative_pnl,
                    self.me.average_price[self.op.single_token],
                    self.me.position[self.op.single_token],
                    self.me.cumulative_pos_value
                )

            case action if -0.6 < action <= -0.4:
                order = Order(side=-1, price=self.op.pub_trade, size=50, order_type="market")
                self.me.check_orders(price=self.op.pub_trade, token=self.op.single_token, orders=[order])
                print(
                    self.op.raw_idx,
                    "sell2",
                    self.price,
                    self.op.pub_trade,
                    self.me.funds,
                    self.me.cumulative_pnl,
                    self.me.average_price[self.op.single_token],
                    self.me.position[self.op.single_token],
                    self.me.cumulative_pos_value
                )

            case action if -0.4 < action <= -0.2:
                order = Order(side=-1, price=self.op.pub_trade, size=10, order_type="market")
                self.me.check_orders(price=self.op.pub_trade, token=self.op.single_token, orders=[order])
                print(
                    self.op.raw_idx,
                    "sell1",
                    self.price,
                    self.op.pub_trade,
                    self.me.funds,
                    self.me.cumulative_pnl,
                    self.me.average_price[self.op.single_token],
                    self.me.position[self.op.single_token],
                    self.me.cumulative_pos_value
                )

            case _:
                self.me.check_orders(price=self.op.pub_trade, token=self.op.single_token)

    def _calculate_reward(self, action):
        pnl = self.me.cumulative_pnl
        pos_value = self.me.cumulative_pos_value
        avg_pos_price = self.me.average_price[self.op.single_token]
        current_price = self.op.pub_trade

        pnl_rate = pnl / self.me.init_funds
        pos_value_rate = pos_value / self.me.funds
        abs_pos_value_rate = abs(pos_value_rate)

        current_pos_rate = 0

        match self.me.side[self.op.single_token]:
            case 1:
                current_pos_rate = (current_price - avg_pos_price) / avg_pos_price

            case -1:
                current_pos_rate = (avg_pos_price - current_price) / avg_pos_price

            case _:
                pass

        self.pnl_rate = pnl_rate
        self.pos_value_rate = pos_value_rate
        self.current_pos_rate = current_pos_rate

        action_reward = -abs(action) if self.op.features_update_sig == 0 and (
                action > 0.2 or action < -0.2
        ) else 0.2 - abs(action)

        addition_reward = abs_pos_value_rate if 0.2 > abs_pos_value_rate >= 0.001 else 0

        # [5790736 rows x 4 columns]
        # Using cpu device
        # 9.972 0 1158147 5790735
        # 200 2000 0.001
        reward = (
                + 50. * current_pos_rate
                - 1. * abs_pos_value_rate
                + 10. * pnl_rate
                + 4. * addition_reward
                + 1. * action_reward
        )

        return scaled_sigmoid(reward, -3, 3) - 0.5


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

        self.features_update_sig = 0
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
            self.features_update_sig = 1

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

        else:
            self.features_update_sig = 0

    def generate_px_pct_bar_on_reset(
            self,
    ) -> pd.DataFrame:
        self.last_px = self.raw_df.iloc[self.raw_idx]["price"]
        self.last_ts = self.raw_df.iloc[self.raw_idx]["timestamp"]

        bars = []

        print(self.last_px, self.raw_idx, self.raw_idx_max_train, self.raw_idx_max)
        features_idx = 0
        print(self.rolling_window, self.rolling_normalize_window, self.price_threshold)
        while features_idx < self.rolling_window + self.rolling_normalize_window:
            px = self.raw_df.iloc[self.raw_idx]["price"]
            sz = self.raw_df.iloc[self.raw_idx]["amount"]
            ts = self.raw_df.iloc[self.raw_idx]["timestamp"]
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
        bars_df = bars_df.dropna()
        return bars_df

    @staticmethod
    def add_new_row(df, new_row, max_len):
        df = pd.concat([df, new_row], ignore_index=True)

        if len(df) > max_len:
            df.drop(df.index[0], inplace=True)

        return df
