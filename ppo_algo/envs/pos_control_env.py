import numpy as np
import gymnasium as gym
import polars as pl

from gymnasium import spaces

from engine.evaluation.order_module import Order
from engine.evaluation.match_module import MatchEngine
from utils.normalizer import scaled_sigmoid


class PositionControlEnv(gym.Env):
    def __init__(
            self,
            data_frame: pl.DataFrame,
            single_token: str,
            mode='train',
    ):
        super(PositionControlEnv, self).__init__()
        self.me = MatchEngine(debug=True)
        self.me.token_default(single_token)

        self.features_df = data_frame.select([
            "ts",
            "price",
            "scaled_sum_buy_size",
            "scaled_sum_sell_size",
            "scaled_timestamp_duration",
            "scaled_price_pct_change",
            "scaled_buy_sell_imbalance",
            "scaled_change_side",
            "scaled_price_pct_change_sum_100",
        ]).drop_nulls()

        self.mode = mode

        self.row_idx = 0
        self.total_idx = self.features_df.shape[0] - 1
        self.total_train_idx = int(self.total_idx * 0.7)

        self.pnl_rate = 0
        self.pos_value_rate = 0
        self.current_pos_rate = 0

        self.token = single_token
        self.price = self.features_df[0, "price"]
        self.pos_hold_penalty = 0
        self.pos_not_open_penalty = 0

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # multi modal | tensorboard --logdir=examples/train/POS_CTR_tensorboard/
        self.observation_space = spaces.Dict({
            "features": spaces.Box(low=0, high=1, shape=(7, ), dtype=np.float32),
            "account_states": spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
        })

    def reset(self, seed=None, options=None):  # 添加 seed 参数
        super().reset(seed=seed)
        # self.row_idx = self.total_train_idx if self.mode == "evaluation" else 0
        self.row_idx = 0
        self.pnl_rate = 0
        self.pos_value_rate = 0
        self.current_pos_rate = 0
        self.pos_hold_penalty = 0
        self.pos_not_open_penalty = 0

        self.me.evaluation_default()
        self.me.account_default()
        self.me.token_default(self.token)

        print(self.features_df)
        print(self.features_df.row(self.row_idx), type(self.features_df.row(self.row_idx)))
        print(self.row_idx, self.total_train_idx, self.total_idx)
        init_obs = {
            "features": np.array(self.features_df.row(self.row_idx)[2:], dtype=np.float32),
            "account_states": np.array(
                [
                    scaled_sigmoid(self.pnl_rate, -0.1, 0.1),
                    scaled_sigmoid(self.pos_value_rate, -1, 1),
                    scaled_sigmoid(self.current_pos_rate, -0.5, 0.5),
                    scaled_sigmoid(self.me.side[self.token], -1, 1),
                ],
                dtype=np.float32,
            ),
        }

        return init_obs, {}

    def step(self, action_state):
        next_obs = self._next_observation()
        self.price = self.features_df.row(self.row_idx)[1]
        self._take_action(action_state)

        reward = self._calculate_reward(action_state)

        terminated = False
        if self.mode == "evaluation":
            if self.row_idx >= self.total_idx:
                terminated = True
                print("done", self.me.funds)
                self.row_idx = 0

        elif self.row_idx >= self.total_train_idx:
            terminated = True
            self.row_idx = 0

        elif self.me.funds < self.me.init_funds * 0.5:
            print("Liquidation!")
            terminated = True
            reward = - 0.5
            self.row_idx = 0

        truncated = False

        infos = {
            "price": self.price,
        }

        # print(next_obs)
        return next_obs, reward, terminated, truncated, infos

    def _next_observation(self):
        next_obs = {
            "features": np.array(self.features_df.row(self.row_idx)[2:], dtype=np.float32),
            "account_states": np.array(
                [
                    scaled_sigmoid(self.pnl_rate, -0.1, 0.1),
                    scaled_sigmoid(self.pos_value_rate, -1, 1),
                    scaled_sigmoid(self.current_pos_rate, -0.5, 0.5),
                    scaled_sigmoid(self.me.side[self.token], -1, 1),
                ],
                dtype=np.float32,
            ),
        }
        self.row_idx = min(self.row_idx + 1, self.total_idx)

        return next_obs

    def _take_action(self, action):
        print(
            self.row_idx,
            action[0],
            self.me.side[self.token],
            self.me.funds,
            self.me.cumulative_pnl,
            self.me.average_price[self.token],
            self.price,
            self.me.position[self.token],
            self.me.cumulative_pos_value
        )
        self.me.update_pos_value()

        contract_ratio = 0.0001
        match action[0]:
            case action if action >= 0.99:
                order = Order(side=1, price=self.price, size=500 * contract_ratio, order_type="market")
                self.me.check_orders(price=self.price, token=self.token, orders=[order])
                print(
                    self.row_idx,
                    "buy5",
                    self.price,
                    self.me.funds,
                    self.me.cumulative_pnl,
                    self.me.average_price[self.token],
                    self.me.position[self.token],
                    self.me.cumulative_pos_value
                )

            case action if 0.99 > action >= 0.8:
                order = Order(side=1, price=self.price, size=200 * contract_ratio, order_type="market")
                self.me.check_orders(price=self.price, token=self.token, orders=[order])
                print(
                    self.row_idx,
                    "buy4",
                    self.price,
                    self.me.funds,
                    self.me.cumulative_pnl,
                    self.me.average_price[self.token],
                    self.me.position[self.token],
                    self.me.cumulative_pos_value
                )

            case action if 0.8 > action >= 0.6:
                order = Order(side=1, price=self.price, size=100 * contract_ratio, order_type="market")
                self.me.check_orders(price=self.price, token=self.token, orders=[order])
                print(
                    self.row_idx,
                    "buy3",
                    self.price,
                    self.me.funds,
                    self.me.cumulative_pnl,
                    self.me.average_price[self.token],
                    self.me.position[self.token],
                    self.me.cumulative_pos_value
                )

            case action if 0.6 > action >= 0.4:
                order = Order(side=1, price=self.price, size=50 * contract_ratio, order_type="market")
                self.me.check_orders(price=self.price, token=self.token, orders=[order])
                print(
                    self.row_idx,
                    "buy2",
                    self.price,
                    self.me.funds,
                    self.me.cumulative_pnl,
                    self.me.average_price[self.token],
                    self.me.position[self.token],
                    self.me.cumulative_pos_value
                )

            case action if 0.4 > action >= 0.2:
                order = Order(side=1, price=self.price, size=10 * contract_ratio, order_type="market")
                self.me.check_orders(price=self.price, token=self.token, orders=[order])
                print(
                    self.row_idx,
                    "buy1",
                    self.price,
                    self.me.funds,
                    self.me.cumulative_pnl,
                    self.me.average_price[self.token],
                    self.me.position[self.token],
                    self.me.cumulative_pos_value
                )

            case action if action <= -0.99:
                order = Order(side=-1, price=self.price, size=500 * contract_ratio, order_type="market")
                self.me.check_orders(price=self.price, token=self.token, orders=[order])
                print(
                    self.row_idx,
                    "sell5",
                    self.price,
                    self.me.funds,
                    self.me.cumulative_pnl,
                    self.me.average_price[self.token],
                    self.me.position[self.token],
                    self.me.cumulative_pos_value
                )

            case action if -0.99 < action <= -0.8:
                order = Order(side=-1, price=self.price, size=200 * contract_ratio, order_type="market")
                self.me.check_orders(price=self.price, token=self.token, orders=[order])
                print(
                    self.row_idx,
                    "sell4",
                    self.price,
                    self.me.funds,
                    self.me.cumulative_pnl,
                    self.me.average_price[self.token],
                    self.me.position[self.token],
                    self.me.cumulative_pos_value
                )

            case action if -0.8 < action <= -0.6:
                order = Order(side=-1, price=self.price, size=100 * contract_ratio, order_type="market")
                self.me.check_orders(price=self.price, token=self.token, orders=[order])
                print(
                    self.row_idx,
                    "sell3",
                    self.price,
                    self.me.funds,
                    self.me.cumulative_pnl,
                    self.me.average_price[self.token],
                    self.me.position[self.token],
                    self.me.cumulative_pos_value
                )

            case action if -0.6 < action <= -0.4:
                order = Order(side=-1, price=self.price, size=50 * contract_ratio, order_type="market")
                self.me.check_orders(price=self.price, token=self.token, orders=[order])
                print(
                    self.row_idx,
                    "sell2",
                    self.price,
                    self.me.funds,
                    self.me.cumulative_pnl,
                    self.me.average_price[self.token],
                    self.me.position[self.token],
                    self.me.cumulative_pos_value
                )

            case action if -0.4 < action <= -0.2:
                order = Order(side=-1, price=self.price, size=10 * contract_ratio, order_type="market")
                self.me.check_orders(price=self.price, token=self.token, orders=[order])
                print(
                    self.row_idx,
                    "sell1",
                    self.price,
                    self.me.funds,
                    self.me.cumulative_pnl,
                    self.me.average_price[self.token],
                    self.me.position[self.token],
                    self.me.cumulative_pos_value
                )

            case _:
                self.me.check_orders(price=self.price, token=self.token)

    def _calculate_reward(self, action):
        pnl = self.me.cumulative_pnl
        pos_value = self.me.cumulative_pos_value
        avg_pos_price = self.me.average_price[self.token]
        current_price = self.price

        pnl_rate = pnl / self.me.init_funds
        pos_value_rate = pos_value / self.me.funds
        abs_pos_value_rate = abs(pos_value_rate)

        current_pos_rate = 0

        match self.me.side[self.token]:
            case 1:
                current_pos_rate = (current_price - avg_pos_price) / avg_pos_price
                if action > -0.2:
                    self.pos_hold_penalty += 10

                else:
                    self.pos_hold_penalty = 0

                self.pos_not_open_penalty = 0

            case -1:
                current_pos_rate = (avg_pos_price - current_price) / avg_pos_price
                if action < 0.2:
                    self.pos_hold_penalty += 10

                else:
                    self.pos_hold_penalty = 0

                self.pos_not_open_penalty = 0

            case _:
                self.pos_hold_penalty = 0
                self.pos_not_open_penalty += 10
                pass

        self.pnl_rate = pnl_rate
        self.pos_value_rate = pos_value_rate
        self.current_pos_rate = current_pos_rate

        pos_hedge_reward = abs_pos_value_rate if 0.3 > abs_pos_value_rate else 0
        action_reward = -abs(action) if action > 0.3 or action < -0.3 else 0.2 - abs(action) * 0.1
        pos_hold_reward = -self.pos_hold_penalty * 0.0001
        pos_not_open_reward = -self.pos_not_open_penalty * 0.0001

        reward = (
                + 50. * current_pos_rate
                - 1. * abs_pos_value_rate
                # # + 1. * pnl_rate
                # + 1. * pos_hedge_reward
                # # + 1. * action_reward
                # + 1. * pos_hold_reward
                # + 1. * pos_not_open_reward
        )

        return scaled_sigmoid(reward, -5, 5) - 0.5
