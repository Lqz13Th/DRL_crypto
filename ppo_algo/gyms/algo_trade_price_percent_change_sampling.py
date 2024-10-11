import numpy as np
import gymnasium as gym

from gymnasium import spaces


class ObservationSpaceParser:
    def __init__(self, data_frame):
        self.raw_df = data_frame




class PricePercentChangeSamplingEnv(gym.Env):
    def __init__(
            self,
            data_frame,
            numb_features: int,
            rolling_window: int,
    ):
        super(PricePercentChangeSamplingEnv, self).__init__()
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

        self.action_space = spaces.Discrete(3)  # 三种动作：保持，买入，卖出
        self.observation_space = spaces.Dict({
            "pub_price": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "features": spaces.Box(low=0, high=1, shape=(numb_features, rolling_window), dtype=np.float32),
        })

    def reset(self, seed=None, options=None):  # 添加 seed 参数
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0

        return self._next_observation(), {}

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

