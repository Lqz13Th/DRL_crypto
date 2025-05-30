import numpy as np
import gymnasium as gym

from gymnasium import spaces


class CandleBarSimpleEnv(gym.Env):
    def __init__(self, data_frame):
        super(CandleBarSimpleEnv, self).__init__()
        self.df = data_frame
        self.current_step = 0
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.position = 0  # 持仓状态：0=空仓，1=持有

        assert all(col in self.df.columns for col in [
            'Open',
            'High',
            'Low',
            'Close',
            'Volume'
        ]), "DataFrame should have columns: 'Open', 'High', 'Low', 'Close', 'Volume'"

        self.action_space = spaces.Discrete(3)  # 三种动作：保持，买入，卖出
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self, seed=None, options=None):  # 添加 seed 参数
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0

        return self._next_observation(), {}

    def step(self, action_state):
        trade_signal = self._take_action(action_state)
        self.current_step += 1

        reward = self._calculate_reward()
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        current_obs = self._next_observation()
        return current_obs, reward, terminated, truncated, {"trade_signal": trade_signal}

    def _next_observation(self):
        observation_value = self.df.iloc[self.current_step][['Open', 'High', 'Low', 'Close', 'Volume']]
        return observation_value.values

    def _take_action(self, action):
        current_price = self.df.loc[self.current_step, 'Close']

        trade_signal = 0

        if action == 1:  # 买入
            if self.position == 0:
                self.position = self.balance / current_price
                self.balance = 0
                trade_signal = 1

        elif action == 2:  # 卖出
            if self.position > 0:
                self.balance = self.position * current_price
                self.position = 0
                trade_signal = -1

        return trade_signal

    def _calculate_reward(self):
        current_price = self.df.loc[self.current_step, 'Close']
        total_value = self.balance + self.position * current_price
        reward = total_value - self.initial_balance
        return reward

