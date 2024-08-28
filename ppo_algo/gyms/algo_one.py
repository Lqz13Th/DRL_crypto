import numpy as np
import pandas as pd
import gymnasium as gym

from gymnasium import spaces
from stable_baselines3.common.env_util import make_vec_env

from ppo_algo.datas import candle_data_parser
from engine.callback import TensorboardCallback
from engine.core import ResearchEngine


class CryptoTradingEnv(gym.Env):
    def __init__(self, data_frame):
        super(CryptoTradingEnv, self).__init__()
        self.df = data_frame
        self.current_step = 0
        self.initial_balance = 1000
        self.balance = self.initial_balance
        self.position = 0  # 持仓状态：0=空仓，1=持有

        assert all(col in df.columns for col in [
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
        self._take_action(action_state)
        self.current_step += 1

        reward = self._calculate_reward()
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        current_obs = self._next_observation()
        return current_obs, reward, terminated, truncated, {}

    def _next_observation(self):
        observation_value = self.df.iloc[self.current_step][['Open', 'High', 'Low', 'Close', 'Volume']]
        return observation_value.values

    def _take_action(self, action):
        current_price = self.df.loc[self.current_step, 'Close']

        if action == 1:  # 买入
            if self.position == 0:
                self.position = self.balance / current_price
                self.balance = 0
        elif action == 2:  # 卖出
            if self.position > 0:
                self.balance = self.position * current_price
                self.position = 0

    def _calculate_reward(self):
        current_price = self.df.loc[self.current_step, 'Close']
        total_value = self.balance + self.position * current_price
        reward = total_value - self.initial_balance
        return reward


if __name__ == '__main__':
    pd.set_option("display.max_rows", 5000)
    pd.set_option("expand_frame_repr", False)

    candle_data = candle_data_parser.ParseCandleData()
    df = candle_data.parse_candle_data_okx('C:/Work Files/data/backtest/candle/candle1m/FIL-USDT1min.csv')
    print(df)

    df = df.tail(int(df.index.max() * 0.05)).reset_index(drop=True)
    max_steps = df.index.max()
    print(df)

    engine = ResearchEngine(
        env=make_vec_env(lambda: CryptoTradingEnv(df), n_envs=1),
        data_type=df
    )

    engine.model_init(
        policy="MlpPolicy",
        verbose=2,
        learning_rate=3e-4,
        n_steps=max(int(max_steps * 0.005), 60 * 24 * 7),
        # Number of steps to collect in each environment before updating
        batch_size=32,  # Batch size used for optimization
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,  # False for discrete actions
        sde_sample_freq=-1,
        target_kl=None,
        stats_window_size=100,
        seed=1,
        device="auto",
        tensorboard_log="./ppo_crypto_trading_tensorboard/",
    ).model_learn(
        test_set_steps=int(max_steps * 0.7),
        callback=TensorboardCallback(),
    ).model_save(
        path="ppo_crypto_trading",
    ).model_load(
        path="ppo_crypto_trading",
    ).run(
        max_steps=max_steps,
    )
