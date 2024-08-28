import pandas as pd

from stable_baselines3.common.env_util import make_vec_env

from ppo_algo.datas import candle_data_parser
from ppo_algo.gyms.algo_one import CryptoTradingEnv
from engine.callback import TensorboardCallback
from engine.core import ResearchEngine

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
