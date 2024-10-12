import pandas as pd

from stable_baselines3.common.env_util import make_vec_env

from ppo_algo.datas import high_frequency_data_parser
from ppo_algo.gyms.algo_trade_price_percent_change_sampling import PricePercentChangeSamplingEnv
from engine.callback import TensorboardCallback
from engine.core import ResearchEngine

if __name__ == '__main__':
    pd.set_option("display.max_rows", 5000)
    pd.set_option("expand_frame_repr", False)

    psd = high_frequency_data_parser.ParseHFTData()
    df = psd.parse_agg_trade_data_binance("C:/Work Files/data/backtest/aggtrade/FILUSDT/FILUSDT-aggTrades-2024-04.csv")

    max_steps = df.index.max()
    print(df)

    # n_updates = total_time_steps // (n_steps * n_envs)

    engine = ResearchEngine(
        env=make_vec_env(lambda: PricePercentChangeSamplingEnv(df, "FIL-USDT"), n_envs=1),
        data_type=df
    )
    # tensorboard --logdir=examples/train/TPPCS_tensorboard

    engine.model_init(
        policy="MultiInputPolicy",
        verbose=2,
        learning_rate=3e-4,
        n_steps=256,
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
        tensorboard_log="./TPPCS_tensorboard/",
    ).model_learn(
        total_timestep=10**100,
        callback=TensorboardCallback(),
    ).model_save(
        path="TPPCS",
    ).model_load(
        path="TPPCS",
    ).run()
