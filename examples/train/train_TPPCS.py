import pandas as pd

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

from engine.evaluation.match_module import MatchEngine
from engine.evaluation.order_module import Order
from ppo_algo.datas import high_frequency_data_parser
from ppo_algo.envs.algo_trade_price_percent_change_sampling import PricePercentChangeSamplingEnv
from engine.callback import TensorboardCallback

if __name__ == '__main__':
    pd.set_option("display.max_rows", 5000)
    pd.set_option("expand_frame_repr", False)

    psd = high_frequency_data_parser.ParseHFTData()
    df = psd.parse_trade_data_tardis(
        "C:/Users/trade/PycharmProjects/DataGrabber/datasets/binance-futures_trades_2024-08-05_FILUSDT.csv.gz"
    )

    print(df)

    token = "FIL-USDT"
    # n_updates = total_time_steps // (n_steps * n_envs)
    env = make_vec_env(lambda: PricePercentChangeSamplingEnv(df, token), n_envs=1)
    # tensorboard --logdir=examples/train/TPPCS_tensorboard

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=2,
        learning_rate=3e-4,
        n_steps=2048,
        # Number of steps to collect in each environment before updating
        batch_size=64,  # Batch size used for optimization
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,  # False for discrete actions
        sde_sample_freq=-1,
        target_kl=None,
        stats_window_size=100,
        seed=13,
        device="auto",
        tensorboard_log="./TPPCS_tensorboard/",
    )

    model.learn(
        total_timesteps=10**10,
        callback=TensorboardCallback(target_episodes=10, verbose=1),
    ).save(
        path="ppo_crypto_trading",
    )

    del model, env

    env = make_vec_env(lambda: PricePercentChangeSamplingEnv(df, token, mode='evaluation'), n_envs=1)
    model = PPO.load("ppo_crypto_trading")
    obs = env.reset()
    backtest = MatchEngine(debug=True)
    backtest.token_default(token)
    a = 0
    while True:
        ppo_action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, infos = env.step(ppo_action)

        price = infos[0]['price']
        match infos[0]['trade_signal']:
            case 1:
                order = Order(side=1, price=price, size=50, order_type="market")
                backtest.check_orders(price=price, token=token, orders=[order])

            case -1:
                order = Order(side=-1, price=price, size=50, order_type="market")
                backtest.check_orders(price=price, token=token, orders=[order])

            case _:
                backtest.check_orders(price=price, token=token)

        if dones[0]:
            break

    backtest.calculate_max_drawdown()
    backtest.plot()
