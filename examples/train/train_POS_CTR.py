import polars as pl
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

from engine.evaluation.match_module import MatchEngine
from engine.evaluation.order_module import Order

from ppo_algo.envs.pos_control_env import PositionControlEnv
from engine.callback import TensorboardCallback

if __name__ == '__main__':
    pl.Config.set_tbl_cols(10)  # 例如设置最多显示 100 列

    df = pl.read_csv(
        "C:/Users/Grayman/PycharmProjects/DRL_crypto/utils/normalized_data_BTC_2024_2025_Q1_0.0002_rolling50.csv"
    )
    print(df)
    print(df.columns)

    token = "BTC-USDT"
    # n_updates = total_time_steps // (n_steps * n_envs)
    env = make_vec_env(lambda: PositionControlEnv(df, token), n_envs=8)
    # tensorboard --logdir=examples/train/pos_ctr_tensorboard

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=2,
        learning_rate=3e-4,
        n_steps=128,
        # Number of steps to collect in each environment before updating
        batch_size=32,  # Batch size used for optimization
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=True,  # False for discrete actions
        sde_sample_freq=20,
        target_kl=None,
        stats_window_size=100,
        seed=13,
        device="auto",
        tensorboard_log="./pos_ctr_tensorboard/",
    )

    model.learn(
        total_timesteps=10**100,
        callback=TensorboardCallback(target_episodes=10, verbose=1),
    ).save(
        path="ppo_pos_ctr_btc",
    )

    del model, env

    env = make_vec_env(lambda: PositionControlEnv(df, token, mode='evaluation'), n_envs=1)
    model = PPO.load("ppo_pos_ctr_btc")
    obs = env.reset()
    backtest = MatchEngine(debug=True)
    backtest.token_default(token)
    a = 0
    contract_ratio = 0.0001
    while True:
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, infos = env.step(action)

        price = infos[0]['price']
        print(backtest.funds, backtest.position[token], backtest.average_price[token], backtest.cumulative_pnl)
        match action[0]:
            case action if action >= 0.99:
                order = Order(side=1, price=price, size=500 * contract_ratio, order_type="market")
                backtest.check_orders(price=price, token=token, orders=[order])
                print(
                    "buy5",
                )

            case action if 0.99 > action >= 0.8:
                order = Order(side=1, price=price, size=200 * contract_ratio, order_type="market")
                backtest.check_orders(price=price, token=token, orders=[order])
                print(
                    "buy4",
                )

            case action if 0.8 > action >= 0.6:
                order = Order(side=1, price=price, size=100 * contract_ratio, order_type="market")
                backtest.check_orders(price=price, token=token, orders=[order])
                print(
                    "buy3",
                )

            case action if 0.6 > action >= 0.4:
                order = Order(side=1, price=price, size=50 * contract_ratio, order_type="market")
                backtest.check_orders(price=price, token=token, orders=[order])
                print(
                    "buy2",
                )

            case action if 0.4 > action >= 0.2:
                order = Order(side=1, price=price, size=10 * contract_ratio, order_type="market")
                backtest.check_orders(price=price, token=token, orders=[order])
                print(
                    "buy1",
                )

            case action if action <= -0.99:
                order = Order(side=-1, price=price, size=500 * contract_ratio, order_type="market")
                backtest.check_orders(price=price, token=token, orders=[order])
                print(
                    "sell5",
                    order.size,
                    order.price
                )

            case action if -0.99 < action <= -0.8:
                order = Order(side=-1, price=price, size=200 * contract_ratio, order_type="market")
                backtest.check_orders(price=price, token=token, orders=[order])
                print(
                    "sell4",
                )

            case action if -0.8 < action <= -0.6:
                order = Order(side=-1, price=price, size=100 * contract_ratio, order_type="market")
                backtest.check_orders(price=price, token=token, orders=[order])
                print(
                    "sell3",
                )

            case action if -0.6 < action <= -0.4:
                order = Order(side=-1, price=price, size=50 * contract_ratio, order_type="market")
                backtest.check_orders(price=price, token=token, orders=[order])
                print(
                    "sell2",
                )

            case action if -0.4 < action <= -0.2:
                order = Order(side=-1, price=price, size=10 * contract_ratio, order_type="market")
                backtest.check_orders(price=price, token=token, orders=[order])
                print(
                    "sell1",
                )

            case _:
                backtest.check_orders(price=price, token=token)

        if dones[0]:
            break

    backtest.calculate_max_drawdown()
    backtest.plot()
