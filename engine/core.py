import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3 import PPO
from typing import Union, Optional

from engine.backtest.backtest_module import BacktestEngine
from engine.backtest.order_module import Order


class ResearchEngine(BacktestEngine):
    def __init__(
            self,
            env: GymEnv,
            data_type: pd.DataFrame
    ):
        super().__init__()
        self.model: Optional[PPO] = None
        self.backtest_data = data_type
        self.env = env

    def model_init(
            self,
            policy="MlpPolicy",
            learning_rate=3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl=None,
            stats_window_size: int = 100,
            tensorboard_log=None,
            verbose: int = 0,
            seed=None,
            device="auto",
    ) -> "ResearchEngine":
        if self.env is None:
            raise ValueError("Environment is not set. Please load an environment before setting the model.")

        self.model = PPO(
            policy,
            self.env,
            verbose=verbose,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            seed=seed,
            device=device,
            tensorboard_log=tensorboard_log,
        )

        return self

    def model_learn(
            self,
            total_timestep: int,
            callback=None,
    ) -> "ResearchEngine":
        if self.model is None:
            raise ValueError("Model is not init. Please call set_model before training.")

        self.model.learn(total_timesteps=total_timestep, callback=callback)

        return self

    def model_save(
            self,
            path: str
    ) -> "ResearchEngine":
        if self.model is None:
            raise ValueError("Model is not init. Please call set_model before training.")

        self.model.save(path)

        return self

    def model_load(
            self,
            path: str
    ) -> "ResearchEngine":
        del self.model

        self.model = PPO.load(path)

        return self

    def run(self, max_steps: int):
        obs = self.env.reset()

        token = "FIL-USDT"
        self.token_default(token)

        while True:
            ppo_action, _states = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, info = self.env.step(ppo_action)
            # print(obs, rewards, i, max_steps)

            price = obs[0][3]
            print(ppo_action, self.eval.total_position, self.position[token])
            match ppo_action:
                case 1:
                    order = Order(side=1, price=price, size=1, order_type="market")
                    self.check_orders(price=obs[0][3], token=token, orders=[order])

                case 2:
                    order = Order(side=-1, price=price, size=1, order_type="market")
                    self.check_orders(price=obs[0][3], token=token, orders=[order])

                case _:
                    self.check_orders(price=obs[0][3], token=token)

            if dones[0]:
                break

        self.eval.calculate_max_drawdown()
        self.eval.plot()
