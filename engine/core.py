import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3 import PPO

from typing import Type, Union, Optional


class ResearchEngine:
    def __init__(
            self,
            env: GymEnv,
            data_type: pd.DataFrame
    ):
        self.model: Optional[PPO] = None
        self.data = data_type
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
            test_set_steps: int,
            callback=None,
    ) -> "ResearchEngine":
        if self.model is None:
            raise ValueError("Model is not init. Please call set_model before training.")

        self.model.learn(total_timesteps=test_set_steps, callback=callback)

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

        px_lst = []
        pnl_lst = []
        for i in range(max_steps):
            ppo_action, _states = self.model.predict(obs)
            obs, rewards, dones, info = self.env.step(ppo_action)
            # print(obs, rewards, i, max_steps)
            if i % 10 == 0:
                px_lst.append(obs[0][3])
                pnl_lst.append(rewards[0])

            if dones[0]:
                break  # 完成后跳出

        plt.style.use('seaborn-v0_8')

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        axs[0].plot(px_lst)
        axs[0].set_title('Price (Close)')

        axs[1].plot(pnl_lst)
        axs[1].set_title('PnL (Reward)')

        plt.tight_layout()
        plt.show()
