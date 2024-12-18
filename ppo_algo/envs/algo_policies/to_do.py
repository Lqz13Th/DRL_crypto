from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import time
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-sample-v0",
                ip_address="127.0.0.1",
                image_shape=(256, 144, 1),
                step_length=3,
            )
        )
    ]
)

env = VecTransposeImage(env)


class CustomCombinedExtractors(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.items():

            if key == "image":

                print(subspace.shape[0], subspace.shape[1], subspace.shape[2])
                image_channels = subspace.shape[0]
                extractors[key] = nn.Sequential(
                    nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Flatten()
                )
                total_concat_size += (subspace.shape[1] // 2) * (subspace.shape[2] // 2) * 32
            elif key == "goal_dis":
                print("subspace.shape[0]:", subspace.shape[0])
                print("subspace:", subspace)
                print(subspace.shape[0])
                extractors[key] = nn.Sequential(
                    nn.Flatten()
                )

                total_concat_size += subspace.shape[0]

            elif key == "angle":
                print("subspace.shape[0]:", subspace.shape[0])
                print("subspace:", subspace)
                extractors[key] = nn.Sequential(
                    nn.Flatten()
                )

                total_concat_size += subspace.shape[0]

        self.extractors = nn.ModuleDict(extractors)

        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:

        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))

        return torch.cat(encoded_tensor_list, dim=1)


class CustomNetwork(nn.Module):

    def __init__(
            self,
            feature_dim: int,
            last_layer_dim_pi: int = 9,
            last_layer_dim_vf: int = 1,
    ):
        super().__init__()

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.policy_net = nn.Sequential(
            nn.Linear(294914, last_layer_dim_pi),
            nn.ReLU()
        )

        self.value_net = nn.Sequential(
            nn.Linear(294914, last_layer_dim_vf)
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(MultiInputActorCriticPolicy):

    def __init__(
            self,
            observation_space: spaces.Dict,
            action_space: spaces.Space,
            lr_schedule: Callable[[float], float],
            *args,
            **kwargs,
    ):
        kwargs["ortho_init"] = False,
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.features_extractor = CustomCombinedExtractors(self.observation_space.spaces)
        self.mlp_extractor = CustomNetwork(self.features_dim)


model = PPO(
    CustomActorCriticPolicy,
    env,
    gamma=0.99,
    learning_rate=0.00025,
    gae_lambda=0.95,
    n_steps=128,
    batch_size=128,
    n_epochs=10,
    clip_range=0.2,
    normalize_advantage=True,
    ent_coef=0.05,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./tb_logs/",
    seed=100,
    device="cuda",
    use_sde=False,
    verbose=1,
)

callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path="./tb_logs/",
    eval_freq=10000,
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

model.learn(
    total_timesteps=5e5,
    tb_log_name="ppo_airsim_drone_run_" + str(time.time()),
    **kwargs
)

model.save("ppo_airsim_drone_policy")
