from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    def __init__(self, target_episodes, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.target_episodes = target_episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        price = self.training_env.unwrapped.get_attr("price")[0]
        pnl_rate = self.training_env.unwrapped.get_attr("pnl_rate")[0]
        pos_value_rate = self.training_env.unwrapped.get_attr("pos_value_rate")[0]

        self.logger.record('train/reward', reward)
        self.logger.record('train/price', price)
        self.logger.record('train/pnl_rate', pnl_rate)
        self.logger.record('train/pos_value_rate', pos_value_rate)

        if self.locals.get("dones", [False])[0]:
            self.episode_count += 1
            if self.verbose > 0:
                print(f"Episode {self.episode_count} ended")

            if self.episode_count >= self.target_episodes:
                print(f"Reached target episodes ({self.target_episodes}). Stopping training.")
                return False

        return True
