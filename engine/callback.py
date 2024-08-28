from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        self.logger.record('train/reward', reward)
        return True
