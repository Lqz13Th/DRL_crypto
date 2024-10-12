from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        price = self.training_env.get_attr("price")[0]
        pnl_rate = self.training_env.get_attr("pnl_rate")[0]
        pos_value_rate = self.training_env.get_attr("pos_value_rate")[0]

        self.logger.record('train/reward', reward)
        self.logger.record('train/price', price)
        self.logger.record('train/pnl_rate', pnl_rate)
        self.logger.record('train/pos_value_rate', pos_value_rate)
        return True
