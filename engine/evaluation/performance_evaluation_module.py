import matplotlib.pyplot as plt
from engine.portfolio.account import AccountStatus


class EvaluationEngine(AccountStatus):
    def __init__(self, funds: float):
        super().__init__(funds)
        self.fund_history = []
        self.price_history = []
        self.position_history = []

        self.cumulative_pnl = 0
        self.max_drawdown = 0

    def evaluation_default(self):
        self.fund_history = []
        self.price_history = []
        self.position_history = []

        self.cumulative_pnl = 0
        self.max_drawdown = 0

    def update(self, price: float, pnl: float):
        self.cumulative_pnl += pnl
        self.funds += pnl

        self.price_history.append(price)
        self.fund_history.append(self.funds)
        self.position_history.append(self.cumulative_pos_value)

    def calculate_max_drawdown(self):
        if self.fund_history:
            peak = self.fund_history[0]
            for pnl in self.fund_history:
                if pnl > peak:
                    peak = pnl

                drawdown = peak - pnl
                if drawdown > self.max_drawdown:
                    self.max_drawdown = drawdown

    def plot(self):
        if self.price_history and self.fund_history and self.position_history:
            plt.style.use('seaborn-v0_8')
            fig, axs = plt.subplots(3, 1, figsize=(10, 12))

            axs[0].plot(self.price_history)
            axs[0].set_title('Price (Close)')

            axs[1].plot(self.fund_history)
            axs[1].set_title('PnL (Reward)')

            axs[2].plot(self.position_history)
            axs[2].set_title('Position')

            plt.tight_layout()
            plt.show()

        else:
            print("Algo not traded!")
