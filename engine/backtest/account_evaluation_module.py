import matplotlib.pyplot as plt


class EvaluationEngine:
    def __init__(self, fund: float):
        self.fund_history = []
        self.price_history = []
        self.position_history = []

        self.cumulative_pnl = 0
        self.funds = fund
        self.max_drawdown = 0
        self.total_position_value = 0

    def update(self, price: float, pnl: float):
        self.cumulative_pnl += pnl
        self.funds += pnl

        self.price_history.append(price)
        self.fund_history.append(self.funds)
        self.position_history.append(self.total_position_value)

    def calculate_max_drawdown(self):
        peak = self.fund_history[0]
        for pnl in self.fund_history:
            if pnl > peak:
                peak = pnl

            drawdown = peak - pnl
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown

    def plot(self):
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
