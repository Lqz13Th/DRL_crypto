import matplotlib.pyplot as plt


class BacktestEngine:
    def __init__(self):
        self.price_history = []
        self.pnl_history = []
        self.position_history = []
        self.max_drawdown = 0
        self.cumulative_pnl = 0
        self.current_position = 0

    def update(self, price: float, pnl: float):
        self.price_history.append(price)
        self.pnl_history.append(pnl)
        self.cumulative_pnl += pnl
        self.position_history.append(self.current_position)

    def calculate_max_drawdown(self):
        peak = self.pnl_history[0]
        for pnl in self.pnl_history:
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

        axs[1].plot(self.pnl_history)
        axs[1].set_title('PnL (Reward)')

        axs[2].plot(self.position_history)
        axs[2].set_title('Position')

        plt.tight_layout()
        plt.show()
