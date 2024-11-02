
class AccountStatus:
    def __init__(self, funds: float):
        self.init_funds = funds
        self.funds = funds
        self.cumulative_pos_value = 0

        self.average_price = {}
        self.open_orders = {}
        self.position = {}
        self.side = {}

    def token_default(self, token: str):
        self.average_price[token] = 0
        self.open_orders[token] = []
        self.position[token] = 0
        self.side[token] = 0

    def account_default(self):
        self.funds = self.init_funds
        self.cumulative_pos_value = 0

        self.average_price = {}
        self.open_orders = {}
        self.position = {}
        self.side = {}