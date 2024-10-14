from engine.backtest.account_evaluation_module import EvaluationEngine
from engine.backtest.order_module import Order


class BacktestEngine:
    def __init__(self):
        self.eval = EvaluationEngine(fund=10000)

        self.average_price = {}
        self.open_orders = {}
        self.position = {}
        self.side = {}

        self.maker_commission = 0.0002
        self.taker_commission = 0.0005

    def token_default(self, token: str):
        self.average_price[token] = 0
        self.open_orders[token] = []
        self.position[token] = 0
        self.side[token] = 0

    def check_orders(self, price: float, token: str, orders=None):
        self._check_limit_order_status(price, token)

        if orders:
            for order in orders:
                match order.order_type:
                    case "limit":
                        self.open_orders[token].append(order)

                    case "market" if order.side == 1:
                        self._adjust_order_buy_fills(order, token)

                    case "market" if order.side == -1:
                        self._adjust_order_sell_fills(order, token)

    def _check_limit_order_status(self, price: float, token: str):
        for order in self.open_orders[token]:
            if isinstance(order, Order):
                match order.side:
                    case 1 if price < order.price:
                        self._adjust_order_buy_fills(
                            order=order,
                            token=token,
                        )

                    case -1 if price > order.price:
                        self._adjust_order_sell_fills(
                            order=order,
                            token=token,
                        )

                    case _:
                        pass

    def _adjust_order_buy_fills(self, order: Order, token: str):
        self.eval.total_position_value = sum(
            self.position[key] * self.average_price[key] for key in self.position.keys()
        )

        match self.side[token]:
            case 1:
                if abs(self.eval.total_position_value) < self.eval.funds - order.size * order.price:
                    self.position[token] += order.size
                    self._calculate_average_price(
                        filled_size=order.size,
                        price=order.price,
                        token=token,
                    )

                else:
                    print("Insufficient margin", self.eval.total_position_value, self.eval.funds)

            case -1:
                filled_size = min(abs(self.position[token]), order.size)
                cms = self.taker_commission if order.order_type == "market" else self.maker_commission
                cash_pnl = filled_size * (self.average_price[token] - order.price) - filled_size * cms * 2

                if filled_size >= abs(self.position[token]):
                    self.token_default(token)

                else:
                    self.position[token] += filled_size

                self.eval.update(
                    price=order.price,
                    pnl=cash_pnl,
                )

            case 0:
                self.side[token] = 1
                self.average_price[token] = order.price
                self.position[token] = order.size

            case _:
                pass

    def _adjust_order_sell_fills(self, order: Order, token: str):
        self.eval.total_position_value = sum(
            self.position[key] * self.average_price[key] for key in self.position.keys()
        )

        match self.side[token]:
            case -1:
                if abs(self.eval.total_position_value) < self.eval.funds - order.size * order.price:
                    self.position[token] -= order.size
                    self._calculate_average_price(
                        filled_size=order.size,
                        price=order.price,
                        token=token,
                    )

                else:
                    print("Insufficient margin", self.eval.total_position_value, self.eval.funds)

            case 1:
                filled_size = min(abs(self.position[token]), order.size)
                cms = self.taker_commission if order.order_type == "market" else self.maker_commission
                cash_pnl = filled_size * (order.price - self.average_price[token]) - filled_size * cms * 2

                if filled_size >= abs(self.position[token]):
                    self.token_default(token)

                else:
                    self.position[token] -= filled_size

                self.eval.update(
                    price=order.price,
                    pnl=cash_pnl,
                )

            case 0:
                self.side[token] = -1
                self.average_price[token] = order.price
                self.position[token] = -order.size

            case _:
                pass

    def _calculate_average_price(self, filled_size: float, price: float, token: str):
        self.average_price[token] = (self.average_price[token] * (abs(self.position[token]) - filled_size)
                                     + price * filled_size) / abs(self.position[token])
