from engine.evaluation.performance_evaluation_module import EvaluationEngine
from engine.evaluation.order_module import Order


class MatchEngine(EvaluationEngine):
    def __init__(
            self,
            tokens=None,
            funds=10000,
            maker_commission=0.0002,
            taker_commission=0.0005,
            debug=False
    ):
        super().__init__(funds)
        if tokens is None:
            tokens = ["FIL-USDT"]

        for token in tokens:
            self.token_default(token)

        self.debug = debug

        self.maker_commission = maker_commission
        self.taker_commission = taker_commission

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

    def update_pos_value(self):
        self.cumulative_pos_value = sum(
            self.position[key] * self.average_price[key] for key in self.position.keys()
        )

    def _calculate_average_price(self, filled_size: float, price: float, token: str):
        self.average_price[token] = (self.average_price[token] * (abs(self.position[token]) - filled_size)
                                     + price * filled_size) / abs(self.position[token])

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
        self.update_pos_value()

        match self.side[token]:
            case 1:
                if abs(self.cumulative_pos_value) < self.funds - order.size * order.price:
                    self.position[token] += order.size
                    self._calculate_average_price(
                        filled_size=order.size,
                        price=order.price,
                        token=token,
                    )

                elif self.debug:
                    print("Insufficient margin", self.cumulative_pos_value, self.funds)

            case -1:
                filled_size = min(abs(self.position[token]), order.size)
                cms = self.taker_commission if order.order_type == "market" else self.maker_commission
                cash_pnl = filled_size * (self.average_price[token] - order.price) - filled_size * cms * 2

                if filled_size >= abs(self.position[token]):
                    self.token_default(token)

                else:
                    self.position[token] += filled_size

                self.update(
                    price=order.price,
                    pnl=cash_pnl,
                )

            case 0:
                if self.funds - order.size * order.price > 0:
                    self.side[token] = 1
                    self.average_price[token] = order.price
                    self.position[token] = order.size

                elif self.debug:
                    print("Insufficient margin", self.cumulative_pos_value, self.funds, order.size, order.price)

            case _:
                pass

        self.update_pos_value()

    def _adjust_order_sell_fills(self, order: Order, token: str):
        self.update_pos_value()

        match self.side[token]:
            case -1:
                if abs(self.cumulative_pos_value) < self.funds - order.size * order.price:
                    self.position[token] -= order.size
                    self._calculate_average_price(
                        filled_size=order.size,
                        price=order.price,
                        token=token,
                    )

                elif self.debug:
                    print("Insufficient margin", self.cumulative_pos_value, self.funds)

            case 1:
                filled_size = min(abs(self.position[token]), order.size)
                cms = self.taker_commission if order.order_type == "market" else self.maker_commission
                cash_pnl = filled_size * (order.price - self.average_price[token]) - filled_size * cms * 2

                if filled_size >= abs(self.position[token]):
                    self.token_default(token)

                else:
                    self.position[token] -= filled_size

                self.update(
                    price=order.price,
                    pnl=cash_pnl,
                )

            case 0:
                if self.funds - order.size * order.price > 0:
                    self.side[token] = -1
                    self.average_price[token] = order.price
                    self.position[token] = -order.size

                elif self.debug:
                    print("Insufficient margin", self.cumulative_pos_value, self.funds, order.size, order.price)

            case _:
                pass

        self.update_pos_value()
