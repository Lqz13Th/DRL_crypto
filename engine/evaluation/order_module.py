

class Order:
    def __init__(self, side: int, price: float, size: float, order_type: str):
        self.side = side
        self.size = size
        self.price = price
        self.order_type = order_type  # 'limit' or 'market'
