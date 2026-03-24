"""Order book class, representing the bids and asks for an asset, exposing market state."""
from sympy import Order

from book_tree import BookTree
from price_level import PriceLevel


class OrderBook:
    """Order book class"""

    bids: BookTree
    asks: BookTree
    order_index: dict[str, Order]
    trade_log: list[dict]

    def __init__(self):
        self.bids = BookTree
        self.asks = BookTree
        self.order_index = {}
        self.trade_log = []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.bids}, {self.asks})"

    def add_limit_order(self, order: Order) -> None:
        """Add limit order to book in either bids or asks."""

        if order.side == 'buy':
            self.bids.insert(order)
        else:
            self.asks.insert(order)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""

        raise NotImplementedError()

    def best_bid(self) -> PriceLevel | None:
        """Returns best bid."""

        return self.bids.best()

    def best_ask(self) -> PriceLevel | None:
        """Returns best ask."""

        return self.asks.best()

    def spread(self) -> float | None:
        """Returns spread of best ask and bid."""

        return self.best_ask().price - self.best_bid().price \
            if self.best_bid() is not None and self.best_ask() is not None \
            else None

    def mid_price(self) -> float | None:
        """Returns arithmetic mid-price of best ask and bid."""

        return self.spread() / 2 if self.spread() is not None else None

    def depth_snapshot(self, levels: int = 10) -> dict[str, list]:
        """Returns depth snapshot up to a certain depth (levels)."""

        raise NotImplementedError()

    def log_trade(self, record: dict) -> None:
        """Log trade."""

        raise NotImplementedError()

    def flush_log(self, path: str) -> None:
        """Flush log."""

        raise NotImplementedError()
