"""Quantyze matching engine.

Module Description
==================
This module contains the matching engine used to process incoming market events
against the live order book for Quantyze. It applies price-time priority by
matching better prices first and then enforcing FIFO order within each price
level. The engine also records fills, cancellations, and execution metrics
during replay.

Copyright Information
===============================

Copyright (c) 2026 Nicolas Miranda Cantanhede
"""

from __future__ import annotations
from dataclasses import dataclass

from core.order_book import OrderBook
from core.orders import Event, IncomingOrder, Order
from core.price_level import PriceLevel


@dataclass
class MatchingEngine:
    """Process events against the live order book.

    Instance Attributes:
    - book: the shared live order book used for matching
    - metrics: running counters for fills, slippage, and cancellations
    - execution_log: fill records produced during matching

    Representation Invariants:
    - 'total_filled' in self.metrics
    - 'total_slippage' in self.metrics
    - 'fill_count' in self.metrics
    - 'cancel_count' in self.metrics
    """

    book: OrderBook
    metrics: dict[str, float | int]
    execution_log: list[dict]

    def __init__(self, book: OrderBook) -> None:
        """Initialize this matching engine with the given order book."""

        self.book = book
        self.metrics = {
            "total_filled": 0.0,
            "total_slippage": 0.0,
            "fill_count": 0,
            "cancel_count": 0
        }

        self.execution_log = []

    def process_event(self, event: Event) -> list[dict]:
        """Process <event> and return the fill records produced.

        >>> from datetime import datetime
        >>> book = OrderBook()
        >>> engine = MatchingEngine(book)
        >>> engine.process_event(Event(datetime(2026, 1, 1, 9, 30), 'ask1', 'sell', 'limit', 100.0, 2.0))
        []
        >>> fills = engine.process_event(
        ...     Event(datetime(2026, 1, 1, 9, 30, 1), 'buy1', 'buy', 'limit', 100.0, 1.0)
        ... )
        >>> fills[0]['exec_price']
        100.0
        >>> fills[0]['filled_qty']
        1.0
        >>> book.best_ask().volume
        1.0
        """

        if event.order_type == "limit":
            return self._process_limit(event)
        elif event.order_type == "market":
            return self._process_market(event)
        elif event.order_type == "cancel":
            self._process_cancel(event)
            return []
        else:
            raise ValueError

    def _process_limit(self, event: Event) -> list[dict]:
        """Process a limit-order event.

        Attempt immediate matching against the opposite side of the book.
        If residual quantity remains, rest the order in the appropriate tree.
        """

        incoming = IncomingOrder(event)
        fills = []

        if incoming.side == "buy":
            best_ask = self.book.best_ask()
            expected_price = best_ask.price if best_ask is not None else incoming.price

            while (
                    not incoming.is_complete()
                    and best_ask is not None
                    and incoming.price >= best_ask.price
            ):

                fills.extend(self._match(incoming, best_ask))
                self._clean_empty_levels()
                best_ask = self.book.best_ask()
        else:
            best_bid = self.book.best_bid()
            expected_price = best_bid.price if best_bid is not None else incoming.price

            while (
                    not incoming.is_complete()
                    and best_bid is not None
                    and incoming.price <= best_bid.price
            ):
                fills.extend(self._match(incoming, best_bid))
                self._clean_empty_levels()
                best_bid = self.book.best_bid()

        if not incoming.is_complete():
            resting_event = Event(
                incoming.timestamp,
                incoming.order_id,
                incoming.side,
                "limit",
                incoming.price,
                incoming.remaining_qty
            )
            self.book.add_limit_order(Order(resting_event))

        if fills != []:
            self.metrics["total_slippage"] += self._calc_slippage(expected_price, fills)

        return fills

    def _process_market(self, event: Event) -> list[dict]:
        """Process a market-order event.

        Walk the opposite side of the book from the current best price inward
        until the order is fully filled or the book is empty.
        """

        incoming = IncomingOrder(event)
        fills = []

        if incoming.side == "buy":
            best_ask = self.book.best_ask()
            expected_price = best_ask.price if best_ask is not None else None

            while not incoming.is_complete() and best_ask is not None:
                fills.extend(self._match(incoming, best_ask))
                self._clean_empty_levels()
                best_ask = self.book.best_ask()
        else:
            best_bid = self.book.best_bid()
            expected_price = best_bid.price if best_bid is not None else None

            while not incoming.is_complete() and best_bid is not None:
                fills.extend(self._match(incoming, best_bid))
                self._clean_empty_levels()
                best_bid = self.book.best_bid()

        if fills != [] and expected_price is not None:
            self.metrics["total_slippage"] += self._calc_slippage(expected_price, fills)

        return fills

    def _process_cancel(self, event: Event) -> None:
        """Process a cancellation event."""

        cancelled = self.book.cancel_order(event.order_id)
        if cancelled:
            self.metrics["cancel_count"] += 1

    def _match(self, incoming: IncomingOrder, level: PriceLevel) -> list[dict]:
        """Match <incoming> against the FIFO queue at <level>.

        Produce fill records until either the incoming order is complete or the
        price level is exhausted.
        """

        fills = []

        while not (incoming.is_complete() or level.is_empty()):
            maker = level.peek_order()
            filled_qty = min(incoming.remaining_qty, maker.remaining_qty)

            incoming.fill(filled_qty)
            maker.fill(filled_qty)
            level.update_volume(-filled_qty)

            fill_record = self._build_fill_record(maker, incoming, filled_qty, level.price)
            fills.append(fill_record)
            self.execution_log.append(fill_record)
            self.book.log_trade(fill_record)

            self.metrics["total_filled"] += filled_qty
            self.metrics['fill_count'] += 1

            if maker.is_complete():
                level.pop_order()
                self.book.order_index.pop(maker.order_id, None)

        return fills

    @staticmethod
    def _build_fill_record(maker: Order, taker: IncomingOrder, quantity: float,
                           price: float) -> dict:
        """Return a fill record for a single maker-taker execution."""

        return {
            "timestamp": taker.timestamp.isoformat(),
            "maker_order_id": maker.order_id,
            "taker_order_id": taker.order_id,
            "side": taker.side,
            "filled_qty": quantity,
            "exec_price": price,
            "remaining_qty": taker.remaining_qty,
        }

    @staticmethod
    def _calc_slippage(expected_price: float, fills: list[dict]) -> float:
        """Return the slippage for a batch of fills."""

        if not fills:
            return 0.0

        total_qty = 0.0
        weighted_price_sum = 0.0

        for fill in fills:
            filled_qty = fill["filled_qty"]
            exec_price = fill["exec_price"]

            total_qty += filled_qty
            weighted_price_sum += filled_qty * exec_price

        if total_qty == 0.0:
            return 0.0

        vwap = weighted_price_sum / total_qty
        slippage = vwap - expected_price

        return slippage

    def _clean_empty_levels(self) -> None:
        """Remove any empty PriceLevels left behind after matching."""

        best_bid = self.book.best_bid()
        best_ask = self.book.best_ask()

        while best_bid is not None and best_bid.is_empty():
            self.book.bids.delete(best_bid.price)
            best_bid = self.book.best_bid()

        while best_ask is not None and best_ask.is_empty():
            self.book.asks.delete(best_ask.price)
            best_ask = self.book.best_ask()

    def compute_metrics(self) -> dict:
        """Return the aggregate execution metrics for this engine.

        >>> from datetime import datetime
        >>> book = OrderBook()
        >>> engine = MatchingEngine(book)
        >>> engine.process_event(Event(datetime(2026, 1, 1, 9, 30), 'ask1', 'sell', 'limit', 100.0, 2.0))
        []
        >>> _ = engine.process_event(
        ...     Event(datetime(2026, 1, 1, 9, 30, 1), 'buy1', 'buy', 'limit', 100.0, 1.0)
        ... )
        >>> metrics = engine.compute_metrics()
        >>> (metrics['total_filled'], metrics['fill_count'], metrics['cancel_count'])
        (1.0, 1, 0)
        >>> metrics['average_slippage']
        0.0
        """

        metrics_copy = self.metrics.copy()

        if metrics_copy["fill_count"] == 0:
            metrics_copy["average_slippage"] = 0.0
        else:
            total_slippage = metrics_copy["total_slippage"]
            fill_count = metrics_copy["fill_count"]
            metrics_copy["average_slippage"] = total_slippage / fill_count

        return metrics_copy

    def __repr__(self) -> str:
        """Return a string representation of this engine for debugging."""

        return (
            f"MatchingEngine(total_filled={self.metrics['total_filled']}, "
            f"fill_count={self.metrics['fill_count']}, "
            f"cancel_count={self.metrics['cancel_count']})"
        )


if __name__ == '__main__':
    import doctest
    import python_ta

    doctest.testmod()

    python_ta.check_all(config={
        'extra-imports': ['dataclasses', 'core.order_book', 'core.orders', 'core.price_level', 'doctest', 'python_ta'],
        'allowed-io': [],
        'max-line-length': 120
    })
