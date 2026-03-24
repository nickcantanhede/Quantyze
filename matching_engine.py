"""
Quantyze matching engine

Module Description
==================
This module contains the matching engine used to process incoming market events
against the live order book for Quantyze. It applies price-time priority by matching
better prices first then enforcing FIFO order withing each level.

The matching engine is responsible for:
- routing income events to the correct processing logic
- matching limit and market orders against the opposite side of the book
- handling cancellations
- producing fill records and execution logs
- updating running execution metrics

This module is responsible only for the matching and execution logic. It does not
implement the BST structure itself, dataset parsing or visualization.


Copyright Information
===============================

Copyright (c) 2026 Cade McNelly, Nicolas Miranda Cantanhede,
Sahand Samadirand
"""

from __future__ import annotations
from dataclasses import dataclass

from order_book import OrderBook
from orders import Event, Order
from price_level import PriceLevel

@dataclass
class MatchingEngine:
    """
    Process events against the live OrderBook

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
        """Process <event> and return the fill records produced."""

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

        incoming = Order(event)
        fills = []

        if incoming.side == "buy":
            best_ask = self.book.best_ask()

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

            while (
                    not incoming.is_complete()
                    and best_bid is not None
                    and incoming.price <= best_bid.price
            ):
                fills.extend(self._match(incoming, best_bid))
                self._clean_empty_levels()
                best_bid = self.book.best_bid()

        if not incoming.is_complete():
            self.book.add_limit_order(incoming)

        return fills

    def _process_market(self, event: Event) -> list[dict]:
        """Process a market-order event.

        Walk the opposite side of the book from the current best price inward
        until the order is fully filled or the book is empty.
        """

        incoming = Order(event)
        fills = []

        if incoming.side == "buy":
            best_ask = self.book.best_ask()

            while not incoming.is_complete() and best_ask is not None:
                fills.extend(self._match(incoming, best_ask))
                self._clean_empty_levels()
                best_ask = self.book.best_ask()
        else:
            best_bid = self.book.best_bid()

            while not incoming.is_complete() and best_bid is not None:
                fills.extend(self._match(incoming, best_bid))
                self._clean_empty_levels()
                best_bid = self.book.best_bid()

        return fills

    def _process_cancel(self, event: Event) -> None:
        """Process a cancellation event."""

        cancelled = self.book.cancel_order(event.order_id)
        if cancelled:
            self.metrics["cancel_count"] += 1

    def _match(self, incoming: Order, level: PriceLevel) -> list[dict]:
        """Match <incoming> against the FIFO queue at <level>.

        Produce fill records until either the incoming order is complete or the
        price level is exhausted.
        """

        fills = []

        while not (incoming.is_complete() or level.is_empty()):
            maker = level.peek_front()

            filled_qty = min(incoming.remaining_qty, maker.remaining_qty)
            incoming.fill(filled_qty)
            maker.fill(filled_qty)

            fill_record = self._build_fill_record(maker, incoming, filled_qty, level.price)
            fills.append(fill_record)
            self.execution_log.append(fill_record)

    def _build_fill_record(self, maker: Order, taker: Order, qty: float,
                           price: float) -> dict:
        """Return a fill record for a single maker-taker execution."""

    def _calc_slippage(self, expected_price: float, fills: list[dict]) -> float:
        """Return the slippage for a batch of fills."""

    def _clean_empty_levels(self) -> None:
        """Remove any empty PriceLevels left behind after matching."""

    def compute_metrics(self) -> dict:
        """Return the aggregate execution metrics for this engine."""

    def __repr__(self) -> str:
        """Return a string representation of this engine for debugging."""




