"""Quantyze order book.

Module Description
==================
This module defines OrderBook: the live limit order book that coordinates a bid
BookTree and an ask BookTree. It routes resting limit orders to the correct side,
maintains an order_id → Order map for O(1) cancellation lookup, holds execution
records for logging, and exposes best bid/ask, spread, mid-price, and depth
snapshots to the matching engine and UI.

The book does not perform matching; MatchingEngine consumes OrderBook state and
mutates it through the public API.

Copyright Information
===============================

Copyright (c) 2026 Cade McNelly, Nicolas Miranda Cantanhede,
Sahand Samadirand
"""
from book_tree import BookTree
from orders import Order
from price_level import PriceLevel

import json


class OrderBook:
    """Full limit order book: bid BST, ask BST, order index, and trade log.

    Attributes:
        bids: BST of bid-side PriceLevels; best bid is the maximum price in the tree.
        asks: BST of ask-side PriceLevels; best ask is the minimum price in the tree.
        order_index: Maps order_id to Order for fast cancellation lookup.
        trade_log: Running list of execution records; flushed to disk as log.json.
    """

    bids: BookTree
    asks: BookTree
    order_index: dict[str, Order]
    trade_log: list[dict]

    def __init__(self):
        """Create bid and ask BookTrees and empty order_index and trade_log."""

        self.bids = BookTree('bid')
        self.asks = BookTree('ask')
        self.order_index = {}
        self.trade_log = []

    def __repr__(self) -> str:
        """Show best bid, best ask, and spread in a compact debug string."""

        return f"{self.__class__.__name__}({self.bids}, {self.asks})"

    def add_limit_order(self, order: Order) -> None:
        """Insert the order into the bid or ask BST at its limit price and register it in order_index."""

        price_level = PriceLevel(order.price)
        price_level.add_order(order)

        self.order_index[order.order_id] = order

        if order.side == 'buy':
            if self.bids.is_empty() or order.price not in self.bids:
                self.bids.insert(price_level)
            else:
                self.bids[order.price].add_order(order)
        else:
            if self.asks.is_empty() or order.price not in self.asks:
                self.asks.insert(price_level)
            else:
                self.asks[order.price].add_order(order)

    def cancel_order(self, order_id: str) -> bool:
        """Look up order_id in order_index, mark the order cancelled, remove it from
        its PriceLevel queue, and prune empty price nodes. Return whether the
        cancellation succeeded.
        """

        if order_id not in self.order_index:
            return False

        order = self.order_index[order_id]

        if order.side == "buy":
            if order.price not in self.bids:
                return False

            removed = self.bids[order.price].pop_order_id(order_id)
            if removed is None:
                return False

            removed.cancel()

            if self.bids[order.price].is_empty():
                self.bids.delete(order.price)
        else:
            if order.price not in self.asks:
                return False

            removed = self.asks[order.price].pop_order_id(order_id)
            if removed is None:
                return False

            removed.cancel()

            if self.asks[order.price].is_empty():
                self.asks.delete(order.price)

        self.order_index.pop(order_id, None)
        return True

    def best_bid(self) -> PriceLevel | None:
        """Return the PriceLevel at the best (highest) bid price, or None if empty."""

        return self.bids.best()

    def best_ask(self) -> PriceLevel | None:
        """Return the PriceLevel at the best (lowest) ask price, or None if empty."""

        return self.asks.best()

    def spread(self) -> float | None:
        """Return best_ask.price minus best_bid.price, or None if either side is missing."""

        return self.best_ask().price - self.best_bid().price \
            if self.best_bid() is not None and self.best_ask() is not None \
            else None

    def mid_price(self) -> float | None:
        """Return the arithmetic midpoint of best bid and best ask, or None if undefined."""

        best_bid = self.best_bid()
        best_ask = self.best_ask()

        if best_bid is None or best_ask is None:
            return None

        return (best_bid.price + best_ask.price) / 2

    def depth_snapshot(self, levels: int = 10) -> dict[str, list]:
        """Return a depth-of-book snapshot with up to ``levels`` rows per side.

        Shape is ``{'bids': [...], 'asks': [...]}`` with each side a list of
        ``(price, volume)`` tuples suitable for charts and API responses.
        """

        bid_nodes = self.bids.inorder()
        ask_nodes = self.asks.inorder()

        bid_levels = [(node.price, node.volume) for node in reversed(bid_nodes[:])]
        ask_levels = [(node.price, node.volume) for node in ask_nodes]

        return {
            "bids": bid_levels[:levels],
            "asks": ask_levels[:levels]
        }

    def log_trade(self, record: dict) -> None:
        """Append one execution record dict to the in-memory trade_log."""

        self.trade_log.append(record)

    def flush_log(self, path: str) -> None:
        """Serialize trade_log to ``path`` (e.g. log.json) and clear the in-memory log."""

        with open(path, 'w') as f:
            json.dump(self.trade_log, f)

        self.trade_log.clear()
