"""Quantyze price level.

Module Description
==================
This module defines PriceLevel, the binary search tree node used for each
distinct price on the bid or ask side. A PriceLevel stores the price key, a FIFO
queue of resting Order objects (time priority within the level), and aggregate
volume at that price.

The matching engine and book tree treat the front of the queue as the earliest
arrival and thus the first matched at that price. Queue supports O(1) enqueue
and dequeue at the ends and O(1) removal by order_id via a side index mapping
ids to doubly linked list nodes.

Copyright Information
===============================

Copyright (c) 2026 Cade McNelly, Nicolas Miranda Cantanhede,
Sahand Samadirand
"""
from __future__ import annotations

from collections import deque
from queue import Queue
from orders import Order


class _Node:
    """Internal doubly linked list node holding one ``Order``."""

    item: Order
    prev: _Node | None
    next: _Node | None

    def __init__(self, item: Order) -> None:
        """Wrap ``item`` with unset ``prev`` / ``next`` links."""

        self.item = item
        self.prev = None
        self.next = None


class Queue:
    """FIFO queue for ``PriceLevel.orders``: head dequeue, tail enqueue, dict for id lookup."""

    _nodes: dict[str, _Node]
    _head: _Node | None
    _tail: _Node | None

    def __init__(self) -> None:
        """Initialise an empty queue and empty order_id → node map."""

        self._nodes = {}
        self._head = None
        self._tail = None

    def __repr__(self) -> str:
        """Return a short debug string with head/tail presence."""

        return f"<{self.__class__.__name__} head={self._head is not None} tail={self._tail is not None}>"

    def is_empty(self) -> bool:
        """True when there are no nodes in the list."""

        return self._head is None

    def enque(self, order: Order) -> None:
        """Append ``order`` at the tail and register its ``order_id`` in the index."""

        node = _Node(order)
        if self._head is None:
            self._head = self._tail = node
        else:
            assert self._tail is not None
            node.prev = self._tail
            self._tail.next = node
            self._tail = node
        self._nodes[order.order_id] = node

    def deque(self) -> Order:
        """Remove and return the front order; raise if empty."""

        if self._head is None:
            raise RuntimeError("FIFO queue is empty")
        node = self._head
        self._head = node.next
        if self._head is not None:
            self._head.prev = None
        else:
            self._tail = None
        del self._nodes[node.item.order_id]
        return node.item

    def peek_front(self) -> Order | None:
        """Return the front order without removing it, or None if empty."""

        return self._head.item if self._head is not None else None

    def peek_id(self, order_id: str) -> Order | None:
        """Return the order for ``order_id`` without removing it, or None if absent."""

        node = self._nodes.get(order_id)
        return node.item if node is not None else None

    def remove_id(self, order_id: str) -> Order | None:
        """Unlink the node for ``order_id`` and return its order, or None if not found."""

        node = self._nodes.pop(order_id, None)
        if node is None:
            return None
        if node.prev is not None:
            node.prev.next = node.next
        else:
            self._head = node.next
        if node.next is not None:
            node.next.prev = node.prev
        else:
            self._tail = node.prev
        return node.item


class PriceLevel:
    """BST node mapping one price to a FIFO queue of resting orders.

    Attributes:
        price: The price this node represents; BST search key.
        orders: FIFO queue; earliest arrival is at the front and matches first.
        volume: Sum of resting size at this level; kept in sync with enqueue/dequeue
            and partial fills via update_volume.
        left: Left child PriceLevel (lower prices) when embedded in a BST.
        right: Right child PriceLevel (higher prices) when embedded in a BST.
    """

    price: float
    orders: Queue
    volume: float
    left: PriceLevel | None
    right: PriceLevel | None

    def __init__(self, price: float, orders: Queue | None = None, volume: float = 0.0) -> None:
        """Initialise a level at ``price``; allocate a new ``Queue`` when ``orders`` is omitted."""

        self.price = price
        self.orders = orders if orders is not None else Queue()
        self.volume = volume
        self.left = None
        self.right = None

    def __repr__(self) -> str:
        """Return a short debug string with price, queue, and level state."""

        return f"<{self.__class__.__name__} price={self.price} orders={self.orders}>"

    def is_empty(self) -> bool:
        """True when there are no resting orders at this level (volume should be zero)."""

        return self.orders.is_empty()

    def update_volume(self, delta: float) -> None:
        """Adjust aggregate ``volume`` by ``delta`` (e.g. after a partial fill)."""

        self.volume += delta

    def add_order(self, order: Order) -> None:
        """Enqueue ``order`` at the back of the FIFO and add its quantity to ``volume``."""

        self.orders.enque(order)
        self.update_volume(delta=order.quantity)

    def pop_order(self) -> Order | None:
        """Pop and return the earliest order; decrement volume by its remaining_qty."""

        try:
            order = self.orders.deque()
        except RuntimeError:
            return None
        self.update_volume(delta=-order.remaining_qty)
        return order

    def pop_order_id(self, order_id: str) -> Order | None:
        """Remove the order for ``order_id`` anywhere in the queue; adjust volume."""

        order = self.orders.remove_id(order_id)
        if order is None:
            return None
        self.update_volume(delta=-order.remaining_qty)
        return order

    def peek_order(self) -> Order | None:
        """Return the front order without dequeuing."""

        return self.orders.peek_front()

    def peek_order_id(self, order_id: str) -> Order | None:
        """Return the order for ``order_id`` without removing it."""

        return self.orders.peek_id(order_id)

    def consume_order_by_volume(self, volume: float) -> None:
        """Consume ``volume`` units from the front of the queue (partial/full pops)."""

        raise NotImplementedError
