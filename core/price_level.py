"""Quantyze price level.

Module Description
==================
This module contains the PriceLevel class used as the node type inside the
order-book BST. Each price level stores its price key, a FIFO queue of resting
Order objects, aggregate quantity at that price, and left/right links for tree
membership.

The matching engine and book tree treat the front of the queue as the earliest
arrival and therefore the first order matched at that price. This module also
includes the small internal queue node used to maintain O(1) enqueue, dequeue,
and indexed removal within a level.

Copyright Information
===============================

Copyright (c) 2026 Cade McNelly, Nicolas Miranda Cantanhede,
Sahand Samadirand
"""

from __future__ import annotations

from core.orders import Order


class _Node:
    """Doubly linked list node used internally by ``Queue``.

    Instance Attributes:
    - item: the resting order stored at this node
    - prev: the previous node in the queue, or None if this node is first
    - next: the next node in the queue, or None if this node is last

    Representation Invariants:
    - self.prev is None or self.prev.next is self
    - self.next is None or self.next.prev is self
    """

    item: Order
    prev: _Node | None
    next: _Node | None

    def __init__(self, item: Order) -> None:
        """Wrap ``item`` with unset ``prev`` / ``next`` links."""

        self.item = item
        self.prev = None
        self.next = None


class Queue:
    """FIFO queue of resting orders for one price level.

    Instance Attributes:
    - _nodes: maps each stored order_id to its linked-list node
    - _head: the front node of the queue, or None if the queue is empty
    - _tail: the back node of the queue, or None if the queue is empty

    Representation Invariants:
    - self._head is None if and only if self._tail is None
    - all(order_id == self._nodes[order_id].item.order_id for order_id in self._nodes)
    """

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

    Instance Attributes:
    - price: the price key represented by this level
    - orders: the FIFO queue of resting orders at this price
    - volume: the total remaining quantity resting at this price
    - left: the left child in the BST, or None if absent
    - right: the right child in the BST, or None if absent
    - _parent_price: the parent price used only for debugging, or None if unset

    Representation Invariants:
    - self.volume >= 0
    - self.left is None or self.left.price < self.price
    - self.right is None or self.right.price > self.price
    """

    price: float
    orders: Queue
    volume: float
    left: PriceLevel | None
    right: PriceLevel | None
    _parent_price: float | None

    def __init__(self, price: float, orders: Queue | None = None, volume: float = 0.0,
                 parent_price: float | None = None) -> None:
        """Initialise a level at ``price``; allocate a new ``Queue`` when ``orders`` is omitted."""

        self.price = price
        self.orders = orders if orders is not None else Queue()
        self.volume = volume
        self.left = None
        self.right = None
        self._parent_price = parent_price

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
        """Pop and return the earliest order; decrement volume by its remaining_qty.

        >>> from datetime import datetime
        >>> from core.orders import Event
        >>> level = PriceLevel(100.0)
        >>> level.add_order(Order(Event(datetime(2026, 1, 1, 9, 30), 'a1', 'sell', 'limit', 100.0, 2.0)))
        >>> level.add_order(Order(Event(datetime(2026, 1, 1, 9, 30, 1), 'a2', 'sell', 'limit', 100.0, 3.0)))
        >>> (level.volume, level.peek_order().order_id)
        (5.0, 'a1')
        >>> level.pop_order().order_id
        'a1'
        >>> level.volume
        3.0
        """

        try:
            order = self.orders.deque()
        except RuntimeError:
            return None
        self.update_volume(delta=-order.remaining_qty)
        return order

    def pop_order_id(self, order_id: str) -> Order | None:
        """Remove the order for ``order_id`` anywhere in the queue; adjust volume.

        >>> from datetime import datetime
        >>> from core.orders import Event
        >>> level = PriceLevel(100.0)
        >>> level.add_order(Order(Event(datetime(2026, 1, 1, 9, 30), 'a1', 'sell', 'limit', 100.0, 2.0)))
        >>> level.add_order(Order(Event(datetime(2026, 1, 1, 9, 30, 1), 'a2', 'sell', 'limit', 100.0, 3.0)))
        >>> removed = level.pop_order_id('a2')
        >>> removed.order_id
        'a2'
        >>> level.volume
        2.0
        >>> level.peek_order().order_id
        'a1'
        """

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

    def depth_snapshot(self, levels: int) -> list[tuple[float, float]]:
        """Return up to ``levels`` (price, volume) pairs from this subtree.

        The snapshot is returned in ascending price order.

        >>> root = PriceLevel(100.0, volume=3.0)
        >>> root.left = PriceLevel(99.5, volume=2.0)
        >>> root.right = PriceLevel(100.5, volume=4.0)
        >>> root.depth_snapshot(2)
        [(99.5, 2.0), (100.0, 3.0)]
        """

        if levels <= 0:
            return []
        else:
            snapshot = []

            if self.left is not None:
                snapshot.extend(self.left.depth_snapshot(levels))
                if len(snapshot) >= levels:
                    return snapshot[:levels]

            snapshot.append((self.price, self.volume))
            if len(snapshot) >= levels:
                return snapshot[:levels]

            if self.right is not None:
                snapshot.extend(self.right.depth_snapshot(levels - len(snapshot)))

            return snapshot[:levels]


if __name__ == '__main__':
    import doctest
    import python_ta

    doctest.testmod()

    python_ta.check_all(config={
        'extra-imports': ['core.orders', 'doctest', 'python_ta'],
        'allowed-io': [],
        'max-line-length': 120
    })
