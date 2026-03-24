"""Quantyze order and event data classes.

This module contains the core data objects shared across the rest of the
project: Event, which represents a raw market instruction, and Order, which
represents an order resting in the order book.

Copyright Information
===============================

Copyright (c) 2026 Cade McNelly, Nicolas Miranda Cantanhede,
Sahand Samadirand
"""

from datetime import datetime


class Event:
    """A raw market instruction that enters the trading system.

    Representation Invariants:
    - self.side in {'buy', 'sell'}
    - self.order_type in {'limit', 'market', 'cancel'}
    - self.quantity >= 0
    - self.order_type != 'limit' or self.price is not None
    - self.order_type == 'limit' or self.price is None
    """
    timestamp: datetime
    order_id: str
    side: str
    order_type: str # market, limit, cancel
    price: float | None
    quantity: float

    def __init__(self, timestamp: datetime, order_id: str, side: str, order_type: str,
                 price: float | None, quantity: float):
        """Initialize a new event with the given market order data.

        Preconditions:
        - side in {'buy', 'sell'}
        - order_type in {'limit', 'market', 'cancel'}
        - quantity >= 0
        - order_type != 'limit' or price is not None
        - order_type == 'limit' or price is None
        """
        self.timestamp = timestamp
        self.order_id = order_id
        self.side = side
        self.order_type = order_type
        self.price = price
        self.quantity = quantity

    def validate(self) -> None:
        """Validate this event's data against its invariants.

        Preconditions:
        - This Event has been fully initialized.

        Raises:
        - ValueError: If any event attribute is invalid.
        """
        pass
    
    def __repr__(self) -> str:
        """Return a developer-friendly string representation of this event.

        Preconditions:
        - This Event has been fully initialized.
        """
        pass


    def to_dict(self) -> dict:
        """Return this event as a JSON-safe dictionary.

        Preconditions:
        - This Event has been fully initialized.
        """
        pass


class Order:
    """A single resting order stored in the order book.

    Representation Invariants:
    - self.side in {'buy', 'sell'}
    - self.quantity >= 0
    - self.remaining_qty >= 0
    - self.remaining_qty <= self.quantity
    - self.status in {'open', 'partially_filled', 'filled', 'cancelled'}
    - self.price is not None
    """
    order_id: str
    side: str
    price: float
    quantity: float
    remaining_qty: float
    timestamp: datetime
    status: str


    def __init__(self, event: Event):
        """Initialize a resting order from the given event.

        Preconditions:
        - event.order_type == 'limit'
        - event.price is not None
        """
        self.order_id = event.order_id
        self.side = event.side
        self.price = event.price
        self.quantity = event.quantity
        self.remaining_qty = event.quantity
        self.timestamp = event.timestamp
        self.status = "open"

    def __repr__(self) -> str:
        """Return a developer-friendly string representation of this order.

        Preconditions:
        - This Order has been fully initialized.
        """
        pass 

    
    def to_dict(self) -> dict:
        """Return this order as a JSON-safe dictionary.

        Preconditions:
        - This Order has been fully initialized.
        """
        pass


    def fill(self, qty: float) -> float:
        """Record a fill on this order and update its remaining quantity.

        Preconditions:
        - qty >= 0
        - qty <= self.remaining_qty
        """
        pass

    def cancel(self) -> None:
        """Mark this order as cancelled.

        Preconditions:
        - self.status != 'filled'
        """
        pass            

    def is_complete(self) -> bool:
        """Return whether this order is no longer active in the book.

        Preconditions:
        - This Order has been fully initialized.
        """
        pass

    
