"""Quantyze order and event data classes.

This module contains the core data objects shared across the rest of the
project:
- Event, which represents a raw market instruction
- BaseOrder, which stores the common executable-order fields
- IncomingOrder, which represents a mutable order currently being matched
- Order, which represents a limit order resting in the order book

Copyright Information
===============================

Copyright (c) 2026 Cade McNelly, Nicolas Miranda Cantanhede,
Sahand Samadirand
"""

from datetime import datetime


class Event:
    """A raw market instruction that enters the trading system.

    Instance Attributes:
    - timestamp: the time at which this event occurs
    - order_id: the identifier associated with this event
    - side: whether this event is on the buy side or sell side
    - order_type: the type of order represented by this event
    - price: the order price, or None for non-limit orders
    - quantity: the number of units involved in this event

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
    order_type: str
    price: float | None
    quantity: float

    def __init__(self, timestamp: datetime, order_id: str, side: str, order_type: str,
                 price: float | None, quantity: float) -> None:
        """Initialize this event with the given order data.

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
        """Validate that this event satisfies the required order constraints.

        Preconditions:
        - This Event has been fully initialized.

        Raises:
        - ValueError: If any event attribute is invalid.
        """
        if self.side not in {'buy', 'sell'}:
            raise ValueError(f"Invalid side: {self.side}")
        if self.order_type not in {'limit', 'market', 'cancel'}:
            raise ValueError(f"Invalid order type: {self.order_type}")
        if self.quantity < 0:
            raise ValueError(f"Quantity cannot be negative: {self.quantity}")
        if self.order_type == 'limit' and self.price is None:
            raise ValueError("Limit orders must have a price.")
        if self.order_type != 'limit' and self.price is not None:
            raise ValueError("Only limit orders can have a price.")

    def __repr__(self) -> str:
        """Return a detailed string representation of this event for debugging.

        Preconditions:
        - This Event has been fully initialized.
        """
        return (f"Event(timestamp={self.timestamp.isoformat()}, order_id='{self.order_id}', "
                f"side='{self.side}', order_type='{self.order_type}', "
                f"price={self.price}, quantity={self.quantity})")

    def to_dict(self) -> dict:
        """Return this event as a dictionary of serializable field values.

        Preconditions:
        - This Event has been fully initialized.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "order_id": self.order_id,
            "side": self.side,
            "order_type": self.order_type,
            "price": self.price,
            "quantity": self.quantity
        }


class BaseOrder:
    """Common executable-order state shared by incoming and resting orders.

    Instance Attributes:
    - order_id: the identifier copied from the originating event
    - side: whether this order is a buy or sell order
    - price: the limit price for a limit order, or None for a market order
    - quantity: the original submitted quantity
    - remaining_qty: the quantity that has not yet been filled
    - timestamp: the time at which this order entered the system

    Representation Invariants:
    - self.side in {'buy', 'sell'}
    - self.quantity >= 0
    - self.remaining_qty >= 0
    - self.remaining_qty <= self.quantity
    """

    order_id: str
    side: str
    price: float | None
    quantity: float
    remaining_qty: float
    timestamp: datetime

    def __init__(self, event: Event) -> None:
        """Initialize this order's shared fields from an event."""
        self.order_id = event.order_id
        self.side = event.side
        self.price = event.price
        self.quantity = event.quantity
        self.remaining_qty = event.quantity
        self.timestamp = event.timestamp

    def __repr__(self) -> str:
        """Return a detailed string representation of this order."""

        return (
            f"{self.__class__.__name__}(order_id='{self.order_id}', "
            f"side='{self.side}', price={self.price}, "
            f"quantity={self.quantity}, remaining_qty={self.remaining_qty}, "
            f"timestamp={self.timestamp.isoformat()})"
        )

    def to_dict(self) -> dict:
        """Return this order as a dictionary of serializable values."""

        return {
            "order_id": self.order_id,
            "side": self.side,
            "price": self.price,
            "quantity": self.quantity,
            "remaining_qty": self.remaining_qty,
            "timestamp": self.timestamp.isoformat()
        }

    def fill(self, qty: float) -> float:
        """Apply a fill to this order and reduce its remaining quantity.

        Preconditions:
        - qty >= 0
        - qty <= self.remaining_qty
        """
        if qty < 0 or qty > self.remaining_qty:
            raise ValueError("Fill quantity is outside the valid range.")
        if qty == 0:
            return 0.0

        self.remaining_qty -= qty
        return qty

    def is_complete(self) -> bool:
        """Return whether this order has been fully filled."""

        return self.remaining_qty == 0


class IncomingOrder(BaseOrder):
    """A mutable incoming order used only during matching.

    Instance Attributes:
    - order_type: whether this incoming order is limit or market

    Representation Invariants:
    - self.order_type in {'limit', 'market'}
    - self.order_type != 'limit' or self.price is not None
    """

    order_type: str

    def __init__(self, event: Event) -> None:
        """Initialize this incoming order from a validated limit or market event.

        Preconditions:
        - event.order_type in {'limit', 'market'}
        - event.order_type != 'limit' or event.price is not None
        """
        super().__init__(event)
        self.order_type = event.order_type

    def __repr__(self) -> str:
        """Return a detailed string representation of this incoming order."""

        return (
            f"IncomingOrder(order_id='{self.order_id}', side='{self.side}', "
            f"order_type='{self.order_type}', price={self.price}, "
            f"quantity={self.quantity}, remaining_qty={self.remaining_qty}, "
            f"timestamp={self.timestamp.isoformat()})"
        )

    def to_dict(self) -> dict:
        """Return this incoming order as a dictionary of serializable values."""

        data = super().to_dict()
        data["order_type"] = self.order_type
        return data


class Order(BaseOrder):
    """A single resting limit order stored in the order book.

    Instance Attributes:
    - order_id: the identifier copied from the originating event
    - side: whether this resting order is a buy or sell order
    - price: the limit price at which this order rests
    - quantity: the original submitted quantity
    - remaining_qty: the quantity that has not yet been filled
    - timestamp: the time at which this order entered the book
    - status: the current state of the order in the book

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

    def __init__(self, event: Event) -> None:
        """Initialize this resting order from a validated limit-order event.

        Preconditions:
        - event.order_type == 'limit'
        - event.price is not None
        """
        super().__init__(event)
        self.status = "open"

    def __repr__(self) -> str:
        """Return a detailed string representation of this order for debugging.

        Preconditions:
        - This Order has been fully initialized.
        """
        return (f"Order(order_id='{self.order_id}', side='{self.side}', price={self.price}, "
                f"quantity={self.quantity}, remaining_qty={self.remaining_qty}, "
                f"timestamp={self.timestamp.isoformat()}, status='{self.status}')")

    def to_dict(self) -> dict:
        """Return this order as a dictionary of serializable field values.

        Preconditions:
        - This Order has been fully initialized.
        """
        return {
            "order_id": self.order_id,
            "side": self.side,
            "price": self.price,
            "quantity": self.quantity,
            "remaining_qty": self.remaining_qty,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status
        }

    def fill(self, qty: float) -> float:
        """Apply a fill to this order and update quantity and status accordingly.

        Preconditions:
        - qty >= 0
        - qty <= self.remaining_qty
        """
        if self.status == 'cancelled':
            raise ValueError("Cannot fill a cancelled order.")
        if self.status == 'filled':
            raise ValueError("Cannot fill an order that is already filled.")

        filled_qty = super().fill(qty)

        if self.remaining_qty == 0:
            self.status = 'filled'
        elif self.remaining_qty < self.quantity:
            self.status = 'partially_filled'

        return filled_qty

    def cancel(self) -> None:
        """Mark this order as cancelled so it is no longer active in the book.

        Preconditions:
        - self.status != 'filled'
        """

        if self.status == 'filled':
            raise ValueError

        self.status = 'cancelled'

    def is_complete(self) -> bool:
        """Return whether this order is fully filled or cancelled.

        Preconditions:
        - This Order has been fully initialized.
        """
        return self.status == 'cancelled' or self.remaining_qty == 0


if __name__ == '__main__':
    import doctest
    import python_ta

    doctest.testmod()

    python_ta.check_all(config={
        'extra-imports': ['datetime', 'doctest', 'python_ta'],
        'allowed-io': [],
        'disable': ['too-many-arguments'],
        'max-line-length': 120
    })
