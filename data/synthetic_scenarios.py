"""Quantyze synthetic scenarios.

Module Description
==================
This module contains the deterministic synthetic event generators used by
Quantyze for menu demos, replay testing, and classifier experiments. It builds
stable event streams for the packaged ``balanced``, ``low_liquidity``, and
``high_volatility`` scenarios by replaying synthetic orders through a temporary
order book and matching engine.

Copyright Information
===============================

Copyright (c) 2026 Nicolas Miranda Cantanhede
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from core.matching_engine import MatchingEngine
from core.order_book import OrderBook
from core.orders import Event

SYNTHETIC_START_TIME = datetime(2026, 1, 1, 9, 30, 0)
_BALANCED_SEED_ORDERS = (
    ("seed_bid_0", "buy", 99.9, 12.0),
    ("seed_ask_0", "sell", 100.1, 12.0),
    ("seed_bid_1", "buy", 99.8, 10.0),
    ("seed_ask_1", "sell", 100.2, 10.0),
    ("seed_bid_2", "buy", 99.7, 8.0),
    ("seed_ask_2", "sell", 100.3, 8.0),
)


@dataclass
class _SyntheticScenarioBuilder:
    """Build deterministic synthetic event streams while tracking book state.

    Instance Attributes:
    - time_step_seconds: the timestamp gap between successive synthetic events
    - events: the synthetic events built so far
    - book: the temporary book used to track the live synthetic state
    - tracked_order_ids: the live resting order ids tracked by side
    - engine: the temporary matching engine used to replay synthetic events

    Representation Invariants:
    - self.time_step_seconds > 0
    - set(self.tracked_order_ids.keys()) == {'buy', 'sell'}
    """

    time_step_seconds: int = 1
    events: list[Event] = field(default_factory=list)
    book: OrderBook = field(default_factory=OrderBook)
    tracked_order_ids: dict[str, list[str]] = field(
        default_factory=lambda: {"buy": [], "sell": []}
    )
    engine: MatchingEngine = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the temporary matching engine after the book is created."""
        self.engine = MatchingEngine(self.book)

    def _next_timestamp(self) -> datetime:
        """Return the deterministic timestamp for the next synthetic event."""
        return SYNTHETIC_START_TIME + timedelta(
            seconds=len(self.events) * self.time_step_seconds
        )

    def next_order_id(self, prefix: str) -> str:
        """Return a unique synthetic order id using the current event index."""
        return f"{prefix}_{len(self.events)}"

    def _append(self, event: Event) -> Event:
        """Append ``event`` and replay it through the temporary engine."""
        self.events.append(event)
        self.engine.process_event(event)
        return event

    def add_named_limit(
        self,
        order_id: str,
        side: str,
        price: float,
        quantity: float
    ) -> str:
        """Append one limit order with an already-chosen order id."""
        event = Event(self._next_timestamp(), order_id, side, "limit", price, quantity)
        self._append(event)
        if order_id in self.book.order_index:
            self.tracked_order_ids[side].append(order_id)
        return order_id

    def add_limit(self, prefix: str, side: str, price: float, quantity: float) -> str:
        """Append one resting or crossing limit order and return its id."""
        return self.add_named_limit(self.next_order_id(prefix), side, price, quantity)

    def add_market(self, prefix: str, side: str, quantity: float) -> str:
        """Append one market order and return its id."""
        order_id = self.next_order_id(prefix)
        event = Event(self._next_timestamp(), order_id, side, "market", None, quantity)
        self._append(event)
        return order_id

    def add_cancel(self, order_id: str, side: str) -> None:
        """Append one cancellation for ``order_id`` on ``side``."""
        event = Event(self._next_timestamp(), order_id, side, "cancel", None, 0.0)
        self._append(event)

    def cancel_oldest(self, side: str) -> bool:
        """Cancel the oldest still-live resting order on ``side`` if one exists."""
        queue = self.tracked_order_ids[side]
        while queue and queue[0] not in self.book.order_index:
            queue.pop(0)

        if not queue:
            return False

        self.add_cancel(queue.pop(0), side)
        return True

    def best_bid_price(self) -> float | None:
        """Return the current best bid price in the temporary book."""
        best_bid = self.book.best_bid()
        return None if best_bid is None else best_bid.price

    def best_ask_price(self) -> float | None:
        """Return the current best ask price in the temporary book."""
        best_ask = self.book.best_ask()
        return None if best_ask is None else best_ask.price

    def can_submit_market(self, side: str) -> bool:
        """Return whether the opposite side currently has liquidity."""
        if side == "buy":
            return self.best_ask_price() is not None
        return self.best_bid_price() is not None


def generate_synthetic_events(scenario: str, n: int = 1000) -> list[Event]:
    """Return ``n`` synthetic events for ``scenario``.

    >>> events = generate_synthetic_events('balanced', 3)
    >>> len(events)
    3
    >>> (events[0].order_id, events[0].side, events[0].price)
    ('seed_bid_0', 'buy', 99.9)
    >>> events[1].timestamp > events[0].timestamp
    True

    Preconditions:
    - scenario in {'balanced', 'low_liquidity', 'high_volatility'}
    - n >= 0
    """
    generators = {
        "balanced": _balanced_flow,
        "low_liquidity": _low_liquidity,
        "high_volatility": _high_volatility,
    }
    if scenario not in generators:
        raise ValueError(f"Unknown scenario: {scenario}")
    return generators[scenario](n)


def _balanced_cancel_or_refresh(
    builder: _SyntheticScenarioBuilder,
    side: str,
    prefix: str,
    price: float
) -> None:
    """Cancel the oldest order on ``side`` or add a replacement at ``price``."""
    if not builder.cancel_oldest(side):
        builder.add_limit(prefix, side, price, 4.0)


def _seed_balanced_book(builder: _SyntheticScenarioBuilder, n: int) -> None:
    """Seed the balanced scenario with a small ladder near the midpoint."""
    for order_id, side, price, quantity in _BALANCED_SEED_ORDERS:
        if len(builder.events) >= n:
            return
        builder.add_named_limit(order_id, side, price, quantity)


def _balanced_flow(n: int) -> list[Event]:
    """Return ``n`` synthetic events for a stable but active market."""
    builder = _SyntheticScenarioBuilder()
    _seed_balanced_book(builder, n)
    actions = (
        lambda: builder.add_limit("rest_bid", "buy", 99.7, 4.0),
        lambda: builder.add_limit("rest_ask", "sell", 100.3, 4.0),
        lambda: builder.add_limit("cross_buy", "buy", 100.2, 6.0),
        lambda: builder.add_limit("cross_sell", "sell", 99.8, 6.0),
        lambda: builder.add_market("market_buy", "buy", 4.0),
        lambda: builder.add_market("market_sell", "sell", 4.0),
        lambda: _balanced_cancel_or_refresh(builder, "buy", "fallback_bid", 99.7),
        lambda: _balanced_cancel_or_refresh(builder, "sell", "fallback_ask", 100.3),
    )
    remaining_events = max(0, n - len(builder.events))
    for cycle_index in range(remaining_events):
        actions[cycle_index % len(actions)]()
    return builder.events


def _thin_buy_pressure(
    builder: _SyntheticScenarioBuilder,
    cycle_index: int,
    best_bid_price: float
) -> None:
    """Apply a buy-side liquidity-taking or reset action in the thin market."""
    if cycle_index % 3 == 0 and builder.can_submit_market("buy"):
        builder.add_market("thin_market_buy", "buy", 1.0)
        return

    best_ask_price = builder.best_ask_price()
    if best_ask_price is not None:
        builder.add_limit("thin_cross_buy", "buy", best_ask_price, 1.5)
    else:
        builder.add_limit("thin_reset_bid", "buy", best_bid_price, 1.5)


def _thin_sell_pressure(
    builder: _SyntheticScenarioBuilder,
    cycle_index: int,
    best_ask_price: float
) -> None:
    """Apply a sell-side liquidity-taking or reset action in the thin market."""
    if cycle_index % 3 == 1 and builder.can_submit_market("sell"):
        builder.add_market("thin_market_sell", "sell", 1.0)
        return

    best_bid_price = builder.best_bid_price()
    if best_bid_price is not None:
        builder.add_limit("thin_cross_sell", "sell", best_bid_price, 1.5)
    else:
        builder.add_limit("thin_reset_ask", "sell", best_ask_price, 1.5)


def _support_or_restore_bid(
    builder: _SyntheticScenarioBuilder,
    best_bid_price: float
) -> None:
    """Add support on the bid side or restore the best bid if it vanished."""
    if builder.best_bid_price() is None:
        builder.add_limit("thin_end_bid", "buy", best_bid_price, 2.0)
    else:
        builder.add_limit("thin_support_bid", "buy", best_bid_price, 1.0)


def _support_or_restore_ask(
    builder: _SyntheticScenarioBuilder,
    best_ask_price: float
) -> None:
    """Add support on the ask side or restore the best ask if it vanished."""
    if builder.best_ask_price() is None:
        builder.add_limit("thin_end_ask", "sell", best_ask_price, 2.0)
    else:
        builder.add_limit("thin_support_ask", "sell", best_ask_price, 1.0)


def _low_liquidity(n: int) -> list[Event]:
    """Return ``n`` synthetic events for a thin market with a wide spread."""
    builder = _SyntheticScenarioBuilder(time_step_seconds=10)
    best_bid_price = 98.8
    best_ask_price = 101.2
    deep_bid_price = 98.2
    deep_ask_price = 101.8
    for cycle_index in range(n):
        round_index = cycle_index // 10
        actions = (
            lambda: builder.add_limit("thin_bid", "buy", best_bid_price, 2.0),
            lambda: builder.add_limit("thin_ask", "sell", best_ask_price, 2.0),
            lambda: _thin_buy_pressure(builder, round_index, best_bid_price),
            lambda: builder.add_limit("thin_deep_bid", "buy", deep_bid_price, 1.0),
            lambda: builder.add_limit("thin_deep_ask", "sell", deep_ask_price, 1.0),
            lambda: _thin_sell_pressure(builder, round_index, best_ask_price),
            lambda: _balanced_cancel_or_refresh(builder, "buy", "thin_refresh_bid", best_bid_price),
            lambda: _balanced_cancel_or_refresh(builder, "sell", "thin_refresh_ask", best_ask_price),
            lambda: _support_or_restore_bid(builder, best_bid_price),
            lambda: _support_or_restore_ask(builder, best_ask_price),
        )
        actions[cycle_index % len(actions)]()

    return builder.events


def _volatility_buy_step(
    builder: _SyntheticScenarioBuilder,
    reference_mid: float,
    best_ask_price: float
) -> None:
    """Apply the volatility scenario's buy-side aggression step."""
    if builder.can_submit_market("buy"):
        builder.add_market("vol_market_buy", "buy", 3.0)
    else:
        builder.add_limit("vol_reset_ask", "sell", best_ask_price, 3.0)
    builder.add_limit("vol_cross_buy", "buy", round(reference_mid + 2.2, 2), 7.0)


def _volatility_sell_step(
    builder: _SyntheticScenarioBuilder,
    reference_mid: float,
    best_bid_price: float
) -> None:
    """Apply the volatility scenario's sell-side aggression step."""
    builder.add_limit("vol_cross_sell", "sell", round(reference_mid - 2.2, 2), 7.0)
    if builder.can_submit_market("sell"):
        builder.add_market("vol_market_sell", "sell", 3.0)
    else:
        builder.add_limit("vol_reset_bid", "buy", best_bid_price, 3.0)


def _high_volatility(n: int) -> list[Event]:
    """Return ``n`` synthetic events with rapidly changing prices and order types."""
    builder = _SyntheticScenarioBuilder()
    anchor_mid = 100.0
    mid_offsets = (0.0, 1.6, -1.2, 2.4, -1.8, 0.9, -2.1, 1.3)

    for cycle_index in range(n):
        round_index = cycle_index // 12
        reference_mid = anchor_mid + mid_offsets[round_index % len(mid_offsets)]
        best_bid_price = round(reference_mid - 0.7, 2)
        best_ask_price = round(reference_mid + 0.7, 2)
        deep_bid_price = round(reference_mid - 1.5, 2)
        deep_ask_price = round(reference_mid + 1.5, 2)
        support_bid_price = round(reference_mid - 0.4, 2)
        support_ask_price = round(reference_mid + 0.4, 2)
        actions = (
            lambda: builder.add_limit("vol_bid", "buy", best_bid_price, 5.0),
            lambda: builder.add_limit("vol_ask", "sell", best_ask_price, 5.0),
            lambda: builder.add_limit("vol_deep_bid", "buy", deep_bid_price, 4.0),
            lambda: builder.add_limit("vol_deep_ask", "sell", deep_ask_price, 4.0),
            lambda: _volatility_buy_step(builder, reference_mid, best_ask_price),
            lambda: _volatility_sell_step(builder, reference_mid, best_bid_price),
            lambda: _balanced_cancel_or_refresh(builder, "buy", "vol_refresh_bid", best_bid_price),
            lambda: _balanced_cancel_or_refresh(builder, "sell", "vol_refresh_ask", best_ask_price),
            lambda: builder.add_limit("vol_support_bid", "buy", support_bid_price, 2.0),
            lambda: builder.add_limit("vol_support_ask", "sell", support_ask_price, 2.0),
            lambda: builder.add_limit("vol_support_bid", "buy", support_bid_price, 2.0),
            lambda: builder.add_limit("vol_support_ask", "sell", support_ask_price, 2.0),
        )
        actions[cycle_index % len(actions)]()

    return builder.events


if __name__ == '__main__':
    import doctest
    import python_ta

    doctest.testmod()

    python_ta.check_all(config={
        'extra-imports': [
            'dataclasses', 'datetime', 'core.matching_engine', 'core.order_book',
            'core.orders', 'doctest', 'python_ta'
        ],
        'allowed-io': [],
        'max-line-length': 120
    })
