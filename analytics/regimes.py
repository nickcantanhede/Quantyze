"""Quantyze market-regime interpretation helpers.

Module Description
==================
This module contains the interpretation layer for Quantyze analytics. It takes
raw measurements from the feature helpers and converts them into readable
market-regime labels and short explanatory summaries.

Copyright Information
===============================

Copyright (c) 2026 Nicolas Miranda Cantanhede
"""


def classify_market_regime(
    *,
    event_count: int,
    trade_count: int,
    fill_rate: float,
    cancel_rate: float,
    spread: float | None,
    bid_depth: float,
    ask_depth: float,
    depth_imbalance: float,
    order_flow_imbalance: float,
) -> str:
    """Return a readable market-regime label for a completed simulation."""
    ...


def regime_summary(
    regime: str,
    *,
    order_flow_imbalance: float,
    depth_imbalance: float,
) -> str:
    """Return a one-sentence explanation of a market-regime label."""
    ...
