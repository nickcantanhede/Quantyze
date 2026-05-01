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
    
    if event_count == 0:
        return "empty_replay"
    elif bid_depth == 0 or ask_depth == 0 or spread is None:
        return "one_sided_book"
    elif abs(order_flow_imbalance) >= 0.45 and trade_count >= 5:
        return "directional_flow"
    elif abs(depth_imbalance) >= 0.45:
        return "imbalanced_book"
    elif spread >= 1.0 or bid_depth + ask_depth < 20: # Total depth is the sum of bids and asks depths
        return "thin_liquidity"
    elif cancel_rate >= 0.25:
        return "quote_churn"
    elif fill_rate >= 0.35 and trade_count >= max(10, event_count * 0.2):
        return "high_activity"
    else:
        return "balanced_liquidity"



def regime_summary(regime: str) -> str:
    """Return a one-sentence explanation of a market-regime label."""

    if regime == "empty_replay":
        return "No events were replayed, so there is no market state to evaluate."
    elif regime == "one_sided_book":
        return "The final book is missing one side of liquidity."
    elif regime == "directional_flow":
        return "Executions were strongly one-sided."
    elif regime == "imbalanced_book":
        return "Resting liquidity is heavily skewed to one side of the book."
    elif regime == "thin_liquidity":
        return "Liquidity is thin or spreads are wide, so executions are more fragile."
    elif regime == "quote_churn":
        return "Cancellation activity is high relative to total event flow."
    elif regime == "high_activity":
        return "The replay produced frequent executions and active trading conditions."
    elif regime == "balanced_liquidity":
        return "The replay ended with usable two-sided liquidity and balanced market conditions."
    else:
        return "The replay completed, but the market regime is unclassified."






