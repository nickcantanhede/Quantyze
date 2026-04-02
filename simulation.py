"""Quantyze simulation runtime.

Module Description
==================
This module contains the shared simulation-runtime helpers used by Quantyze.
It builds the replay system from a selected data source, resolves the active
overlay state, runs simulations, and formats the resulting execution summary.

Copyright Information
===============================

Copyright (c) 2026 Cade McNelly, Nicolas Miranda Cantanhede,
Sahand Samadirand
"""

from __future__ import annotations

from dataclasses import dataclass

from data_loader import DataLoader
from event_stream import EventStream
from matching_engine import MatchingEngine
from neural_net import Agent, load_agent
from order_book import OrderBook

from config import (
    LOG_PATH,
    format_decimal,
    format_metric,
    format_optional_path,
    overlay_mode_text,
    resolve_active_model_status,
)


@dataclass(frozen=True)
class RunArgs:
    """Simple runtime configuration used by the Python entrypoints."""

    data: str | None = None
    scenario: str | None = None
    speed: float = 0.0


def make_run_args(
    *,
    data: str | None = None,
    scenario: str | None = None,
    speed: float = 0.0,
) -> RunArgs:
    """Return a small run-configuration object used by internal flows."""
    return RunArgs(data=data, scenario=scenario, speed=speed)


def simulation_source_label(args: RunArgs) -> str:
    """Return a concise label for the selected simulation source."""
    if args.data is not None:
        return args.data
    if args.scenario is not None:
        return f"synthetic {args.scenario}"
    return "synthetic balanced"


def build_system(
    args: RunArgs,
    model_path: str | None = None,
) -> tuple[OrderBook, MatchingEngine, EventStream, Agent | None, DataLoader]:
    """Construct and return the core Quantyze system objects."""
    book = OrderBook()
    engine = MatchingEngine(book)
    loader = DataLoader()

    if args.data is not None:
        loader.filepath = args.data
        events = loader.load_csv()
    elif args.scenario is not None:
        events = loader.generate_synthetic(args.scenario)
    else:
        events = loader.generate_synthetic("balanced")

    stream = EventStream(events, engine, args.speed)
    if model_path is None:
        agent = None
    else:
        agent = load_agent(model_path)
    return book, engine, stream, agent, loader


def run_simulation(stream: EventStream, agent: Agent | None, book: OrderBook) -> None:
    """Run the main Quantyze simulation flow."""
    if agent is None:
        stream.run_all()
        return

    for event in stream.source:
        fills = stream.emit(event)
        if fills:
            agent.step(book, fills[-1]["exec_price"])


def print_summary(engine: MatchingEngine, agent: Agent | None) -> None:
    """Print a summary of the completed Quantyze run."""
    metrics = engine.compute_metrics()
    spread = engine.book.spread()
    mid_price = engine.book.mid_price()

    print("Quantyze Summary")
    print("=" * 30)
    print("Simulation Metrics")
    print("-" * 30)
    print(f"Total Filled: {format_metric(metrics['total_filled'])}")
    print(f"Fill Count: {format_metric(metrics['fill_count'])}")
    print(f"Cancel Count: {format_metric(metrics['cancel_count'])}")
    print(f"Average Slippage: {format_metric(metrics['average_slippage'])}")

    if spread is None:
        print("Spread: Unavailable")
    else:
        print(f"Spread: {format_decimal(spread)}")

    if mid_price is None:
        print("Mid Price: Unavailable")
    else:
        print(f"Mid Price: {format_decimal(mid_price)}")

    if agent is not None:
        print("-" * 30)
        print("Agent Overlay")
        print("-" * 30)
        print(f"Agent Overlay Mark-to-Market P&L: {format_decimal(agent.current_pnl())}")

    print("=" * 30)


def run_simulation_from_config(args: RunArgs) -> None:
    """Run one simulation from a prepared runtime configuration."""
    active_model_status = resolve_active_model_status()
    model_path = active_model_status["model_path"]
    book, engine, stream, agent, loader = build_system(
        args,
        model_path if isinstance(model_path, str) else None,
    )

    print("Quantyze Run Configuration")
    print("=" * 30)
    print(f"Event Source: {simulation_source_label(args)}")
    print(f"Simulation Overlay Mode: {overlay_mode_text(active_model_status['mode'])}")
    print(f"Simulation Overlay Path: {format_optional_path(active_model_status['model_path'])}")
    print(f"Overlay Provenance: {active_model_status['dataset_label']}")
    print("Overlay Role: optional classifier inference")
    if active_model_status["note"]:
        print(f"Note: {active_model_status['note']}")
    print("=" * 30)

    if agent is None:
        print("Running without a model overlay.")
    else:
        print(f"Loaded simulation overlay checkpoint from {active_model_status['model_path']}.")

    if args.data is not None:
        print(f"Replay dataset path: {args.data}")
        print(f"Detected source format: {loader.source_format}")
        if loader.source_format == "lobster":
            print(
                "Recognized "
                f"{len(loader.special_events)} non-replayable LOBSTER annotations."
            )
    elif args.scenario is not None:
        print(f"Running synthetic scenario: {args.scenario}")
    else:
        print("Running default synthetic scenario: balanced")

    run_simulation(stream, agent, book)
    print_summary(engine, agent)
    if agent is None:
        print("Simulation used no model overlay.")
    else:
        print(f"Simulation used the {active_model_status['mode']} overlay.")

    book.flush_log(LOG_PATH)
    print(f"Execution log written to {LOG_PATH}.")


if __name__ == '__main__':
    import doctest
    import python_ta

    doctest.testmod()

    python_ta.check_all(config={
        'extra-imports': [
            'dataclasses', 'data_loader', 'event_stream', 'matching_engine',
            'neural_net', 'order_book', 'config', 'doctest', 'python_ta'
        ],
        'allowed-io': ['print_summary', 'run_simulation_from_config'],
        'max-line-length': 120
    })
