"""Quantyze main entry point.

Module Description
==================
This module contains the top-level architecture for Quantyze, responsible for building the
system components, loading or generating data, running the event stream through the matching
engine and print summary of the simulation results.

The main module is responsible for:
- parsing command-line arguments
- constructing the core Quantyze system objects
- selecting between datast replay and synthetic scenarions
- runnig the simulation loop
- trigerring optional ML training or inference flows
- printing final summary information and flushing logs

This module does not implement BST logic, order matching, event validation, or
visualization logic itself. It coordinates the other modules and serves as the
program entry point.

Copyright Information
===============================

Copyright (c) 2026 Cade McNelly, Nicolas Miranda Cantanhede,
Sahand Samadirand
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

from data_loader import DataLoader
from event_stream import EventStream
from matching_engine import MatchingEngine
from neural_net import Agent, Trainer, OrderBookNet
from order_book import OrderBook


def parse_args() -> argparse.Namespace:
    """Return the parsed command-line arguments for running Quantyze.

    The supported arguments determine whether Quantyze loads event data from a
    dataset file or generates a synthetic scenario, whether the optional ML
    path is enabled, and how the simulation should be configured.
    """

    parser = argparse.ArgumentParser(
        prog="Quantyze", description="Run the Quantyze limit order book simulator"
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to a CSV dataset file containing market events"
    )

    parser.add_argument(
        "--scenario",
        type=str,
        choices=["balanced", "low_liquidity", "high_volatility"],
        default=None,
        help="Synthetic scenario to generate if no dataset is provided."
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="Replay speed for the event stream (0.0 = as fast as possible)"
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Run the optional ML training flow instead of the simulation"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Port used for the Flask or UI server"
    )

    parser.add_argument(
        "--no-ui",
        type=str,
        default=None,
        help="Skip the Flask or UI server"
    )

    return parser.parse_args()


def build_system(args: argparse.Namespace) -> tuple[OrderBook, MatchingEngine, EventStream, Agent | None]:
    """Construct and return the core Quantyze system objects.

    This function creates the order book, matching engine, event stream, and
    optional agent needed to run the program from the parsed command-line
    arguments.
    """

    book = OrderBook()
    engine = MatchingEngine(book)
    loader = DataLoader()

    if args.data is not None:
        loader.filepath = args.data
        events = loader.load_csv()
    elif args.scenario is not None:
        events = loader.generate_synthetic(args.scenario)
    else:
        events = loader.generate_synthetic("balanced") # default scenario

    stream = EventStream(events, engine, args.speed)

    if args.train:
        agent = None
    else:
        agent = Agent()

    return book, engine, stream, agent


def run_simulation(stream: EventStream, agent: Agent | None, book: OrderBook) -> None:
    """Run the main Quantyze simulation flow.

    This function processes the event stream through the matching engine and
    allows any optional agent logic to observe or react to the evolving order
    book state.
    """

    if agent is None:
        stream.run_all()
        return

    for event in stream.source:
        fills = stream.emit(event)

        if fills != []:
            last_fill_price = fills[-1]["exec_price"]
            agent.step(book, last_fill_price)


def train_model(data_path: str) -> None:
    """Train the optional OrderBookNet model using data from <data_path>.

    This function loads training data, constructs the neural-network training
    objects, and saves model weights for later inference.
    """

    data_loader = DataLoader(data_path)
    events = data_loader.load_csv()

    if events == []:
        raise ValueError("No training events were loaded from the given data path.")

    features = data_loader.to_feature_matrix()
    labels = data_loader.to_label_vector()

    feature_tensor = torch.tensor(features, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(feature_tensor, label_tensor)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_loader = TorchDataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=32, shuffle=False)

    model = OrderBookNet(feature_dim=features.shape[1])
    trainer = Trainer(model)
    trainer.fit(train_loader, val_loader)
    trainer.save("model.pt")




def start_flask(app, port: int) -> None:
    """Start the Quantyze Flask application on the given port.

    This function is responsible only for launching the API or UI server layer
    after the core system has been initialized.
    """


def print_summary(engine: MatchingEngine, agent: Agent) -> None:
    """Print a summary of the completed Quantyze run.

    The summary may include execution metrics from the matching engine and any
    available ML-related results, such as the agent's total P&L.
    """


def main() -> None:
    """Run the Quantyze program.

    This function parses command-line arguments, builds the system, chooses the
    requested runtime mode, executes the simulation or training flow, and
    prints final summary information.
    """


if __name__ == "__main__":
    main()
