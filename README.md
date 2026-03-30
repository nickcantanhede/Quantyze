# Quantyze

Quantyze is a CSC111 project that models a limit order book in pure Python.
The system is structured around two binary search trees of price levels, one
for bids and one for asks, with FIFO order queues at each price.

## Project Status

This repository is currently in active development.

Current backend status:
- the order book, matching engine, and event replay path compile and run
- the synthetic `balanced` scenario now targets an active market instead of a
  completely static non-crossing book
- the training pipeline can build model-ready datasets, export
  `training_data.csv`
- the main training flow now writes both `training_data.csv` and `model.pt`
- raw LOBSTER message and orderbook files are supported for dataset building

## Repository Layout

The assignment expects a flat top-level layout. The main source files are:

- `orders.py`: `Event` and `Order` data objects
- `price_level.py`: one BST node per price with FIFO order storage
- `book_tree.py`: bid/ask-side BST of `PriceLevel` nodes
- `order_book.py`: coordinates the bid tree, ask tree, order index, and trade log
- `matching_engine.py`: routes events and performs price-time-priority matching
- `data_loader.py`: internal CSV and raw LOBSTER loading, validation, dataset building, and annotation capture
- `event_stream.py`: replay pipeline for emitting events through the engine
- `neural_net.py`: optional ML model, trainer, and agent scaffolds
- `main.py`: entry point for system orchestration
- `ui.js`: front-end or browser-side visualization entry point

Project references:
- `Quantyze_Class_Reference.docx`
- team documentation in the project proposal folder

Generated or runtime artifacts currently present in the repo root:
- `log.json`
- `model.pt`
- `training_data.csv`

## Architecture Summary

The intended runtime flow is:

1. `DataLoader` loads or generates `Event` objects.
2. `EventStream` emits each event into the `MatchingEngine`.
3. `MatchingEngine` routes each event to limit, market, or cancel handling.
4. `OrderBook` coordinates the bid/ask `BookTree` instances.
5. Each `BookTree` stores `PriceLevel` nodes keyed by price.
6. Each `PriceLevel` maintains FIFO time priority for resting orders.
7. Executions are written to in-memory logs and later flushed for analysis.

## LOBSTER Notes

When raw LOBSTER files are used:

- message types `1` to `5` are treated as replay-compatible under the current
  engine assumptions
- message type `6` (cross trade) is recognized, preserved as a visualization
  annotation, and kept in the supervised dataset path
- message type `7` (halt / quote resume / trade resume) is recognized and
  preserved as a visualization annotation, but excluded from replay and
  training examples

This keeps the simulator core limited to clean limit / market / cancel replay
while still exposing richer real-market metadata for plots, notebooks, or a
future UI layer.

## ML and Dataset Notes

The current training/data path uses one shared 12-feature schema for both
training and inference:

1. best bid price
2. best bid size
3. best ask price
4. best ask size
5. spread
6. mid-price
7. level-1 imbalance
8. level-2 bid price
9. level-2 bid size
10. level-2 ask price
11. level-2 ask size
12. event-side feature

Labels use the next-changed-mid rule:
- `0 = buy`
- `1 = sell`
- `2 = hold`

`DataLoader.export_training_csv(...)` writes the exact feature rows and labels
used by the training pipeline to `training_data.csv`.

Checkpoint behavior:
- a valid non-empty checkpoint means the ML agent can be activated
- a missing, empty, or invalid checkpoint should be treated as "simulation
  only"
- the public loader contract returns `None` when no valid checkpoint is
  available, so simulation can run safely without ML

## Useful Commands

Syntax check:

```bash
python3 -m py_compile *.py
```

Run the default synthetic simulation:

```bash
python3 main.py --no-ui
```

Train on an internal CSV:

```bash
python3 main.py --train --data <internal_csv_path>
```

Train on a raw LOBSTER message file:

```bash
python3 main.py --train --data <lobster_message_csv>
```

Export training data manually from Python:

```python
from data_loader import DataLoader

loader = DataLoader()
loader.generate_synthetic("balanced", 100)
loader.export_training_csv("training_data.csv")
```

For raw LOBSTER training, the paired orderbook file is inferred from the
message filename by replacing `_message_` with `_orderbook_`.


## Current Validation

At the moment, the safest lightweight validation command is:

```bash
python3 -m py_compile *.py
```

That checks syntax across the Python modules without claiming the full system is
ready to run end-to-end.
