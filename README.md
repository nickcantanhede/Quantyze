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
- `training_metrics.json`
- `training_data.csv`
- `latest_model.pt`
- `latest_training_metrics.json`
- `latest_training_data.csv`

Submission dataset package:
- `quantyze_datasets.zip`
- package contents: `sample_internal.csv`, `huge_internal.csv`,
  `dataset_manifest.txt`, `model.pt`, and `training_metrics.json`

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

The current training/data path uses one shared 16-feature schema for both
training and inference. It includes the current level-1 / level-2 snapshot
plus a short one-step history signal:

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
13. one-step best bid price delta
14. one-step best ask price delta
15. one-step mid-price delta
16. one-step imbalance delta

Labels use a fixed 50-event horizon with a +/- $0.01 move threshold:
- `0 = buy`
- `1 = sell`
- `2 = hold`

More concretely:
- if the mid-price 50 events later is more than `0.01` above the current mid,
  the label is `buy`
- if it is more than `0.01` below the current mid, the label is `sell`
- otherwise the label is `hold`

The main training pipeline standardizes features using the training split mean
and standard deviation before fitting the model. The exported
`training_data.csv` artifact contains those normalized 16-D feature rows
together with their labels.

Training uses class-weighted cross-entropy so rare classes, especially `hold`,
contribute more strongly to the loss.

Checkpoint behavior:
- a valid non-empty checkpoint means the ML agent can be activated
- a missing, empty, or invalid checkpoint should be treated as "simulation
  only"
- the public loader contract returns `None` when no valid checkpoint is
  available, so simulation can run safely without ML

After training, Quantyze also writes `training_metrics.json`, which stores the
loss curves, validation accuracy, majority-class baseline, per-class recall,
prediction counts, and confusion matrix.

The shipped saved-state artifacts are:
- `model.pt`
- `training_metrics.json`
- `training_data.csv`

These baseline artifacts are intended to come from `huge_internal.csv`. New
training runs from the menu or the direct CLI should write to:
- `latest_model.pt`
- `latest_training_metrics.json`
- `latest_training_data.csv`

This keeps the packaged saved checkpoint stable for grading while still
letting the TA run a short training example and inspect the new results.

## TA Menu

Running `main.py` with no arguments now opens the TA-facing interactive menu.
This is the primary grading path.

```bash
python3 main.py
```

```powershell
py -3 main.py
```

The menu supports:
- running the default simulation with a saved checkpoint
- running a simulation on a chosen synthetic scenario
- training on the packaged `sample_internal.csv`
- training on a custom CSV path
- viewing both the packaged baseline metrics and any newer training metrics

The submitted dataset zip should be extracted beside `main.py` before using
the sample-training option. The default simulation continues to use `model.pt`
even after new training runs create `latest_model.pt`.

## Useful Commands

Syntax check:

```bash
python3 -m py_compile *.py
```

Run the default synthetic simulation directly:

```bash
python3 main.py --no-ui
```

```powershell
py -3 main.py --no-ui
```

This writes `log.json` in addition to printing the terminal summary.

Train on an internal CSV:

```bash
python3 main.py --train --data sample_internal.csv
```

```powershell
py -3 main.py --train --data sample_internal.csv
```

This writes:
- `latest_model.pt`
- `latest_training_metrics.json`
- `latest_training_data.csv`

It does not overwrite the shipped `model.pt` or `training_metrics.json`.

Train on a raw LOBSTER message file:

```bash
python3 main.py --train --data <lobster_message_csv>
```

```powershell
py -3 main.py --train --data <lobster_message_csv>
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

The current lightweight validation flow is:

```bash
python3 -m py_compile *.py
python3 main.py
```

This confirms:
- Python modules compile cleanly
- the default run opens the interactive TA menu
- the default or scenario simulation still runs end-to-end
- the simulation run writes `log.json` with the execution records from the simulation
- the shipped `model.pt` checkpoint can be loaded during simulation if it is valid
- a short training run writes separate `latest_*` artifacts without replacing the shipped checkpoint
