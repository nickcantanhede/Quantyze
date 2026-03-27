# Quantyze

Quantyze is a CSC111 project that models a limit order book in pure Python.
The system is structured around two binary search trees of price levels, one
for bids and one for asks, with FIFO order queues at each price.

## Project Status

This repository is currently in active development.

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


## Current Validation

At the moment, the safest lightweight validation command is:

```bash
python3 -m py_compile *.py
```

That checks syntax across the Python modules without claiming the full system is
ready to run end-to-end.
