# Quantyze

Quantyze is a CSC111 project that simulates a limit order book in pure Python.
The program uses two binary search trees of price levels, one for bids and one
for asks, to replay market events under price-time-priority matching. On top of
the simulator, Quantyze includes a lightweight neural-network classifier that
can be trained on order-book features and then used as an optional inference
overlay during simulation.

## Quickstart

1. Install the libraries in `requirements.txt`.
2. Extract `quantyze_datasets.zip` beside `main.py`.
3. Run `main.py`.

After extraction, the packaged files that appear beside `main.py` are:
- `sample_internal.csv`
- `huge_internal.csv`
- `instructions.txt`
- `model.pt`
- `training_metrics.json`

The primary TA-facing workflow is the interactive menu opened by running
`main.py` with no arguments.

## What the TA Should Run

Recommended grading path:
- run `main.py`
- open `Artifacts & Metrics -> View active model status`
- open `Simulation -> Run default synthetic demo (balanced)`
- optionally run `low_liquidity` and `high_volatility`
- optionally open `Training` and run either packaged retraining option
- if a new model is trained, choose whether to activate it for future
  simulations

The simulation menu is the main way to experience the project. The training
menu is a separate workflow that writes `latest_*` artifacts and can then hand
that checkpoint off to simulation through the active-model setting.

## Workflow Model

There are three different concepts in the final project:

- `balanced`, `low_liquidity`, and `high_volatility` are synthetic simulation
  sources
- `sample_internal.csv` is the short retraining demo
- `huge_internal.csv` is the larger packaged retraining demo

The shipped baseline model is the packaged checkpoint `model.pt`. New training
runs write:
- `latest_model.pt`
- `latest_training_metrics.json`
- `latest_training_data.csv`

The currently selected simulation checkpoint is stored in `active_model.json`.
It can be:
- `baseline`
- `latest`
- `none`

This means simulation source and model source are intentionally separate. For
example, the program can replay synthetic `balanced` while using either the
packaged baseline checkpoint, the latest trained checkpoint, or no model.

## Important Output Interpretation

Each simulation run prints two kinds of information:

1. Run configuration
- simulation source
- active model mode
- active model path
- model provenance label

2. Matching-engine summary
- `Total Filled`
- `Fill Count`
- `Cancel Count`
- `Average Slippage`
- `Spread`
- `Mid Price`
- `Current Mark-to-Market P&L` if a model is active

The matching-engine metrics describe the event replay itself. The P&L line is a
simple agent overlay metric, not a claim that the model is a profitable trading
strategy.

## Top-Level Files

The final submission tree is intentionally flat. The key files are:
- `main.py`: entry point and top-level orchestration
- `cli_menu.py`: interactive terminal menu
- `data_loader.py`: internal CSV, synthetic scenario, and raw LOBSTER loading
- `matching_engine.py`: price-time-priority matching logic
- `order_book.py`, `book_tree.py`, `price_level.py`: tree-based order book
- `neural_net.py`: model, training loop, and agent
- `server.py`: optional Flask API helper
- `ui.js`: optional browser-side API helper for future UI work
- `project_report.tex`
- `project_report.pdf`
- `quantyze_datasets.zip`
- `requirements.txt`
- `active_model.json`

## Useful Commands

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the interactive menu:

```bash
python3 main.py
```

Run the default simulation directly:

```bash
python3 main.py --no-ui
```

Train on an extracted packaged dataset directly:

```bash
python3 main.py --train --data sample_internal.csv
```

```bash
python3 main.py --train --data huge_internal.csv
```

Train on a custom internal CSV or raw LOBSTER message CSV:

```bash
python3 main.py --train --data <csv_path>
```
