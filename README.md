# Quantyze

Quantyze is a CSC111 project that simulates a limit order book in pure Python.
The program uses two binary search trees of price levels, one for bids and one
for asks, to replay market events under price-time-priority matching. On top of
the simulator, Quantyze includes a lightweight neural-network classifier that
can be trained on order-book features and then used as an optional inference
overlay during simulation.

`main.py` is the single entry point for the final project. Running it in
PyCharm, or running `python3 main.py`, launches the browser UI directly.

## Quickstart

1. Install the libraries in `requirements.txt`.
2. Extract `quantyze_datasets.zip` beside `main.py`.
3. Run `main.py` in PyCharm or with `python3 main.py`.

After extraction, the packaged files that appear beside `main.py` are:
- `sample_internal.csv`
- `huge_internal.csv`
- `aapl_lobster_2012-06-21_message_5level_sample.csv`
- `aapl_lobster_2012-06-21_orderbook_5level_sample.csv`
- `instructions.txt`
- `model.pt`
- `training_metrics.json`

`quantyze_datasets.zip` is the single TA download artifact included in the
final MarkUs submission. Its size is under the course upload limit, so the TA
does not need any external download link.

## What the TA Should Run

Recommended grading path:
- run `main.py` in PyCharm
- open the printed `http://127.0.0.1:9000` URL
- use the `Simulate` tab for the default synthetic demo
- optionally use the `Train` tab for packaged retraining, including the
  packaged AAPL LOBSTER message file
- use the `Artifacts` and `Charts` tabs to inspect saved status and market
  outputs after a run

The `huge_internal.csv` option and the packaged AAPL LOBSTER option are both
large and can take noticeably longer to load and build training features.

The browser UI is the primary way to experience the project. A terminal-menu
fallback still exists through `main(run_ui=False)`, but the intended TA path is
the browser UI launched from `main.py`.

The `Charts` tab loads `Chart.js` from a CDN, so browser chart rendering
assumes a normal internet connection.

To open the browser frontend:
- run `main.py` in PyCharm, or run `python3 main.py`
- open `http://127.0.0.1:9000` in a browser
- use the `Simulate`, `Train`, `Artifacts`, and `Charts` tabs

If port `9000` is already in use, change the `port` argument in the final
`main(...)` call inside `main.py` before running it again in PyCharm.

## Workflow Model

There are three different concepts in the final project:

- `balanced`, `low_liquidity`, and `high_volatility` are synthetic simulation
  sources
- `sample_internal.csv` is the short retraining demo
- `huge_internal.csv` is the larger packaged retraining demo and takes longer
  to load
- `aapl_lobster_2012-06-21_message_5level_sample.csv` is the packaged raw
  LOBSTER message file, paired with its extracted orderbook file
- `aapl_lobster_2012-06-21_orderbook_5level_sample.csv` is the paired
  packaged LOBSTER orderbook file

The shipped baseline model is the packaged checkpoint `model.pt`. New training
runs write:
- `latest_model.pt`
- `latest_training_metrics.json`
- `latest_training_data.csv`

The currently selected simulation overlay is stored in `active_model.json`.
It can be:
- `baseline`
- `latest`
- `none`

This means simulation source and model source are intentionally separate. For
example, the program can replay synthetic `balanced` while using either the
packaged baseline checkpoint, the latest trained checkpoint, or no model
overlay.

## Important Output Interpretation

Each simulation run prints two kinds of information:

1. Run configuration
- event source
- simulation overlay mode
- simulation overlay path
- overlay provenance label

2. Matching-engine summary
- `Total Filled`
- `Fill Count`
- `Cancel Count`
- `Average Slippage`
- `Spread`
- `Mid Price`
- `Agent Overlay Mark-to-Market P&L` if a model is active

The matching-engine metrics describe the event replay itself. The P&L line is a
simple agent overlay metric, not a claim that the model is a profitable trading
strategy.

## Top-Level Files

The final submission tree is intentionally flat. The key files are:
- `main.py`: small entry point for the browser UI and terminal fallback
- `terminal_menu.py`: interactive terminal menu
- `data_loader.py`: internal CSV, synthetic scenario, and raw LOBSTER loading
- `matching_engine.py`: price-time-priority matching logic
- `order_book.py`, `book_tree.py`, `price_level.py`: tree-based order book
- `neural_net.py`: model, training loop, and agent
- `config.py`: shared constants, dataset presets, and overlay-state logic
- `simulation.py`: simulation runtime helpers
- `training.py`: classifier training and metrics generation
- `api_payloads.py`: JSON serialization helpers for the browser API
- `web.py`: Flask browser UI and API wiring
- `index.html`, `ui.css`, `ui.js`: browser-side files for the web UI
- `project_report.tex`
- `quantyze_datasets.zip`
- `requirements.txt`
- `active_model.json`

The final MarkUs submission should also include the compiled `project_report.pdf`.

## Useful Entrypoints

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the browser UI:

```bash
python3 main.py
```

Then open:

```text
http://127.0.0.1:9000
```

Open the terminal menu from Python:

```bash
python3 -c "import main; main.main(run_ui=False)"
```
