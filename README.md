# Quantyze

A limit order book simulator built on real NASDAQ tick data. Feed it market data, watch the order book rebuild event by event, and get execution metrics ( slippage, fill rate, spread, depth ) in your browser.

---

## What it does

- Reconstructs a full limit order book from [LOBSTER](https://lobsterdata.com) tick data using BST + FIFO price-time priority matching
- Simulates realistic order execution: full fills, partial fills, cancellations, hidden orders
- Tracks slippage (VWAP-based), fill rate, spread, and mark-to-market P&L
- Neural net overlay (PyTorch) trained on book features to suggest buy/sell/hold signals
- Interactive browser UI — simulate, train, inspect artifacts, and view market charts

## Stack

Python 3.13 · Flask · PyTorch · Plotly · LOBSTER tick data (AMZN, AAPL — Level 1 & 5)

## Getting started

```bash
git clone https://github.com/your-username/quantyze
cd quantyze
pip install -r requirements.txt
python main.py
```

Open `localhost:9000` in your browser.
