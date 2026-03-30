"""Quantyze neural network.

Module Description
==================
PyTorch model (OrderBookNet), training loop (Trainer), and inference wrapper
(Agent) that map order-book features to discrete actions (buy / sell / hold).
Trained on event or feature data derived from training_data.csv; checkpoints
are saved as model.pt for Agent.load-style workflows.

build_features and Agent.observe extract a fixed-size vector from OrderBook
snapshots (spread, depth, mid, imbalance, etc.) for forward passes.

This module does not run the matching engine or Flask API; it consumes book
state and produces logits or action strings.

Copyright Information
===============================

Copyright (c) 2026 Cade McNelly, Nicolas Miranda Cantanhede,
Sahand Samadirand
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from data_loader import DataLoader as QuantyzeDataLoader

if TYPE_CHECKING:
    from order_book import OrderBook


class OrderBookNet(nn.Module):
    """Feed-forward classifier: book features -> logits over {buy, sell, hold}."""

    fc1: nn.Linear
    fc2: nn.Linear
    fc3: nn.Linear
    relu: nn.ReLU
    dropout: nn.Dropout

    def __init__(self, feature_dim: int = 12) -> None:
        """Build layers; ``feature_dim`` matches the length of build_features output."""

        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: Tensor) -> Tensor:
        """Run one forward pass; return raw logits of shape (batch, 3)."""

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)


class Trainer:
    """Training loop, loss, optimiser, and checkpoint I/O for OrderBookNet."""

    model: OrderBookNet
    optimizer: optim.Adam
    criterion: nn.CrossEntropyLoss
    device: torch.device
    history: dict[str, list[float]]

    def __init__(self, model: OrderBookNet) -> None:
        """Attach Adam (lr=1e-3), cross-entropy loss, and move model to CPU/CUDA."""

        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.history = {"train_loss": [], "val_loss": []}

    def train_epoch(self, loader: DataLoader) -> float:
        """One full training pass over ``loader``; return mean batch loss."""

        self.model.train()
        total_loss = 0.0
        total_n = 0

        for x, y in loader:
            x = x.to(self.device).float()
            y = y.to(self.device).long()
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            bs = x.size(0)
            total_loss += loss.item() * bs
            total_n += bs

        return total_loss / max(total_n, 1)

    def validate(self, loader: DataLoader) -> float:
        """Eval-mode pass over ``loader``; return mean validation loss."""

        self.model.eval()
        total_loss = 0.0
        total_n = 0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device).float()
                y = y.to(self.device).long()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                bs = x.size(0)
                total_loss += loss.item() * bs
                total_n += bs

        return total_loss / max(total_n, 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
    ) -> None:
        """Train for ``epochs``; track history and persist the best weights to model.pt."""

        best_val = float("inf")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                self.save("model.pt")

            print(f"Epoch {epoch + 1:03d}/{epochs} | train={train_loss:.6f} | val={val_loss:.6f}")

    def save(self, path: str = "model.pt") -> None:
        """Serialise ``model.state_dict()`` to ``path``."""

        torch.save(self.model.state_dict(), path)

    def load(self, path: str = "model.pt") -> None:
        """Load weights from ``path`` into ``model`` and set eval mode."""

        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()


class Agent:
    """Loads OrderBookNet in eval mode; maps snapshots to an action string."""

    model: OrderBookNet
    action_map: dict[int, str]
    balance: float
    position: float
    pnl_log: list[float]

    def __init__(self, model_path: str = "model.pt") -> None:
        """Construct network, load ``model_path`` if present, init balance/position/pnl_log."""

        self.model = OrderBookNet()
        self.action_map = {0: "buy", 1: "sell", 2: "hold"}
        self.balance = 0.0
        self.position = 0.0
        self.pnl_log = []
        self._model_path = model_path

        self.model.eval()

        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            try:
                state = torch.load(model_path, map_location=torch.device("cpu"))
                self.model.load_state_dict(state)
                self.model.eval()
            except (EOFError, OSError, RuntimeError, ValueError) as exc:
                warnings.warn(
                    f"Could not load checkpoint from {model_path!r}; using an untrained model instead. "
                    f"Reason: {exc}",
                    RuntimeWarning,
                )

    @staticmethod
    def observe(book: OrderBook) -> np.ndarray:
        """Extract the feature vector (spread, depth bands, mid, imbalance, etc.)."""

        return build_features(book)

    def act(self, book: OrderBook) -> str:
        """observe -> forward -> argmax -> action_map string."""

        features = self.observe(book)  # (feature_dim,)
        device = next(self.model.parameters()).device
        x = torch.from_numpy(features).float().unsqueeze(0).to(device)  # (1, feature_dim)
        self.model.eval()

        with torch.no_grad():
            logits = self.model(x)  # (1, 3)
            cls = int(torch.argmax(logits, dim=1).item())

        return self.action_map[cls]

    def step(self, book: OrderBook, fill_price: float) -> dict:
        """Choose action, apply simulated economics, append P&L; return a step record."""

        action = self.act(book)
        qty = 1.0
        if action == "buy":
            self.position += qty
            self.balance -= qty * fill_price
        elif action == "sell":
            self.position -= qty
            self.balance += qty * fill_price
        pnl = self.balance + self.position * fill_price
        self.pnl_log.append(pnl)
        return {
            "action": action,
            "fill_price": float(fill_price),
            "position": float(self.position),
            "balance": float(self.balance),
            "pnl": float(pnl),
        }

    def total_pnl(self) -> float:
        """Sum of ``pnl_log`` entries."""

        return float(sum(self.pnl_log))


def build_features(book: OrderBook) -> np.ndarray:
    """Return the shared 12-D inference feature vector from the current book."""

    snapshot = book.depth_snapshot(levels=2)
    bids = snapshot.get("bids", []) or []
    asks = snapshot.get("asks", []) or []
    return QuantyzeDataLoader._feature_vector_from_levels(bids, asks, 0.0)


def load_agent(path: str) -> Agent:
    """Construct Agent and load weights from ``path`` for inference."""

    return Agent(model_path=path)
