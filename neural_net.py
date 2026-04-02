"""Quantyze neural network.

Module Description
==================
This module contains the PyTorch classifier components used by Quantyze: the
OrderBookNet model, the Trainer, and the Agent wrapper used for
simulation-time inference. It builds fixed-size order-book feature vectors,
trains or saves checkpoints, and converts model predictions into overlay
actions.

Copyright Information
===============================

Copyright (c) 2026 Cade McNelly, Nicolas Miranda Cantanhede,
Sahand Samadirand
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from data_loader import DataLoader as QuantyzeDataLoader
from order_book import OrderBook


def normalize_feature_vector(
    features: np.ndarray,
    feature_mean: np.ndarray | None,
    feature_std: np.ndarray | None
) -> np.ndarray:
    """Return standardized features when normalization statistics are available.

    >>> normalize_feature_vector(
    ...     np.array([2.0, 4.0], dtype=np.float32),
    ...     np.array([1.0, 1.0], dtype=np.float32),
    ...     np.array([1.0, 0.0], dtype=np.float32)
    ... ).tolist()
    [1.0, 3.0]
    """
    if feature_mean is None or feature_std is None:
        return features.astype(np.float32, copy=False)

    safe_std = np.where(feature_std == 0.0, 1.0, feature_std).astype(np.float32, copy=False)
    return ((features - feature_mean) / safe_std).astype(np.float32, copy=False)


def _load_checkpoint_payload(
    path: str,
    map_location: torch.device | str
) -> tuple[dict[str, Tensor], np.ndarray | None, np.ndarray | None, int]:
    """Return model weights and optional normalization stats from ``path``."""
    raw_state = torch.load(path, map_location=map_location)

    if isinstance(raw_state, dict) and "state_dict" in raw_state:
        state_dict = raw_state["state_dict"]
        feature_mean = raw_state.get("feature_mean")
        feature_std = raw_state.get("feature_std")
        mean_array = None if feature_mean is None else np.asarray(feature_mean, dtype=np.float32)
        std_array = None if feature_std is None else np.asarray(feature_std, dtype=np.float32)
        feature_dim = int(raw_state.get("feature_dim", state_dict["fc1.weight"].shape[1]))
        return state_dict, mean_array, std_array, feature_dim

    feature_dim = int(raw_state["fc1.weight"].shape[1])
    return raw_state, None, None, feature_dim


class OrderBookNet(nn.Module):
    """Feed-forward classifier: book features -> logits over {buy, sell, hold}."""

    fc1: nn.Linear
    fc2: nn.Linear
    fc3: nn.Linear
    relu: nn.ReLU
    dropout: nn.Dropout

    def __init__(self, feature_dim: int = QuantyzeDataLoader.feature_dim) -> None:
        """Build layers; ``feature_dim`` matches the length of build_features output."""

        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

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

    def __init__(
        self,
        model: OrderBookNet,
        class_weights: Tensor | None = None,
        learning_rate: float = 3e-4
    ) -> None:
        """Attach Adam, optional weighted cross-entropy, and move model to CPU/CUDA."""

        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        if class_weights is not None:
            class_weights = class_weights.to(self.device).float()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
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
        """Train for ``epochs`` and keep the best validation-loss state in memory."""

        best_val = float("inf")
        best_state: dict[str, Tensor] | None = None

        for _ in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def save(
        self,
        path: str = "model.pt",
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None
    ) -> None:
        """Serialise model weights and optional feature normalization stats."""

        payload: dict[str, object] = {"state_dict": self.model.state_dict()}
        if feature_mean is not None:
            payload["feature_mean"] = np.asarray(feature_mean, dtype=np.float32).tolist()
        if feature_std is not None:
            payload["feature_std"] = np.asarray(feature_std, dtype=np.float32).tolist()
        torch.save(payload, path)

    def load(self, path: str = "model.pt") -> None:
        """Load weights from ``path`` into ``model`` and set eval mode."""

        state = _load_checkpoint_payload(path, self.device)[0]
        self.model.load_state_dict(state)
        self.model.eval()


@dataclass
class _PortfolioState:
    """Track the simulated overlay inventory and profit state."""

    balance: float = 0.0
    position: float = 0.0
    pnl_log: list[float] = field(default_factory=list)


@dataclass
class _FeatureState:
    """Track feature normalization and one-step feature history for the agent."""

    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    previous_base: np.ndarray | None = None


class Agent:
    """Loads OrderBookNet in eval mode; maps snapshots to an action string."""

    model: OrderBookNet
    action_map: dict[int, str]
    portfolio: _PortfolioState
    feature_state: _FeatureState
    model_loaded: bool

    def __init__(self, model_path: str = "model.pt") -> None:
        """Construct network, load ``model_path`` if present, init balance/position/pnl_log."""

        self.action_map = {0: "buy", 1: "sell", 2: "hold"}
        self.portfolio = _PortfolioState()
        self.feature_state = _FeatureState()
        self.model_loaded = False

        feature_dim = QuantyzeDataLoader.feature_dim

        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            try:
                state, feature_mean, feature_std, feature_dim = _load_checkpoint_payload(
                    model_path,
                    torch.device("cpu")
                )
                self.model = OrderBookNet(feature_dim=feature_dim)
                self.model.load_state_dict(state)
                self.feature_state.mean = feature_mean
                self.feature_state.std = feature_std
                self.model.eval()
                self.model_loaded = True
            except (EOFError, OSError, RuntimeError, ValueError) as exc:
                warnings.warn(
                    f"Could not load checkpoint from {model_path!r}; using an untrained model instead. "
                    f"Reason: {exc}",
                    RuntimeWarning,
                )
                self.model = OrderBookNet(feature_dim=feature_dim)
                self.model.eval()
        else:
            self.model = OrderBookNet(feature_dim=feature_dim)
            self.model.eval()

    def observe(self, book: OrderBook) -> np.ndarray:
        """Extract the shared inference feature vector including one-step history."""

        base_features = build_base_features(book)
        features = QuantyzeDataLoader.augment_feature_vector(
            base_features,
            self.feature_state.previous_base
        )
        self.feature_state.previous_base = base_features
        return features

    def act(self, book: OrderBook) -> str:
        """observe -> forward -> argmax -> action_map string."""

        features = self.observe(book)  # (feature_dim,)
        features = normalize_feature_vector(
            features,
            self.feature_state.mean,
            self.feature_state.std
        )
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
            self.portfolio.position += qty
            self.portfolio.balance -= qty * fill_price
        elif action == "sell":
            self.portfolio.position -= qty
            self.portfolio.balance += qty * fill_price
        pnl = self.portfolio.balance + self.portfolio.position * fill_price
        self.portfolio.pnl_log.append(pnl)
        return {
            "action": action,
            "fill_price": float(fill_price),
            "position": float(self.portfolio.position),
            "balance": float(self.portfolio.balance),
            "pnl": float(pnl),
        }

    def current_pnl(self) -> float:
        """Return the latest mark-to-market P&L value, or 0.0 if no fills occurred."""

        if self.portfolio.pnl_log == []:
            return 0.0

        return float(self.portfolio.pnl_log[-1])

    def total_pnl(self) -> float:
        """Return the current mark-to-market P&L for backward compatibility."""

        return self.current_pnl()


def build_base_features(book: OrderBook) -> np.ndarray:
    """Return the shared base inference vector from the current book."""

    snapshot = book.depth_snapshot(levels=2)
    bids = snapshot.get("bids", []) or []
    asks = snapshot.get("asks", []) or []
    return QuantyzeDataLoader.feature_vector_from_levels(bids, asks, 0.0)


def build_features(
    book: OrderBook,
    previous_base_features: np.ndarray | None = None
) -> np.ndarray:
    """Return the shared history-aware inference vector from the current book."""

    base_features = build_base_features(book)
    return QuantyzeDataLoader.augment_feature_vector(base_features, previous_base_features)


def load_agent(path: str) -> Agent | None:
    """Construct Agent and load weights from ``path`` for inference."""

    if not os.path.exists(path):
        return None

    if os.path.getsize(path) == 0:
        return None

    agent = Agent(model_path=path)

    if not agent.model_loaded:
        return None

    return agent


if __name__ == '__main__':
    import doctest
    import python_ta

    doctest.testmod()

    python_ta.check_all(config={
        'extra-imports': [
            'dataclasses', 'os', 'warnings', 'numpy', 'torch',
            'torch.utils.data', 'data_loader', 'order_book', 'doctest', 'python_ta'
        ],
        'allowed-io': [
            '_load_checkpoint_payload', 'Trainer.save', 'Trainer.load',
            'Agent.__init__', 'load_agent'
        ],
        'max-line-length': 120
    })
