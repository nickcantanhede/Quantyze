"""Quantyze event loading, dataset building, and LOBSTER annotations.

This module defines the DataLoader class, which is responsible for reading
event data from CSV files, validating the loaded data, creating synthetic
event sequences for experiments and testing, building model-ready feature
and label arrays from either canonical Quantyze CSVs or raw LOBSTER files,
and exposing non-replayable LOBSTER rows as visualization-ready annotations.

Copyright Information
===============================

Copyright (c) 2026 Cade McNelly, Nicolas Miranda Cantanhede,
Sahand Samadirand
"""

from __future__ import annotations

import csv
import os
import re
from datetime import date, datetime, timedelta

import numpy as np

from matching_engine import MatchingEngine
from order_book import OrderBook
from orders import Event


class DataLoader:
    """Load, validate, and generate collections of Event objects.

    Instance Attributes:
    - filepath: the path to the CSV file being loaded, if one is being used
    - events: the list of loaded or synthetically generated events
    - schema: the column names expected or found in the input CSV
    - source_format: one of {'internal', 'lobster', 'synthetic'}, or None if unset
    - raw_rows: raw message rows kept for LOBSTER-backed dataset building
    - raw_orderbook_rows: raw orderbook rows aligned with LOBSTER message rows
    - special_events: JSON-safe annotations for recognized non-replayable
      LOBSTER rows such as cross trades and trading halts
    """

    filepath: str | None
    events: list[Event]
    schema: list[str]
    source_format: str | None
    raw_rows: list[list[str]]
    raw_orderbook_rows: list[list[float]]
    special_events: list[dict]
    _training_features: np.ndarray | None
    _training_labels: np.ndarray | None

    def __init__(self, filepath: str | None = None) -> None:
        """Initialize this data loader with an optional CSV file path.

        Preconditions:
        - filepath is None or filepath != ''
        """
        self.filepath = filepath
        self.events = []
        self.schema = []
        self.source_format = None
        self.raw_rows = []
        self.raw_orderbook_rows = []
        self.special_events = []
        self._training_features = None
        self._training_labels = None

    def load_csv(self) -> list[Event]:
        """Read this loader's CSV file and convert its rows into Event objects.

        This method auto-detects whether the source is a canonical Quantyze CSV
        or a raw LOBSTER message file.
        """
        if self.filepath is None:
            raise ValueError("Filepath cannot be None when loading CSV.")

        with open(self.filepath, newline='') as file:
            reader = csv.reader(file)
            try:
                first_row = next(reader)
            except StopIteration:
                self.schema = []
                self.events = []
                self.raw_rows = []
                self.raw_orderbook_rows = []
                self.special_events = []
                self.source_format = None
                self._reset_training_cache()
                return []

        detected = self._detect_source_format(first_row)
        if detected == 'internal':
            return self._load_internal_csv()

        trade_date = self._infer_trade_date()
        return self._load_lobster_messages(first_row, trade_date, detected)

    @staticmethod
    def _detect_source_format(first_row: list[str]) -> str:
        """Return the source format implied by <first_row>."""
        normalized = [cell.strip().lower() for cell in first_row]

        if normalized == ['timestamp', 'order_id', 'side', 'order_type', 'price', 'quantity']:
            return 'internal'

        if normalized == ['time', 'event type', 'order id', 'size', 'price', 'direction']:
            return 'lobster_header'

        if len(first_row) == 6:
            try:
                float(first_row[0])
                int(float(first_row[1]))
                int(float(first_row[2]))
                int(float(first_row[3]))
                int(float(first_row[4]))
                int(float(first_row[5]))
                return 'lobster'
            except ValueError:
                pass

        raise ValueError("Could not detect whether the file is internal CSV or raw LOBSTER.")

    def _load_internal_csv(self) -> list[Event]:
        """Load a canonical Quantyze CSV with a header row."""
        if self.filepath is None:
            raise ValueError("Filepath cannot be None when loading CSV.")

        events = []
        with open(self.filepath, newline='') as file:
            reader = csv.DictReader(file)
            self.schema = list(reader.fieldnames) if reader.fieldnames is not None else []
            for row in reader:
                events.append(self._row_to_event(row))

        self.events = events
        self.source_format = 'internal'
        self.raw_rows = []
        self.raw_orderbook_rows = []
        self.special_events = []
        self._reset_training_cache()
        return self.events

    def _load_lobster_messages(
        self, first_row: list[str], trade_date: date, detected_format: str
    ) -> list[Event]:
        """Load a raw LOBSTER message file and convert replayable rows into Events.

        Types 6 and 7 are preserved as visualization annotations instead of
        replayable Events.
        """
        if self.filepath is None:
            raise ValueError("Filepath cannot be None when loading CSV.")

        with open(self.filepath, newline='') as file:
            rows = [row for row in csv.reader(file)]

        if detected_format == 'lobster_header':
            data_rows = rows[1:]
        else:
            data_rows = rows

        events = []
        annotations = []
        for row_index, row in enumerate(data_rows):
            event = self._lobster_row_to_event(row, trade_date)
            if event is not None:
                events.append(event)
            annotation = self._lobster_row_to_annotation(row, trade_date, row_index)
            if annotation is not None:
                annotations.append(annotation)

        self.schema = ['Time', 'Event Type', 'Order ID', 'Size', 'Price', 'Direction']
        self.events = events
        self.source_format = 'lobster'
        self.raw_rows = data_rows
        self.raw_orderbook_rows = []
        self.special_events = annotations
        self._reset_training_cache()
        return self.events

    @staticmethod
    def _row_to_event(row: dict[str, str]) -> Event:
        """Convert a single CSV row dictionary into an Event object."""
        try:
            try:
                timestamp = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
            except ValueError:
                timestamp = datetime.fromisoformat(row['timestamp'])

            order_id = row['order_id']
            side = row['side']
            order_type = row['order_type']
            price = float(row['price']) if row['price'] not in {'', None} else None
            quantity = float(row['quantity']) if row['quantity'] not in {'', None} else 0.0

        except KeyError as e:
            raise ValueError(f"Missing required column: {e}") from e
        except ValueError as e:
            raise ValueError(f"Invalid data format: {e}") from e

        event = Event(timestamp, order_id, side, order_type, price, quantity)
        event.validate()
        return event

    def _infer_trade_date(self) -> date:
        """Infer the trading date from this loader's filepath."""
        if self.filepath is None:
            raise ValueError("Cannot infer a trade date without a filepath.")

        match = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(self.filepath))
        if match is None:
            raise ValueError(
                "Could not infer the LOBSTER trade date from the filepath. "
                "Expected a YYYY-MM-DD date in the filename."
            )

        return datetime.fromisoformat(match.group(1)).date()

    @staticmethod
    def _lobster_direction_to_side(direction: int) -> str:
        """Convert a LOBSTER direction code into a resting-order side."""
        if direction == 1:
            return 'buy'
        if direction == -1:
            return 'sell'
        raise ValueError(f"Invalid LOBSTER direction: {direction}")

    @staticmethod
    def _aggressor_side(resting_side: str) -> str:
        """Return the opposite side of a resting order."""
        return 'sell' if resting_side == 'buy' else 'buy'

    @staticmethod
    def _parse_lobster_row(row: list[str], trade_date: date) -> dict:
        """Parse one raw LOBSTER row into a normalized dictionary."""
        if len(row) != 6:
            raise ValueError(f"LOBSTER rows must have exactly 6 columns, got {len(row)}.")

        try:
            time_seconds = float(row[0])
            event_type = int(float(row[1]))
            order_id = str(int(float(row[2])))
            size = float(row[3])
            price_int = int(float(row[4]))
            direction = int(float(row[5]))
        except ValueError as e:
            raise ValueError(f"Invalid LOBSTER row format: {e}") from e

        timestamp = datetime.combine(trade_date, datetime.min.time()) + timedelta(seconds=time_seconds)

        parsed = {
            "timestamp": timestamp,
            "event_type": event_type,
            "order_id": order_id,
            "size": size,
            "price_int": price_int,
            "direction": direction,
        }

        if event_type != 7:
            parsed["resting_side"] = DataLoader._lobster_direction_to_side(direction)
        else:
            parsed["resting_side"] = None

        return parsed

    @staticmethod
    def _lobster_row_to_event(row: list[str], trade_date: date) -> Event | None:
        """Convert one raw LOBSTER row into an Event when replay is well-defined.

        Cross trades and halts are recognized but not converted into replay
        events because they are surfaced as visualization annotations instead.
        """
        parsed = DataLoader._parse_lobster_row(row, trade_date)
        event_type = parsed["event_type"]
        timestamp = parsed["timestamp"]
        order_id = parsed["order_id"]
        size = parsed["size"]
        price_int = parsed["price_int"]
        resting_side = parsed["resting_side"]

        if event_type == 1:
            event = Event(timestamp, order_id, resting_side, 'limit', price_int / 10000, size)
        elif event_type in {2, 3}:
            event = Event(timestamp, order_id, resting_side, 'cancel', None, size)
        elif event_type in {4, 5}:
            event = Event(
                timestamp,
                order_id,
                DataLoader._aggressor_side(resting_side),
                'market',
                None,
                size
            )
        elif event_type in {6, 7}:
            return None
        else:
            raise ValueError(f"Unsupported LOBSTER event type: {event_type}")

        event.validate()
        return event

    @staticmethod
    def _lobster_row_to_annotation(
        row: list[str], trade_date: date, row_index: int
    ) -> dict | None:
        """Convert one raw LOBSTER row into a visualization annotation, if applicable."""
        parsed = DataLoader._parse_lobster_row(row, trade_date)
        event_type = parsed["event_type"]
        timestamp = parsed["timestamp"]
        price_int = parsed["price_int"]
        size = parsed["size"]
        direction = parsed["direction"]
        resting_side = parsed["resting_side"]

        if event_type == 6:
            return {
                "row_index": row_index,
                "timestamp": timestamp.isoformat(),
                "lobster_event_type": 6,
                "kind": "cross_trade",
                "price": price_int / 10000 if price_int > 0 else None,
                "size": size,
                "direction": direction,
                "resting_side": resting_side,
                "aggressor_side": None,
            }

        if event_type == 7:
            if price_int == -1:
                kind = "halt"
            elif price_int == 0:
                kind = "quote_resume"
            elif price_int == 1:
                kind = "trade_resume"
            else:
                kind = "halt"

            return {
                "row_index": row_index,
                "timestamp": timestamp.isoformat(),
                "lobster_event_type": 7,
                "kind": kind,
                "price": None,
                "size": size,
                "direction": direction,
                "resting_side": None,
                "aggressor_side": None,
            }

        return None

    def generate_synthetic(self, scenario: str, n: int = 1000) -> list[Event]:
        """Generate and store n synthetic events for the given market scenario.

        Preconditions:
        - scenario in {'balanced', 'low_liquidity', 'high_volatility'}
        - n >= 0
        """
        if scenario == 'balanced':
            self.events = self._balanced_flow(n)
        elif scenario == 'low_liquidity':
            self.events = self._low_liquidity(n)
        elif scenario == 'high_volatility':
            self.events = self._high_volatility(n)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        self.source_format = 'synthetic'
        self.raw_rows = []
        self.raw_orderbook_rows = []
        self.special_events = []
        self._reset_training_cache()
        return self.events

    @staticmethod
    def _balanced_flow(n: int) -> list[Event]:
        """Return n synthetic limit-order events with balanced buy and sell flow.

        Preconditions:
        - n >= 0
        """
        base_time = datetime.now()
        synthetic_events = []

        for i in range(n):
            timestamp = base_time + timedelta(seconds=i)
            side = 'buy' if i % 2 == 0 else 'sell'
            price = 99.5 if side == 'buy' else 100.5
            event = Event(timestamp, f"order_{i}", side, 'limit', price, 10.0)
            synthetic_events.append(event)

        return synthetic_events

    @staticmethod
    def _low_liquidity(n: int) -> list[Event]:
        """Return n synthetic events representing a thin market with a wide spread.

        Preconditions:
        - n >= 0
        """
        base_time = datetime.now()
        synthetic_events = []

        for i in range(n):
            timestamp = base_time + timedelta(seconds=i * 10)
            side = 'buy' if i % 10 == 0 else 'sell'
            price = 98.0 if side == 'buy' else 102.0
            event = Event(timestamp, f"order_{i}", side, 'limit', price, 5.0)
            synthetic_events.append(event)

        return synthetic_events

    @staticmethod
    def _high_volatility(n: int) -> list[Event]:
        """Return n synthetic events with rapidly changing prices and order types.

        Preconditions:
        - n >= 0
        """
        base_time = datetime.now()
        synthetic_events = []
        current_price = 100.0

        for i in range(n):
            timestamp = base_time + timedelta(seconds=i)
            side = 'buy' if i % 2 == 0 else 'sell'

            if i % 4 == 0:
                current_price += 3.0
            elif i % 4 == 1:
                current_price -= 2.5
            elif i % 4 == 2:
                current_price += 4.0
            else:
                current_price -= 3.5

            if i % 5 == 0:
                event = Event(timestamp, f"volatility_order_{i}", side, 'market', None, 5.0 + (i % 4) * 2.0)
            else:
                event = Event(
                    timestamp,
                    f"volatility_order_{i}",
                    side,
                    'limit',
                    current_price,
                    5.0 + (i % 4) * 2.0
                )
            synthetic_events.append(event)

        return synthetic_events

    def validate(self) -> list[str]:
        """Return a list of data-quality errors found in this loader's events."""
        errors = []
        previous_timestamp = None

        for event in self.events:
            if not isinstance(event, Event):
                errors.append(f"Invalid event type: {event} instead of Event.")
                continue

            if not isinstance(event.order_id, str) or event.order_id.strip() == '':
                errors.append(f"Invalid order_id: {event.order_id}")
            if event.side not in {'buy', 'sell'}:
                errors.append(f"Invalid side: {event.side} in event {event.order_id}")
            if event.order_type not in {'limit', 'market', 'cancel'}:
                errors.append(f"Invalid order_type: {event.order_type} in event {event.order_id}")
            if event.quantity < 0:
                errors.append(f"Negative quantity: {event.quantity} in event {event.order_id}")
            if event.order_type == 'limit' and event.price is None:
                errors.append(f"Limit order with missing price in event {event.order_id}")
            if event.order_type != 'limit' and event.price is not None:
                errors.append(f"Non-limit order with price specified in event {event.order_id}")
            if previous_timestamp is not None and event.timestamp <= previous_timestamp:
                errors.append(f"Timestamps are not in strictly increasing order at event {event.order_id}")

            previous_timestamp = event.timestamp

        return errors

    def build_training_dataset(self, orderbook_path: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return model-ready features and labels for this loader's current source."""
        if self._training_features is not None and self._training_labels is not None:
            return self._training_features, self._training_labels

        if self.filepath is not None and not self.events:
            self.load_csv()

        if self.source_format == 'lobster':
            features, labels = self._build_lobster_training_dataset(orderbook_path)
        elif self.events:
            features, labels = self._build_internal_training_dataset()
        else:
            raise ValueError("No source data is loaded for training dataset construction.")

        self._training_features = features
        self._training_labels = labels
        return features, labels

    def _build_internal_training_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """Build training rows by replaying events through the local engine."""
        if not self.events:
            raise ValueError("No events are available for internal training dataset construction.")

        book = OrderBook()
        engine = MatchingEngine(book)
        feature_rows = []
        mids = []

        for event in self.events:
            engine.process_event(event)
            mid = book.mid_price()
            if mid is None:
                continue
            snapshot = book.depth_snapshot(levels=2)
            feature_rows.append(
                self._feature_vector_from_levels(
                    snapshot.get('bids', []),
                    snapshot.get('asks', []),
                    self._event_side_value(event.side)
                )
            )
            mids.append(mid)

        if not feature_rows:
            raise ValueError("Could not build any valid internal training examples.")

        return np.vstack(feature_rows).astype(np.float32), self._labels_from_mid_sequence(mids)

    def _build_lobster_training_dataset(self, orderbook_path: str | None) -> tuple[np.ndarray, np.ndarray]:
        """Build training rows from aligned raw LOBSTER message and orderbook files."""
        if not self.raw_rows:
            raise ValueError("No raw LOBSTER message rows are loaded.")

        resolved_orderbook = self._resolve_lobster_orderbook_path(orderbook_path)
        orderbook_rows = self._load_lobster_orderbook(resolved_orderbook)

        if len(orderbook_rows) != len(self.raw_rows):
            raise ValueError("LOBSTER message and orderbook files must have the same number of rows.")

        feature_rows = []
        mids = []

        for message_row, book_row in zip(self.raw_rows, orderbook_rows):
            event_type = int(float(message_row[1]))
            if event_type == 7:
                continue

            bids, asks = self._levels_from_lobster_orderbook_row(book_row, levels=2)
            mid = self._mid_from_levels(bids, asks)
            if mid is None:
                continue

            feature_rows.append(
                self._feature_vector_from_levels(
                    bids,
                    asks,
                    self._lobster_event_side_value(message_row)
                )
            )
            mids.append(mid)

        if not feature_rows:
            raise ValueError("Could not build any valid LOBSTER training examples.")

        return np.vstack(feature_rows).astype(np.float32), self._labels_from_mid_sequence(mids)

    def to_feature_matrix(self, orderbook_path: str | None = None) -> np.ndarray:
        """Return the model-ready feature matrix for the current source."""
        features, _ = self.build_training_dataset(orderbook_path)
        return features

    def to_label_vector(self, orderbook_path: str | None = None) -> np.ndarray:
        """Return the model-ready label vector for the current source."""
        _, labels = self.build_training_dataset(orderbook_path)
        return labels

    def get_visualization_annotations(self) -> list[dict]:
        """Return a copy of the LOBSTER-specific visualization annotations."""
        return [annotation.copy() for annotation in self.special_events]

    def _resolve_lobster_orderbook_path(self, explicit_path: str | None) -> str:
        """Return the paired LOBSTER orderbook filepath for this message file."""
        if explicit_path is not None:
            if not os.path.exists(explicit_path):
                raise ValueError(f"LOBSTER orderbook file does not exist: {explicit_path}")
            return explicit_path

        if self.filepath is None:
            raise ValueError("Cannot infer a LOBSTER orderbook path without a message filepath.")

        if '_message_' in self.filepath:
            candidate = self.filepath.replace('_message_', '_orderbook_', 1)
        elif self.filepath.endswith('_message.csv'):
            candidate = self.filepath[:-12] + '_orderbook.csv'
        else:
            raise ValueError(
                "Could not infer the paired LOBSTER orderbook path from the message filepath."
            )

        if not os.path.exists(candidate):
            raise ValueError(f"Inferred LOBSTER orderbook file does not exist: {candidate}")

        return candidate

    def _load_lobster_orderbook(self, orderbook_path: str) -> list[list[float]]:
        """Load and decode the paired LOBSTER orderbook file."""
        rows: list[list[float]] = []

        with open(orderbook_path, newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if not row:
                    continue
                if not self._is_numeric_row(row):
                    continue
                rows.append([float(cell) for cell in row])

        self.raw_orderbook_rows = rows
        return rows

    @staticmethod
    def _is_numeric_row(row: list[str]) -> bool:
        """Return whether every cell in <row> can be parsed as a float."""
        try:
            for cell in row:
                float(cell)
        except ValueError:
            return False
        return True

    @staticmethod
    def _event_side_value(side: str) -> float:
        """Return the signed event-side feature value for an event side string."""
        if side == 'buy':
            return 1.0
        if side == 'sell':
            return -1.0
        return 0.0

    @staticmethod
    def _lobster_event_side_value(message_row: list[str]) -> float:
        """Return the signed event-side feature value for a raw LOBSTER row."""
        event_type = int(float(message_row[1]))
        direction = int(float(message_row[5]))
        resting_side = DataLoader._lobster_direction_to_side(direction)

        if event_type in {1, 2, 3}:
            return DataLoader._event_side_value(resting_side)
        if event_type in {4, 5}:
            return DataLoader._event_side_value(DataLoader._aggressor_side(resting_side))
        return 0.0

    @staticmethod
    def _levels_from_lobster_orderbook_row(
        row: list[float], levels: int = 2
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Return best-first bid and ask levels from one raw LOBSTER orderbook row."""
        asks = []
        bids = []

        for level in range(levels):
            base = 4 * level
            if base + 3 >= len(row):
                break

            ask_price_raw = int(row[base])
            ask_size = float(row[base + 1])
            bid_price_raw = int(row[base + 2])
            bid_size = float(row[base + 3])

            if ask_size > 0.0 and ask_price_raw != 9999999999:
                asks.append((ask_price_raw / 10000, ask_size))
            if bid_size > 0.0 and bid_price_raw != -9999999999:
                bids.append((bid_price_raw / 10000, bid_size))

        return bids, asks

    @staticmethod
    def _mid_from_levels(
        bids: list[tuple[float, float]], asks: list[tuple[float, float]]
    ) -> float | None:
        """Return the midpoint from best-first bid and ask levels, if defined."""
        if not bids or not asks:
            return None
        return (bids[0][0] + asks[0][0]) / 2

    @staticmethod
    def _feature_vector_from_levels(
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        event_side: float,
    ) -> np.ndarray:
        """Return the shared 12-D feature vector from best-first bid/ask levels."""
        best_bid_price = bids[0][0] if len(bids) >= 1 else 0.0
        best_bid_size = bids[0][1] if len(bids) >= 1 else 0.0
        best_ask_price = asks[0][0] if len(asks) >= 1 else 0.0
        best_ask_size = asks[0][1] if len(asks) >= 1 else 0.0

        if best_bid_price > 0.0 and best_ask_price > 0.0:
            spread = best_ask_price - best_bid_price
            mid_price = (best_bid_price + best_ask_price) / 2
        else:
            spread = 0.0
            mid_price = 0.0

        imbalance_denom = best_bid_size + best_ask_size
        if imbalance_denom == 0.0:
            imbalance = 0.0
        else:
            imbalance = (best_bid_size - best_ask_size) / imbalance_denom

        bid_price_2 = bids[1][0] if len(bids) >= 2 else 0.0
        bid_size_2 = bids[1][1] if len(bids) >= 2 else 0.0
        ask_price_2 = asks[1][0] if len(asks) >= 2 else 0.0
        ask_size_2 = asks[1][1] if len(asks) >= 2 else 0.0

        return np.array([
            best_bid_price,
            best_bid_size,
            best_ask_price,
            best_ask_size,
            spread,
            mid_price,
            imbalance,
            bid_price_2,
            bid_size_2,
            ask_price_2,
            ask_size_2,
            event_side,
        ], dtype=np.float32)

    @staticmethod
    def _labels_from_mid_sequence(mids: list[float]) -> np.ndarray:
        """Return labels using the next-changed-mid rule."""
        if not mids:
            raise ValueError("Cannot derive labels from an empty mid-price sequence.")

        next_diff_mid: list[float | None] = [None] * len(mids)
        for i in range(len(mids) - 2, -1, -1):
            if mids[i + 1] != mids[i]:
                next_diff_mid[i] = mids[i + 1]
            else:
                next_diff_mid[i] = next_diff_mid[i + 1]

        labels = []
        for current_mid, future_mid in zip(mids, next_diff_mid):
            if future_mid is None:
                labels.append(2)
            elif future_mid > current_mid:
                labels.append(0)
            else:
                labels.append(1)

        return np.array(labels, dtype=np.int64)

    def _reset_training_cache(self) -> None:
        """Clear any cached training dataset arrays."""
        self._training_features = None
        self._training_labels = None
