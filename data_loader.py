"""Quantyze event loading and synthetic-data scaffolds.

This module defines the DataLoader class, which is responsible for reading
event data from CSV files, validating the loaded data, and creating synthetic
event sequences for experiments and testing.

Copyright Information
===============================

Copyright (c) 2026 Cade McNelly, Nicolas Miranda Cantanhede,
Sahand Samadirand
"""

from orders import Event
import numpy as np
import csv
from datetime import datetime, timedelta


class DataLoader:
    """Load, validate, and generate collections of Event objects.

    Instance Attributes:
    - filepath: the path to the CSV file being loaded, if one is being used
    - events: the list of loaded or synthetically generated events
    - schema: the column names expected or found in the input CSV

    Representation Invariants:
    - self.filepath is None or self.filepath != ''
    - all(isinstance(event, Event) for event in self.events)
    - self.schema is not None
    """

    filepath: str | None
    events: list[Event]
    schema: list[str]

    def __init__(self, filepath: str | None = None) -> None:
        """Initialize this data loader with an optional CSV file path.

        Preconditions:
        - filepath is None or filepath != ''
        """
        self.filepath = filepath
        self.events = []
        self.schema: list[str] = []

    def load_csv(self) -> list[Event]:
        """Read this loader's CSV file and convert each row into an Event.

        Preconditions:
        - self.filepath is not None
        """
        if self.filepath is None:
            raise ValueError("Filepath cannot be None when loading CSV.")

        events = []

        with open(self.filepath) as file:
            reader = csv.reader(file)
            try:
                header = next(reader)  # Assume first row is header
            except StopIteration:
                self.schema = []  # No header found, set schema to empty
                self.events = []  # No events to load
                return []  # Return empty list of events

            self.schema = header  # Store column names and types in schema

            for row in reader:
                if len(row) != len(header):
                    raise ValueError(f"Row length {len(row)} does not match header length {len(header)}.")
                else:
                    row_dict = {}
                    for i in range(len(header)):
                        row_dict[header[i]] = row[i]

                    event = self._row_to_event(row_dict)
                    events.append(event)

        self.events = events
        return self.events

    @staticmethod
    def _row_to_event(row: dict) -> Event:
        """Convert a single CSV row dictionary into an Event object.

        Preconditions:
        - row contains the keys needed to build an Event
        """
        try:
            try:
                timestamp = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
            except ValueError:
                timestamp = datetime.fromisoformat(row['timestamp'])
            order_id = row['order_id']
            side = row['side']
            order_type = row['order_type']

            if row['price'] is not None and row['price'] != '':
                price = float(row['price'])
            else:
                price = None

            if row['quantity'] is not None and row['quantity'] != '':
                quantity = float(row['quantity'])
            else:
                quantity = 0.0

        except KeyError as e:
            raise ValueError(f"Missing required column: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid data format: {e}")

        return Event(timestamp, order_id, side, order_type, price, quantity)

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
            time_stamp = base_time + timedelta(seconds=i)

            if i % 2 == 0:
                side = 'buy'
                price = 99.5
            else:
                side = 'sell'
                price = 100.5

            order_id = f"order_{i}"
            order_type = 'limit'
            quantity = 10.0

            event = Event(time_stamp, order_id, side, order_type, price, quantity)
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
            timestamp = base_time + timedelta(seconds=i*10)  # Events are spaced out by 10 seconds

            if i % 10 == 0:
                side = 'buy'
                price = 98.0
            else:
                side = 'sell'
                price = 102.0

            order_id = f"order_{i}"
            order_type = 'limit'
            quantity = 5.0

            event = Event(timestamp, order_id, side, order_type, price, quantity)
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

            if i % 2 == 0:
                side = 'buy'
            else:
                side = 'sell'

            if i % 4 == 0:
                current_price += 3.0
            elif i % 4 == 1:
                current_price -= 2.5
            elif i % 4 == 2:
                current_price += 4.0
            else:
                current_price -= 3.5

            if i % 5 == 0:
                order_type = 'market'
                price = None
            else:
                order_type = 'limit'
                price = current_price

            quantity = 5.0 + (i % 4) * 2.0

            order_id = f"volatility_order_{i}"
            event = Event(timestamp, order_id, side, order_type, price, quantity)
            synthetic_events.append(event)

        return synthetic_events

    def validate(self) -> list[str]:
        """Return a list of data-quality errors found in this loader's events.

        Preconditions:
        - all(isinstance(event, Event) for event in self.events)
        """
        errors = []
        previous_timestamp = None

        for event in self.events:
            if not isinstance(event, Event):  # Check if event is an instance of Event
                errors.append(f"Invalid event type: {event} instead of Event.")
            else:
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

    def to_feature_matrix(self) -> np.ndarray:
        """Convert this loader's events into a numeric NumPy feature matrix.

        Each row has the form:
        [timestamp, side, order_type, price, quantity]

        where side is encoded as 1.0 for buy and 0.0 for sale, and
        order_type is encoded as 0.0 for limit, 1.0 for market, and
        2.0 for cancel. If an event price is None, it is represented as 0.0.

        Preconditions:
        - self.events != []
        """
        if self.events == []:
            raise ValueError("No events to convert to feature matrix.")

        side_mapping = {'buy': 1.0, 'sell': 0.0}
        order_type_mapping = {'limit': 0.0, 'market': 1.0, 'cancel': 2.0}

        feature_matrix = []
        for event in self.events:
            if event.price is None:
                price = 0.0
            else:
                price = event.price

            features = [
                event.timestamp.timestamp(),
                side_mapping[event.side],
                order_type_mapping[event.order_type],
                price,
                event.quantity
            ]
            feature_matrix.append(features)

        return np.array(feature_matrix, dtype=float)

    def to_label_vector(self) -> np.ndarray:
        """Convert this loader's events into a NumPy label vector.
        Each label encodes the target action associated with one event or one derived
        training example, using the project's action-mapping convention for the neural
        network (for example, buy, sell, or hold).

        Preconditions:
        - self.events != []
        - every event in self.events has enough information to determine a target label
        """
        if self.events == []:
            raise ValueError("No events to convert to a label vector.")

        labels = []

        for event in self.events:
            if event.order_type == 'cancel':
                label = 2
            elif event.side == 'buy':
                label = 0
            elif event.side == 'sell':
                label = 1
            else:
                raise ValueError(f"Cannot derive a label from event side: {event.side}")

            labels.append(label)

        return np.array(labels, dtype=int)

