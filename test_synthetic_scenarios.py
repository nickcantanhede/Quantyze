"""Smoke tests for Quantyze synthetic scenarios and summary metrics."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from data_loader import DataLoader
from main import print_summary
from matching_engine import MatchingEngine
from neural_net import Agent
from order_book import OrderBook


class SyntheticScenarioTests(unittest.TestCase):
    """Replay synthetic scenarios and verify usable demo-market behavior."""

    def _replay_scenario(self, scenario: str, n: int = 1000) -> tuple[dict, OrderBook]:
        """Generate and replay one synthetic scenario."""
        loader = DataLoader()
        events = loader.generate_synthetic(scenario, n)

        self.assertEqual(len(events), n)
        self.assertEqual(loader.validate(), [])

        for index in range(1, len(events)):
            self.assertGreater(events[index].timestamp, events[index - 1].timestamp)

        book = OrderBook()
        engine = MatchingEngine(book)
        for event in events:
            engine.process_event(event)

        return engine.compute_metrics(), book

    def test_balanced_scenario_replay(self) -> None:
        """Balanced should remain an active two-sided market."""
        metrics, book = self._replay_scenario("balanced")

        self.assertGreater(metrics["fill_count"], 0)
        self.assertGreater(metrics["cancel_count"], 0)
        self.assertIsNotNone(book.spread())
        self.assertIsNotNone(book.mid_price())

    def test_low_liquidity_scenario_replay(self) -> None:
        """Low-liquidity should stay active but thinner and wider than balanced."""
        balanced_metrics, balanced_book = self._replay_scenario("balanced")
        metrics, book = self._replay_scenario("low_liquidity")

        self.assertGreater(metrics["fill_count"], 0)
        self.assertGreater(metrics["cancel_count"], 0)
        self.assertIsNotNone(book.spread())
        self.assertIsNotNone(book.mid_price())
        self.assertLess(metrics["fill_count"], balanced_metrics["fill_count"])
        self.assertGreater(book.spread(), balanced_book.spread())

    def test_high_volatility_scenario_replay(self) -> None:
        """High-volatility should remain two-sided and show larger slippage swings."""
        balanced_metrics, _ = self._replay_scenario("balanced")
        metrics, book = self._replay_scenario("high_volatility")

        self.assertGreater(metrics["fill_count"], 0)
        self.assertIsNotNone(book.spread())
        self.assertIsNotNone(book.mid_price())
        self.assertGreater(
            abs(metrics["average_slippage"]),
            abs(balanced_metrics["average_slippage"])
        )


class AgentMetricTests(unittest.TestCase):
    """Validate the corrected user-facing agent P&L semantics."""

    def test_current_pnl_returns_latest_logged_value(self) -> None:
        """Current P&L should reflect the most recent mark-to-market value."""
        agent = Agent(model_path="definitely_missing_checkpoint.pt")
        agent.pnl_log = [12.0, -4.5, 7.25]

        self.assertEqual(agent.current_pnl(), 7.25)
        self.assertEqual(agent.total_pnl(), 7.25)

    def test_current_pnl_defaults_to_zero_without_fills(self) -> None:
        """Current P&L should be 0.0 before any simulated fills occur."""
        agent = Agent(model_path="definitely_missing_checkpoint.pt")
        self.assertEqual(agent.current_pnl(), 0.0)

    def test_print_summary_uses_current_mark_to_market_label(self) -> None:
        """The summary should print the corrected P&L label and value."""
        engine = MatchingEngine(OrderBook())
        agent = Agent(model_path="definitely_missing_checkpoint.pt")
        output_buffer = io.StringIO()

        with redirect_stdout(output_buffer):
            print_summary(engine, agent)

        output = output_buffer.getvalue()
        self.assertIn("Current Mark-to-Market P&L: 0.0", output)


if __name__ == "__main__":
    unittest.main()
