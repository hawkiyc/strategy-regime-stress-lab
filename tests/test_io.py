import tempfile
import unittest
from pathlib import Path

import pandas as pd

from regime_lab import load_market_data, load_strategy_returns


class LoadDataTests(unittest.TestCase):
    def test_load_market_data_normalizes_required_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "market.csv"
            pd.DataFrame(
                {
                    "date": ["2024-01-03", "2024-01-02"],
                    "symbol": ["SPY", "SPY"],
                    "open": [101.0, 100.0],
                    "high": [102.0, 101.0],
                    "low": [100.0, 99.0],
                    "close": [101.5, 100.5],
                    "volume": [1200, 1100],
                }
            ).to_csv(path, index=False)

            data = load_market_data(path)

            self.assertEqual(list(data.columns), ["date", "symbol", "open", "high", "low", "close", "volume"])
            self.assertEqual(data["date"].tolist(), [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")])
            self.assertEqual(data["symbol"].tolist(), ["SPY", "SPY"])

    def test_load_market_data_rejects_missing_required_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "market.csv"
            pd.DataFrame({"date": ["2024-01-02"], "symbol": ["SPY"], "close": [100.0]}).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "market_prices.csv missing required columns"):
                load_market_data(path)

    def test_load_strategy_returns_accepts_optional_columns_and_sorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "returns.csv"
            pd.DataFrame(
                {
                    "date": ["2024-01-03", "2024-01-02"],
                    "strategy_id": ["alpha", "alpha"],
                    "return": [0.02, -0.01],
                    "benchmark": [0.01, -0.02],
                    "exposure": [0.5, 0.4],
                }
            ).to_csv(path, index=False)

            data = load_strategy_returns(path)

            self.assertEqual(data["date"].tolist(), [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")])
            self.assertEqual(data["return"].tolist(), [-0.01, 0.02])
            self.assertIn("benchmark", data.columns)
            self.assertIn("exposure", data.columns)

    def test_load_strategy_returns_rejects_null_returns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "returns.csv"
            pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03"],
                    "strategy_id": ["alpha", "alpha"],
                    "return": [0.01, None],
                }
            ).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "strategy_returns.csv contains null values"):
                load_strategy_returns(path)


if __name__ == "__main__":
    unittest.main()
