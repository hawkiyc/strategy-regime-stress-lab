import unittest

import numpy as np
import pandas as pd

from regime_lab import fit_regime_model


def make_market_data(days=90):
    dates = pd.bdate_range("2023-01-02", periods=days)
    rows = []
    for symbol, offset in [("SPY", 0.0), ("TLT", 0.2), ("GLD", -0.1)]:
        close = 100.0
        for i, date in enumerate(dates):
            base = 0.001 * np.sin(i / 5 + offset)
            shock = -0.04 if i >= 65 and symbol == "SPY" else 0.0
            close *= 1.0 + base + shock
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 1_000_000 + i,
                }
            )
    return pd.DataFrame(rows)


class RegimeModelTests(unittest.TestCase):
    def test_fit_regime_model_returns_regimes_and_features(self):
        result = fit_regime_model(make_market_data(), n_regimes=3, lookback=10, min_train=20, seed=11)

        self.assertEqual(result.n_regimes, 3)
        self.assertIn("regime", result.regimes.columns)
        self.assertIn("regime_name", result.regimes.columns)
        self.assertIn("market_return_mean", result.features.columns)
        self.assertTrue(set(result.regimes["regime"]).issubset({0, 1, 2}))
        self.assertGreater(len(result.regimes), 0)

    def test_fit_regime_model_rejects_short_history(self):
        with self.assertRaisesRegex(ValueError, "Not enough market history"):
            fit_regime_model(make_market_data(days=12), n_regimes=3, lookback=10, min_train=20, seed=11)

    def test_regime_labels_do_not_change_when_future_data_is_appended(self):
        full = make_market_data(days=90)
        cutoff = pd.Timestamp("2023-03-17")
        truncated = full[full["date"] <= cutoff]

        full_result = fit_regime_model(full, n_regimes=3, lookback=10, min_train=20, seed=5)
        truncated_result = fit_regime_model(truncated, n_regimes=3, lookback=10, min_train=20, seed=5)

        left = full_result.regimes[full_result.regimes["date"] <= cutoff][["date", "regime"]].reset_index(drop=True)
        right = truncated_result.regimes[["date", "regime"]].reset_index(drop=True)

        pd.testing.assert_frame_equal(left, right)


if __name__ == "__main__":
    unittest.main()
