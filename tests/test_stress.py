import unittest

import numpy as np
import pandas as pd

from regime_lab import stress_test
from regime_lab.metrics import calculate_metrics


def make_strategy_returns(days=80):
    dates = pd.bdate_range("2023-01-02", periods=days)
    returns = 0.001 + 0.01 * np.sin(np.arange(days) / 4)
    returns[45:50] -= 0.035
    return pd.DataFrame({"date": dates, "strategy_id": "alpha", "return": returns})


def make_regimes(days=80):
    dates = pd.bdate_range("2023-01-02", periods=days)
    regimes = np.where(np.arange(days) < 40, 0, 1)
    return pd.DataFrame({"date": dates, "regime": regimes, "regime_name": np.where(regimes == 0, "steady", "stress")})


class StressTestTests(unittest.TestCase):
    def test_calculate_metrics_handles_constant_positive_returns(self):
        metrics = calculate_metrics(pd.Series([0.001] * 30), periods_per_year=252)

        self.assertAlmostEqual(metrics["volatility"], 0.0)
        self.assertAlmostEqual(metrics["max_drawdown"], 0.0)
        self.assertTrue(np.isnan(metrics["sharpe"]))

    def test_stress_test_is_reproducible_for_same_seed(self):
        first = stress_test(make_strategy_returns(), make_regimes(), n_sims=30, horizon=20, block_size=5, seed=123)
        second = stress_test(make_strategy_returns(), make_regimes(), n_sims=30, horizon=20, block_size=5, seed=123)

        pd.testing.assert_frame_equal(first.summary, second.summary)
        pd.testing.assert_frame_equal(first.simulations, second.simulations)
        pd.testing.assert_frame_equal(first.regime_summary, second.regime_summary)

    def test_stress_test_rejects_short_strategy_history(self):
        with self.assertRaisesRegex(ValueError, "Not enough strategy return history"):
            stress_test(make_strategy_returns(days=4), make_regimes(days=4), n_sims=10, horizon=5, block_size=5, seed=1)

    def test_stress_test_outputs_required_metrics(self):
        result = stress_test(make_strategy_returns(), make_regimes(), n_sims=20, horizon=15, block_size=5, seed=7)

        required = {"cagr", "volatility", "sharpe", "sortino", "max_drawdown", "var_95", "cvar_95", "skew", "kurtosis"}
        self.assertTrue(required.issubset(set(result.summary.columns)))
        self.assertEqual(set(result.simulations["strategy_id"]), {"alpha"})
        self.assertIn("stress", set(result.regime_summary["regime_name"]))


if __name__ == "__main__":
    unittest.main()
