import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from regime_lab import build_report, fit_deep_generator, stress_test


def make_market_csv(path):
    dates = pd.bdate_range("2023-01-02", periods=70)
    rows = []
    for symbol, phase in [("SPY", 0.0), ("QQQ", 0.5), ("TLT", 1.0)]:
        close = 100.0
        for i, date in enumerate(dates):
            close *= 1.0 + 0.002 * np.sin(i / 5 + phase)
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 100_000 + i,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def make_returns_csv(path):
    dates = pd.bdate_range("2023-01-02", periods=70)
    values = 0.001 + 0.008 * np.cos(np.arange(70) / 4)
    pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "strategy_id": "alpha", "return": values}).to_csv(path, index=False)


class ReportCliGeneratorTests(unittest.TestCase):
    def test_build_report_writes_html_with_guardrails(self):
        regimes = pd.DataFrame(
            {
                "date": pd.bdate_range("2023-01-02", periods=50),
                "regime": [0] * 25 + [1] * 25,
                "regime_name": ["steady"] * 25 + ["stress"] * 25,
            }
        )
        returns = pd.DataFrame(
            {
                "date": pd.bdate_range("2023-01-02", periods=50),
                "strategy_id": "alpha",
                "return": 0.001,
            }
        )
        result = stress_test(returns, regimes, n_sims=10, horizon=10, block_size=5, seed=3)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.html"
            build_report(result, output=output)
            html = output.read_text(encoding="utf-8")

        self.assertIn("Strategy Regime Stress Lab", html)
        self.assertIn("research and risk analysis tool", html)
        self.assertNotIn("Buy", html)
        self.assertNotIn("Sell", html)
        self.assertNotIn("target price", html)

    def test_cli_stress_command_creates_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            market = Path(tmp) / "market.csv"
            returns = Path(tmp) / "returns.csv"
            output = Path(tmp) / "report.html"
            make_market_csv(market)
            make_returns_csv(returns)

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "regime_lab.cli",
                    "stress",
                    "--market",
                    str(market),
                    "--returns",
                    str(returns),
                    "--out",
                    str(output),
                    "--n-sims",
                    "12",
                    "--horizon",
                    "10",
                    "--block-size",
                    "5",
                    "--n-regimes",
                    "3",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.exists())
            self.assertIn("Wrote stress report", completed.stdout)

    def test_fit_deep_generator_samples_windows_without_torch(self):
        windows = np.arange(120, dtype=float).reshape(12, 10) / 100.0
        labels = np.array([0, 1] * 6)

        generator = fit_deep_generator(windows, labels, epochs=1, seed=99)
        sample = generator.sample(n=4, regime=1, seed=42)

        self.assertEqual(sample.shape, (4, 10))
        self.assertIn(generator.backend, {"bootstrap", "torch"})

    def test_notebooks_are_valid_json(self):
        root = Path(__file__).resolve().parents[1]
        for name in [
            "01_regime_detection.ipynb",
            "02_strategy_stress_test.ipynb",
            "03_deep_generator_experiment.ipynb",
            "04_colab_deep_generator_training.ipynb",
        ]:
            notebook = root / "notebooks" / name
            data = json.loads(notebook.read_text(encoding="utf-8"))
            self.assertEqual(data["nbformat"], 4)
            self.assertGreater(len(data["cells"]), 0)


if __name__ == "__main__":
    unittest.main()
