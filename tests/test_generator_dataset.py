import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from regime_lab import fit_deep_generator
from regime_lab.generator import load_generator, make_return_windows, save_generator


class GeneratorDatasetTests(unittest.TestCase):
    def test_make_return_windows_uses_strategy_returns_and_regime_labels(self):
        dates = pd.bdate_range("2024-01-02", periods=8)
        returns = pd.DataFrame(
            {
                "date": dates,
                "strategy_id": "alpha",
                "return": np.arange(8, dtype=float) / 100.0,
            }
        )
        regimes = pd.DataFrame(
            {
                "date": dates,
                "regime": [0, 0, 0, 1, 1, 1, 1, 1],
                "regime_name": ["steady"] * 3 + ["stress"] * 5,
            }
        )

        dataset = make_return_windows(returns, regimes, window_length=4, stride=2)

        self.assertEqual(dataset.windows.shape, (3, 4))
        self.assertEqual(dataset.labels.tolist(), [0, 1, 1])
        self.assertEqual(dataset.strategy_ids.tolist(), ["alpha", "alpha", "alpha"])
        self.assertEqual(dataset.source.tolist(), ["real", "real", "real"])

    def test_generator_augment_returns_real_and_synthetic_windows(self):
        windows = np.arange(60, dtype=float).reshape(6, 10) / 100.0
        labels = np.array([0, 0, 1, 1, 1, 0])
        generator = fit_deep_generator(windows, labels, seed=3, backend="bootstrap")

        augmented = generator.augment(windows, labels, n_per_regime=2, seed=7)

        self.assertEqual(augmented.windows.shape[1], 10)
        self.assertEqual((augmented.source == "real").sum(), 6)
        self.assertEqual((augmented.source == "synthetic").sum(), 4)
        self.assertEqual(set(augmented.labels.tolist()), {0, 1})

    def test_generator_can_be_saved_and_loaded_after_colab_training(self):
        windows = np.arange(40, dtype=float).reshape(4, 10) / 100.0
        labels = np.array([0, 0, 1, 1])
        generator = fit_deep_generator(windows, labels, seed=11, backend="bootstrap")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deep_generator.pkl"
            save_generator(generator, path)
            loaded = load_generator(path)

        sample = loaded.sample(3, regime=1, seed=5)
        self.assertEqual(sample.shape, (3, 10))
        self.assertEqual(loaded.backend, "bootstrap")

    def test_default_backend_does_not_auto_train_torch(self):
        windows = np.arange(40, dtype=float).reshape(4, 10) / 100.0
        labels = np.array([0, 0, 1, 1])

        generator = fit_deep_generator(windows, labels)

        self.assertEqual(generator.backend, "bootstrap")


if __name__ == "__main__":
    unittest.main()
