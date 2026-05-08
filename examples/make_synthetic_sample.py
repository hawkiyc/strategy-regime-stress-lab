from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parent
    dates = pd.bdate_range("2023-01-02", periods=90)
    market_rows = []
    for symbol, phase in [("SPY", 0.0), ("QQQ", 0.4), ("IWM", 0.8), ("TLT", 1.2), ("GLD", 1.6), ("USO", 2.0)]:
        close = 100.0
        for i, date in enumerate(dates):
            stress = -0.025 if 55 <= i <= 62 and symbol in {"SPY", "QQQ", "IWM"} else 0.0
            daily_return = 0.0005 + 0.004 * np.sin(i / 7 + phase) + stress
            close *= 1.0 + daily_return
            market_rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "open": round(close * 0.995, 4),
                    "high": round(close * 1.012, 4),
                    "low": round(close * 0.988, 4),
                    "close": round(close, 4),
                    "volume": int(1_000_000 + 10_000 * np.cos(i / 3 + phase)),
                }
            )

    strategy_return = 0.0008 + 0.006 * np.cos(np.arange(len(dates)) / 5)
    strategy_return[55:63] -= 0.018
    strategy = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "strategy_id": "alpha_demo",
            "return": strategy_return.round(6),
            "benchmark": (0.0004 + 0.005 * np.sin(np.arange(len(dates)) / 6)).round(6),
            "exposure": np.clip(0.65 + 0.1 * np.sin(np.arange(len(dates)) / 9), 0.0, 1.0).round(4),
        }
    )

    pd.DataFrame(market_rows).to_csv(root / "sample_market_prices.csv", index=False)
    strategy.to_csv(root / "sample_strategy_returns.csv", index=False)


if __name__ == "__main__":
    main()
