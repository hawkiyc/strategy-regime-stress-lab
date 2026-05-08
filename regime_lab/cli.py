from __future__ import annotations

import argparse
from pathlib import Path

from regime_lab import build_report, fit_regime_model, load_market_data, load_strategy_returns, stress_test


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="regime-lab", description="Strategy regime stress testing toolkit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stress_parser = subparsers.add_parser("stress", help="Run regime stress testing and write an HTML report.")
    stress_parser.add_argument("--market", required=True, help="Path to market_prices.csv")
    stress_parser.add_argument("--returns", required=True, help="Path to strategy_returns.csv")
    stress_parser.add_argument("--out", default="report.html", help="Output HTML report path")
    stress_parser.add_argument("--n-regimes", type=int, default=4)
    stress_parser.add_argument("--lookback", type=int, default=20)
    stress_parser.add_argument("--min-train", type=int, default=30)
    stress_parser.add_argument("--n-sims", type=int, default=1000)
    stress_parser.add_argument("--horizon", type=int, default=63)
    stress_parser.add_argument("--block-size", type=int, default=5)
    stress_parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)
    if args.command == "stress":
        return _run_stress(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


def _run_stress(args: argparse.Namespace) -> int:
    market = load_market_data(args.market)
    returns = load_strategy_returns(args.returns)
    regimes = fit_regime_model(
        market,
        n_regimes=args.n_regimes,
        lookback=args.lookback,
        min_train=args.min_train,
        seed=args.seed,
    )
    result = stress_test(
        returns,
        regimes,
        n_sims=args.n_sims,
        horizon=args.horizon,
        block_size=args.block_size,
        seed=args.seed,
    )
    output = build_report(result, output=Path(args.out))
    print(f"Wrote stress report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
