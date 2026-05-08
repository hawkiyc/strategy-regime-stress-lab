from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime_lab.metrics import calculate_metrics


@dataclass(frozen=True)
class StressTestResult:
    summary: pd.DataFrame
    simulations: pd.DataFrame
    regime_summary: pd.DataFrame
    crisis_replay: pd.DataFrame
    metadata: dict[str, object]


def stress_test(
    strategy_returns: pd.DataFrame,
    regimes,
    n_sims: int = 1000,
    horizon: int = 63,
    block_size: int = 5,
    seed: int = 42,
    periods_per_year: int = 252,
) -> StressTestResult:
    regime_frame = regimes.regimes if hasattr(regimes, "regimes") else regimes
    aligned = _align_returns_and_regimes(strategy_returns, regime_frame)
    if aligned.empty:
        raise ValueError("No overlapping dates between strategy returns and regimes")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")

    rng = np.random.default_rng(seed)
    summary_rows: list[dict[str, object]] = []
    simulation_rows: list[dict[str, object]] = []
    regime_rows: list[dict[str, object]] = []
    replay_rows: list[dict[str, object]] = []

    for strategy_id, group in aligned.groupby("strategy_id", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        if len(group) < block_size:
            raise ValueError(
                f"Not enough strategy return history for {strategy_id}: need at least {block_size}, got {len(group)}"
            )

        returns = group["return"].astype(float)
        observed_metrics = calculate_metrics(returns, periods_per_year=periods_per_year)
        summary_row = {"strategy_id": strategy_id, "scenario": "observed", **observed_metrics}

        for regime_key, regime_group in group.groupby(["regime", "regime_name"], sort=True):
            regime, regime_name = regime_key
            metrics = calculate_metrics(regime_group["return"], periods_per_year=periods_per_year)
            regime_rows.append({"strategy_id": strategy_id, "regime": regime, "regime_name": regime_name, **metrics})

        sim_metrics = []
        for sim_id in range(n_sims):
            simulated = _simulate_path(group, horizon=horizon, block_size=block_size, rng=rng)
            metrics = calculate_metrics(pd.Series(simulated), periods_per_year=periods_per_year)
            horizon_return = float(np.prod(1.0 + simulated) - 1.0)
            simulation_rows.append(
                {
                    "strategy_id": strategy_id,
                    "sim_id": sim_id,
                    "horizon_return": horizon_return,
                    **metrics,
                }
            )
            sim_metrics.append(metrics)

        summary_row.update(_simulation_percentiles(sim_metrics))
        summary_rows.append(summary_row)
        replay_rows.extend(_historical_replays(strategy_id, group, horizon=horizon, periods_per_year=periods_per_year))

    return StressTestResult(
        summary=pd.DataFrame(summary_rows).sort_values("strategy_id").reset_index(drop=True),
        simulations=pd.DataFrame(simulation_rows).sort_values(["strategy_id", "sim_id"]).reset_index(drop=True),
        regime_summary=pd.DataFrame(regime_rows).sort_values(["strategy_id", "regime"]).reset_index(drop=True),
        crisis_replay=pd.DataFrame(replay_rows).sort_values(["strategy_id", "start_date"]).reset_index(drop=True),
        metadata={
            "n_sims": n_sims,
            "horizon": horizon,
            "block_size": block_size,
            "seed": seed,
            "periods_per_year": periods_per_year,
            "method": "regime_conditioned_block_bootstrap",
        },
    )


def _align_returns_and_regimes(strategy_returns: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    required_returns = {"date", "strategy_id", "return"}
    required_regimes = {"date", "regime", "regime_name"}
    missing_returns = sorted(required_returns.difference(strategy_returns.columns))
    missing_regimes = sorted(required_regimes.difference(regimes.columns))
    if missing_returns:
        raise ValueError(f"strategy_returns missing required columns: {missing_returns}")
    if missing_regimes:
        raise ValueError(f"regimes missing required columns: {missing_regimes}")

    left = strategy_returns.copy()
    right = regimes.copy()
    left["date"] = pd.to_datetime(left["date"]).dt.tz_localize(None)
    right["date"] = pd.to_datetime(right["date"]).dt.tz_localize(None)
    merged = left.merge(right[["date", "regime", "regime_name"]], on="date", how="inner")
    merged["return"] = pd.to_numeric(merged["return"], errors="raise")
    return merged.sort_values(["strategy_id", "date"]).reset_index(drop=True)


def _simulate_path(group: pd.DataFrame, horizon: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    regimes = group[["regime", "regime_name"]].drop_duplicates().reset_index(drop=True)
    regime_counts = group["regime"].value_counts(normalize=True).sort_index()
    regime_values = regime_counts.index.to_numpy()
    regime_probs = regime_counts.to_numpy()
    simulated: list[float] = []

    while len(simulated) < horizon:
        selected_regime = rng.choice(regime_values, p=regime_probs)
        candidates = group[group["regime"] == selected_regime]["return"].to_numpy(dtype=float)
        if len(candidates) < block_size:
            candidates = group["return"].to_numpy(dtype=float)
        start_max = max(len(candidates) - block_size, 0)
        start = int(rng.integers(0, start_max + 1)) if start_max else 0
        simulated.extend(candidates[start : start + block_size].tolist())

    return np.asarray(simulated[:horizon], dtype=float)


def _simulation_percentiles(sim_metrics: list[dict[str, float]]) -> dict[str, float]:
    frame = pd.DataFrame(sim_metrics)
    output: dict[str, float] = {}
    for metric in ["cagr", "max_drawdown", "var_95", "cvar_95"]:
        output[f"sim_{metric}_p05"] = float(frame[metric].quantile(0.05))
        output[f"sim_{metric}_p50"] = float(frame[metric].quantile(0.50))
        output[f"sim_{metric}_p95"] = float(frame[metric].quantile(0.95))
    return output


def _historical_replays(
    strategy_id: str,
    group: pd.DataFrame,
    horizon: int,
    periods_per_year: int,
    top_n: int = 5,
) -> list[dict[str, object]]:
    if len(group) < horizon:
        horizon = len(group)
    if horizon <= 0:
        return []

    windows = []
    returns = group["return"].to_numpy(dtype=float)
    for start in range(0, len(group) - horizon + 1):
        window = returns[start : start + horizon]
        metrics = calculate_metrics(pd.Series(window), periods_per_year=periods_per_year)
        windows.append(
            {
                "strategy_id": strategy_id,
                "start_date": group.loc[start, "date"],
                "end_date": group.loc[start + horizon - 1, "date"],
                "horizon_return": float(np.prod(1.0 + window) - 1.0),
                **metrics,
            }
        )
    return sorted(windows, key=lambda row: row["max_drawdown"])[:top_n]
