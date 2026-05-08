from __future__ import annotations

import math

import numpy as np
import pandas as pd


def calculate_metrics(returns: pd.Series | np.ndarray, periods_per_year: int = 252) -> dict[str, float]:
    series = pd.Series(returns, dtype="float64").dropna()
    if series.empty:
        return {
            "cagr": math.nan,
            "volatility": math.nan,
            "sharpe": math.nan,
            "sortino": math.nan,
            "max_drawdown": math.nan,
            "var_95": math.nan,
            "cvar_95": math.nan,
            "skew": math.nan,
            "kurtosis": math.nan,
        }

    values = series.to_numpy(dtype=float)
    cumulative_return = float(np.prod(1.0 + values) - 1.0)
    if cumulative_return <= -1.0:
        cagr = -1.0
    else:
        cagr = float((1.0 + cumulative_return) ** (periods_per_year / len(values)) - 1.0)

    daily_std = float(np.std(values, ddof=0))
    if daily_std < 1e-12:
        daily_std = 0.0
    volatility = daily_std * math.sqrt(periods_per_year)
    daily_mean = float(np.mean(values))
    sharpe = daily_mean / daily_std * math.sqrt(periods_per_year) if daily_std > 0 else math.nan

    downside = values[values < 0.0]
    downside_std = float(np.std(downside, ddof=0)) if len(downside) else 0.0
    if downside_std < 1e-12:
        downside_std = 0.0
    sortino = daily_mean / downside_std * math.sqrt(periods_per_year) if downside_std > 0 else math.nan

    equity = np.cumprod(1.0 + values)
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1.0
    max_drawdown = float(np.min(drawdowns))

    var_95 = float(np.quantile(values, 0.05))
    tail = values[values <= var_95]
    cvar_95 = float(np.mean(tail)) if len(tail) else var_95

    skew, kurtosis = _shape_moments(values)

    return {
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "skew": skew,
        "kurtosis": kurtosis,
    }


def _shape_moments(values: np.ndarray) -> tuple[float, float]:
    std = float(np.std(values, ddof=0))
    if std == 0.0:
        return math.nan, math.nan
    centered = values - float(np.mean(values))
    skew = float(np.mean((centered / std) ** 3))
    kurtosis = float(np.mean((centered / std) ** 4) - 3.0)
    return skew, kurtosis
