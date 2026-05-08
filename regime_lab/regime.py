from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeModelResult:
    regimes: pd.DataFrame
    features: pd.DataFrame
    centroids: pd.DataFrame
    n_regimes: int
    metadata: dict[str, object]


def fit_regime_model(
    market_data: pd.DataFrame,
    n_regimes: int = 4,
    lookback: int = 20,
    min_train: int = 30,
    seed: int = 42,
) -> RegimeModelResult:
    if n_regimes < 2:
        raise ValueError("n_regimes must be at least 2")
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if min_train < n_regimes:
        raise ValueError("min_train must be greater than or equal to n_regimes")

    features = _build_market_features(market_data, lookback=lookback)
    if len(features) < min_train:
        raise ValueError(
            f"Not enough market history after feature construction: need at least {min_train}, got {len(features)}"
        )

    feature_columns = [column for column in features.columns if column != "date"]
    regimes: list[dict[str, object]] = []
    centroids_by_date: list[pd.DataFrame] = []

    for index in range(min_train - 1, len(features)):
        train = features.iloc[: index + 1][feature_columns].to_numpy(dtype=float)
        scaled, mean, std = _standardize(train)
        centroids_scaled = _kmeans(scaled, n_clusters=n_regimes, seed=seed)
        current_scaled = (features.iloc[index][feature_columns].to_numpy(dtype=float) - mean) / std
        label = int(_nearest_centroid(current_scaled, centroids_scaled))
        centroids_raw = centroids_scaled * std + mean
        name = _name_regime(centroids_raw[label], feature_columns)

        row = features.iloc[index]
        regimes.append({"date": row["date"], "regime": label, "regime_name": name})
        centroids_by_date.append(
            pd.DataFrame(centroids_raw, columns=feature_columns).assign(date=row["date"], regime=range(n_regimes))
        )

    regime_frame = pd.DataFrame(regimes).reset_index(drop=True)
    centroids = pd.concat(centroids_by_date, ignore_index=True)
    return RegimeModelResult(
        regimes=regime_frame,
        features=features.reset_index(drop=True),
        centroids=centroids,
        n_regimes=n_regimes,
        metadata={"lookback": lookback, "min_train": min_train, "seed": seed, "model": "rolling_kmeans"},
    )


def _build_market_features(market_data: pd.DataFrame, lookback: int) -> pd.DataFrame:
    required = {"date", "symbol", "close"}
    missing = sorted(required.difference(market_data.columns))
    if missing:
        raise ValueError(f"market_data missing required columns for regime model: {missing}")

    prices = (
        market_data[["date", "symbol", "close"]]
        .copy()
        .assign(date=lambda frame: pd.to_datetime(frame["date"]).dt.tz_localize(None))
        .pivot(index="date", columns="symbol", values="close")
        .sort_index()
    )
    if prices.shape[1] < 2:
        raise ValueError("fit_regime_model requires at least two symbols for cross-asset regime features")

    returns = prices.pct_change(fill_method=None).dropna(how="all")
    market_return = returns.mean(axis=1)
    market_index = (1.0 + market_return.fillna(0.0)).cumprod()
    rolling_max = market_index.rolling(lookback, min_periods=lookback).max()
    rolling_corr = _rolling_average_correlation(returns, lookback=lookback)

    features = pd.DataFrame(
        {
            "date": returns.index,
            "market_return_mean": market_return.rolling(lookback, min_periods=lookback).mean(),
            "market_volatility": market_return.rolling(lookback, min_periods=lookback).std(ddof=0),
            "market_drawdown": market_index / rolling_max - 1.0,
            "market_trend": prices.mean(axis=1).pct_change(lookback).reindex(returns.index),
            "average_correlation": rolling_corr,
        }
    )
    return features.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def _rolling_average_correlation(returns: pd.DataFrame, lookback: int) -> pd.Series:
    values: list[float] = []
    dates = returns.index
    for end in range(len(returns)):
        if end + 1 < lookback:
            values.append(np.nan)
            continue
        window = returns.iloc[end + 1 - lookback : end + 1].dropna(axis=1, how="all")
        if window.shape[1] < 2:
            values.append(np.nan)
            continue
        corr = window.corr().to_numpy(dtype=float)
        mask = ~np.eye(corr.shape[0], dtype=bool)
        values.append(float(np.nanmean(corr[mask])))
    return pd.Series(values, index=dates)


def _standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return (values - mean) / std, mean, std


def _kmeans(values: np.ndarray, n_clusters: int, seed: int, max_iter: int = 50) -> np.ndarray:
    if len(values) < n_clusters:
        raise ValueError("Not enough rows to fit requested number of regimes")

    centroids = _initial_centroids(values, n_clusters=n_clusters, seed=seed)
    for _ in range(max_iter):
        distances = np.linalg.norm(values[:, None, :] - centroids[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        updated = centroids.copy()
        for cluster in range(n_clusters):
            members = values[labels == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
        if np.allclose(updated, centroids):
            break
        centroids = updated
    return centroids


def _initial_centroids(values: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    scores = values[:, 0] + values[:, 1] - values[:, 2]
    order = np.argsort(scores, kind="mergesort")
    positions = np.linspace(0, len(order) - 1, n_clusters).round().astype(int)
    centroids = values[order[positions]].copy()
    if len(np.unique(centroids, axis=0)) < n_clusters:
        rng = np.random.default_rng(seed)
        jitter = rng.normal(0.0, 1e-6, size=centroids.shape)
        centroids = centroids + jitter
    return centroids


def _nearest_centroid(value: np.ndarray, centroids: np.ndarray) -> int:
    distances = np.linalg.norm(centroids - value, axis=1)
    return int(distances.argmin())


def _name_regime(centroid: np.ndarray, feature_columns: list[str]) -> str:
    lookup = dict(zip(feature_columns, centroid, strict=True))
    risk_score = (
        lookup.get("market_volatility", 0.0)
        - lookup.get("market_return_mean", 0.0)
        - lookup.get("market_drawdown", 0.0)
        - lookup.get("market_trend", 0.0)
    )
    if risk_score > 0.03:
        return "stress"
    if lookup.get("market_return_mean", 0.0) > 0 and lookup.get("market_trend", 0.0) > 0:
        return "uptrend"
    if lookup.get("market_drawdown", 0.0) < -0.03:
        return "drawdown"
    return "steady"
