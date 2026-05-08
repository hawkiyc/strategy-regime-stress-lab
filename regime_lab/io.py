from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


MARKET_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]
STRATEGY_REQUIRED_COLUMNS = ["date", "strategy_id", "return"]


def load_market_data(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    _require_columns(data, MARKET_COLUMNS, "market_prices.csv")
    _reject_nulls(data, MARKET_COLUMNS, "market_prices.csv")

    normalized = data[MARKET_COLUMNS].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], utc=False).dt.tz_localize(None)
    normalized["symbol"] = normalized["symbol"].astype(str)
    for column in ["open", "high", "low", "close", "volume"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise")

    duplicate_mask = normalized.duplicated(subset=["date", "symbol"], keep=False)
    if duplicate_mask.any():
        duplicates = normalized.loc[duplicate_mask, ["date", "symbol"]].head(5).to_dict("records")
        raise ValueError(f"market_prices.csv contains duplicate date/symbol rows: {duplicates}")

    return normalized.sort_values(["date", "symbol"]).reset_index(drop=True)


def load_strategy_returns(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    _require_columns(data, STRATEGY_REQUIRED_COLUMNS, "strategy_returns.csv")
    _reject_nulls(data, STRATEGY_REQUIRED_COLUMNS, "strategy_returns.csv")

    optional_columns = [column for column in ["benchmark", "exposure"] if column in data.columns]
    columns = STRATEGY_REQUIRED_COLUMNS + optional_columns
    normalized = data[columns].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], utc=False).dt.tz_localize(None)
    normalized["strategy_id"] = normalized["strategy_id"].astype(str)
    normalized["return"] = pd.to_numeric(normalized["return"], errors="raise")
    for column in optional_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise")

    duplicate_mask = normalized.duplicated(subset=["date", "strategy_id"], keep=False)
    if duplicate_mask.any():
        duplicates = normalized.loc[duplicate_mask, ["date", "strategy_id"]].head(5).to_dict("records")
        raise ValueError(f"strategy_returns.csv contains duplicate date/strategy_id rows: {duplicates}")

    return normalized.sort_values(["date", "strategy_id"]).reset_index(drop=True)


def _require_columns(data: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _reject_nulls(data: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    null_columns = [column for column in columns if data[column].isna().any()]
    if null_columns:
        raise ValueError(f"{label} contains null values in columns: {null_columns}")
