from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReturnWindowDataset:
    windows: np.ndarray
    labels: np.ndarray
    strategy_ids: np.ndarray
    end_dates: np.ndarray
    source: np.ndarray


@dataclass(frozen=True)
class WindowGenerator:
    windows: np.ndarray
    labels: np.ndarray
    backend: str
    seed: int

    def sample(self, n: int, regime: int | None = None, seed: int | None = None) -> np.ndarray:
        if n < 1:
            raise ValueError("n must be positive")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        if regime is None:
            candidates = self.windows
        else:
            candidates = self.windows[self.labels == regime]
            if len(candidates) == 0:
                raise ValueError(f"No windows available for regime {regime}")
        indices = rng.integers(0, len(candidates), size=n)
        noise_scale = np.maximum(candidates.std(axis=0, ddof=0), 1e-12) * 0.02
        noise = rng.normal(0.0, noise_scale, size=(n, candidates.shape[1]))
        return candidates[indices] + noise

    def augment(
        self,
        windows,
        regime_labels,
        n_per_regime: int = 100,
        seed: int | None = None,
    ) -> ReturnWindowDataset:
        return _augment_with_generator(self, windows, regime_labels, n_per_regime=n_per_regime, seed=seed)


def fit_deep_generator(
    windows,
    regime_labels,
    epochs: int = 25,
    seed: int = 42,
    backend: str = "bootstrap",
) -> WindowGenerator:
    window_array = np.asarray(windows, dtype=float)
    label_array = np.asarray(regime_labels)
    if window_array.ndim != 2:
        raise ValueError("windows must be a 2D array shaped as (n_windows, window_length)")
    if len(window_array) != len(label_array):
        raise ValueError("windows and regime_labels must have the same length")
    if len(window_array) == 0:
        raise ValueError("windows must not be empty")
    if epochs < 1:
        raise ValueError("epochs must be positive")

    if backend not in {"bootstrap", "torch"}:
        raise ValueError("backend must be one of: bootstrap, torch")
    if backend == "torch":
        return _fit_torch_generator(window_array, label_array, epochs=epochs, seed=seed)

    return WindowGenerator(windows=window_array, labels=label_array, backend="bootstrap", seed=seed)


def make_return_windows(
    strategy_returns: pd.DataFrame,
    regimes: pd.DataFrame,
    window_length: int = 20,
    stride: int = 1,
    return_column: str = "return",
) -> ReturnWindowDataset:
    if window_length < 2:
        raise ValueError("window_length must be at least 2")
    if stride < 1:
        raise ValueError("stride must be positive")
    required_returns = {"date", "strategy_id", return_column}
    required_regimes = {"date", "regime"}
    missing_returns = sorted(required_returns.difference(strategy_returns.columns))
    missing_regimes = sorted(required_regimes.difference(regimes.columns))
    if missing_returns:
        raise ValueError(f"strategy_returns missing required columns: {missing_returns}")
    if missing_regimes:
        raise ValueError(f"regimes missing required columns: {missing_regimes}")

    returns = strategy_returns.copy()
    regime_frame = regimes.copy()
    returns["date"] = pd.to_datetime(returns["date"]).dt.tz_localize(None)
    regime_frame["date"] = pd.to_datetime(regime_frame["date"]).dt.tz_localize(None)
    merged = returns.merge(regime_frame[["date", "regime"]], on="date", how="inner")
    merged[return_column] = pd.to_numeric(merged[return_column], errors="raise")

    windows: list[np.ndarray] = []
    labels: list[object] = []
    strategy_ids: list[str] = []
    end_dates: list[pd.Timestamp] = []
    for strategy_id, group in merged.groupby("strategy_id", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        values = group[return_column].to_numpy(dtype=float)
        regime_values = group["regime"].to_numpy()
        for start in range(0, len(group) - window_length + 1, stride):
            end = start + window_length
            window_regimes = regime_values[start:end]
            label = _majority_label(window_regimes)
            windows.append(values[start:end])
            labels.append(label)
            strategy_ids.append(str(strategy_id))
            end_dates.append(group.loc[end - 1, "date"])

    if not windows:
        raise ValueError("Not enough overlapping strategy/regime history to create return windows")
    return ReturnWindowDataset(
        windows=np.asarray(windows, dtype=float),
        labels=np.asarray(labels),
        strategy_ids=np.asarray(strategy_ids),
        end_dates=np.asarray(end_dates, dtype="datetime64[ns]"),
        source=np.asarray(["real"] * len(windows)),
    )


def save_generator(generator, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(generator, handle)
    return output


def load_generator(path: str | Path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _require_torch() -> None:
    if not _torch_available():
        raise ImportError("PyTorch is required for backend='torch'. Install regime-lab[deep].")


@dataclass(frozen=True)
class TorchWindowGenerator:
    state_dict: dict
    label_values: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    latent_dim: int
    hidden_dim: int
    window_length: int
    seed: int
    backend: str = "torch"

    def sample(self, n: int, regime: int | None = None, seed: int | None = None) -> np.ndarray:
        if n < 1:
            raise ValueError("n must be positive")
        _require_torch()
        import torch
        import torch.nn.functional as functional

        torch.manual_seed(self.seed if seed is None else seed)
        generator = _build_torch_generator(
            torch.nn,
            latent_dim=self.latent_dim,
            label_count=len(self.label_values),
            window_length=self.window_length,
            hidden_dim=self.hidden_dim,
        )
        generator.load_state_dict(self.state_dict)
        generator.eval()

        if regime is None:
            label_indices = torch.randint(0, len(self.label_values), (n,), dtype=torch.long)
        else:
            matches = np.where(self.label_values == regime)[0]
            if len(matches) == 0:
                raise ValueError(f"No windows available for regime {regime}")
            label_indices = torch.full((n,), int(matches[0]), dtype=torch.long)

        one_hot = functional.one_hot(label_indices, num_classes=len(self.label_values)).float()
        noise = torch.randn(n, self.latent_dim)
        with torch.no_grad():
            normalized = generator(torch.cat([noise, one_hot], dim=1)).numpy()
        return normalized * self.std + self.mean

    def augment(
        self,
        windows,
        regime_labels,
        n_per_regime: int = 100,
        seed: int | None = None,
    ) -> ReturnWindowDataset:
        return _augment_with_generator(self, windows, regime_labels, n_per_regime=n_per_regime, seed=seed)


def _fit_torch_generator(windows: np.ndarray, labels: np.ndarray, epochs: int, seed: int) -> TorchWindowGenerator:
    _require_torch()
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(seed)
    label_values, label_indices = np.unique(labels, return_inverse=True)
    mean = windows.mean(axis=0)
    std = windows.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    normalized = (windows - mean) / std

    latent_dim = min(16, max(4, windows.shape[1]))
    hidden_dim = max(32, windows.shape[1] * 2)
    label_count = len(label_values)
    generator = _build_torch_generator(
        torch.nn,
        latent_dim=latent_dim,
        label_count=label_count,
        window_length=windows.shape[1],
        hidden_dim=hidden_dim,
    )
    critic = _build_torch_critic(
        torch.nn,
        label_count=label_count,
        window_length=windows.shape[1],
        hidden_dim=hidden_dim,
    )
    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-3, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(critic.parameters(), lr=1e-3, betas=(0.5, 0.9))
    x = torch.tensor(normalized, dtype=torch.float32)
    y = torch.tensor(label_indices, dtype=torch.long)
    batch_size = min(32, len(x))

    for _ in range(epochs):
        order = torch.randperm(len(x))
        for start in range(0, len(x), batch_size):
            batch_index = order[start : start + batch_size]
            real = x[batch_index]
            batch_labels = y[batch_index]
            one_hot = functional.one_hot(batch_labels, num_classes=label_count).float()

            noise = torch.randn(len(real), latent_dim)
            fake = generator(torch.cat([noise, one_hot], dim=1)).detach()
            real_score = critic(torch.cat([real, one_hot], dim=1)).mean()
            fake_score = critic(torch.cat([fake, one_hot], dim=1)).mean()
            penalty = _gradient_penalty(torch, critic, real, fake, one_hot)
            loss_d = fake_score - real_score + 10.0 * penalty
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            noise = torch.randn(len(real), latent_dim)
            generated = generator(torch.cat([noise, one_hot], dim=1))
            loss_g = -critic(torch.cat([generated, one_hot], dim=1)).mean()
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

    state = {key: value.detach().cpu() for key, value in generator.state_dict().items()}
    return TorchWindowGenerator(
        state_dict=state,
        label_values=label_values,
        mean=mean,
        std=std,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        window_length=windows.shape[1],
        seed=seed,
    )


def _build_torch_generator(nn, latent_dim: int, label_count: int, window_length: int, hidden_dim: int):
    return nn.Sequential(
        nn.Linear(latent_dim + label_count, hidden_dim),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_dim, window_length),
    )


def _build_torch_critic(nn, label_count: int, window_length: int, hidden_dim: int):
    return nn.Sequential(
        nn.Linear(window_length + label_count, hidden_dim),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_dim, 1),
    )


def _gradient_penalty(torch, critic, real, fake, one_hot):
    alpha = torch.rand(len(real), 1)
    interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    score = critic(torch.cat([interpolated, one_hot], dim=1))
    gradients = torch.autograd.grad(
        outputs=score.sum(),
        inputs=interpolated,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


def _augment_with_generator(generator, windows, regime_labels, n_per_regime: int, seed: int | None) -> ReturnWindowDataset:
    if n_per_regime < 1:
        raise ValueError("n_per_regime must be positive")
    real_windows = np.asarray(windows, dtype=float)
    real_labels = np.asarray(regime_labels)
    if real_windows.ndim != 2:
        raise ValueError("windows must be a 2D array shaped as (n_windows, window_length)")
    if len(real_windows) != len(real_labels):
        raise ValueError("windows and regime_labels must have the same length")

    synthetic_windows: list[np.ndarray] = []
    synthetic_labels: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for regime in np.unique(real_labels):
        regime_seed = int(rng.integers(0, 2**32 - 1))
        sample = generator.sample(n=n_per_regime, regime=regime, seed=regime_seed)
        synthetic_windows.append(sample)
        synthetic_labels.append(np.asarray([regime] * len(sample)))

    generated = np.vstack(synthetic_windows)
    generated_labels = np.concatenate(synthetic_labels)
    all_windows = np.vstack([real_windows, generated])
    all_labels = np.concatenate([real_labels, generated_labels])
    return ReturnWindowDataset(
        windows=all_windows,
        labels=all_labels,
        strategy_ids=np.asarray(["unknown"] * len(all_windows)),
        end_dates=np.asarray([np.datetime64("NaT")] * len(all_windows), dtype="datetime64[ns]"),
        source=np.asarray(["real"] * len(real_windows) + ["synthetic"] * len(generated)),
    )


def _majority_label(values: np.ndarray):
    labels, counts = np.unique(values, return_counts=True)
    return labels[np.argmax(counts)]
