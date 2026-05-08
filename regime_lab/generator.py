from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


def fit_deep_generator(
    windows,
    regime_labels,
    epochs: int = 25,
    seed: int = 42,
    backend: str = "auto",
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

    if backend not in {"auto", "bootstrap", "torch"}:
        raise ValueError("backend must be one of: auto, bootstrap, torch")
    if backend == "torch":
        return _fit_torch_generator(window_array, label_array, epochs=epochs, seed=seed)
    elif backend == "auto" and _torch_available():
        return _fit_torch_generator(window_array, label_array, epochs=epochs, seed=seed)

    return WindowGenerator(windows=window_array, labels=label_array, backend="bootstrap", seed=seed)


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
