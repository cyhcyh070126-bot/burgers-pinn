"""Original formal Burgers model, sampling and diagnostics.

Selected from the archived vanilla baseline; numerical functions preserve the
original implementation. Entropy-reference diagnostics never enter the loss.
"""
from __future__ import annotations
import csv
import math
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TrainingConfig:
    """Frozen numerical and optimization contract for baseline one."""

    seed: int = 1234
    x_min: float = -1.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0
    shock_speed: float = 0.5
    hidden_width: int = 64
    hidden_layers: int = 8
    activation: str = "Tanh"
    initial_points_per_side: int = 2048
    boundary_points_per_side: int = 2048
    pde_points: int = 16384
    initial_points_per_side_batch: int = 64
    boundary_points_per_side_batch: int = 64
    pde_batch: int = 512
    epochs: int = 1000
    learning_rate_start: float = 1.0e-3
    learning_rate_min: float = 1.0e-5
    evaluation_interval_epochs: int = 5
    evaluation_x_points: int = 1001
    evaluation_t_points: int = 101
    inference_batch_size: int = 8192

    @property
    def total_batch_size(self) -> int:
        return (
            2 * self.initial_points_per_side_batch
            + 2 * self.boundary_points_per_side_batch
            + self.pde_batch
        )

    @property
    def batches_per_epoch(self) -> int:
        """Return the shared fixed-pool sweep count, or reject partial sweeps."""
        counts = (
            self.initial_points_per_side // self.initial_points_per_side_batch,
            self.boundary_points_per_side // self.boundary_points_per_side_batch,
            self.pde_points // self.pde_batch,
        )
        divisors = (
            self.initial_points_per_side % self.initial_points_per_side_batch,
            self.boundary_points_per_side % self.boundary_points_per_side_batch,
            self.pde_points % self.pde_batch,
        )
        if any(divisors) or len(set(counts)) != 1:
            raise ValueError(
                "Fixed IC, BC, and PDE pools must divide into the same number "
                "of complete mini-batches."
            )
        return counts[0]

    @property
    def total_fixed_points(self) -> int:
        return (
            2 * self.initial_points_per_side
            + 2 * self.boundary_points_per_side
            + self.pde_points
        )

    @property
    def total_optimizer_steps(self) -> int:
        return self.epochs * self.batches_per_epoch


class VanillaBurgersPINN(nn.Module):
    """Shared coordinate network u_theta(x, t)."""

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        widths = [2] + [config.hidden_width] * config.hidden_layers + [1]
        layers: list[nn.Module] = []
        for input_width, output_width in pairwise(widths):
            layers.append(nn.Linear(input_width, output_width))
        self.layers = nn.ModuleList(layers)
        self.activation = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        values = torch.cat((x, t), dim=1)
        for layer in self.layers[:-1]:
            values = self.activation(layer(values))
        return self.layers[-1](values)


def set_reproducible_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def uniform_points(count: int, low: float, high: float, device: torch.device) -> Tensor:
    return low + (high - low) * torch.rand((count, 1), device=device)


def build_fixed_training_points(
    config: TrainingConfig, device: torch.device
) -> dict[str, Tensor]:
    """Create the fixed IC, BC, and interior pools once on the selected device."""
    x_ic_left = uniform_points(
        config.initial_points_per_side, config.x_min, 0.0, device
    )
    x_ic_right = uniform_points(
        config.initial_points_per_side, 0.0, config.x_max, device
    )
    t_bc_left = uniform_points(
        config.boundary_points_per_side, config.t_min, config.t_max, device
    )
    t_bc_right = uniform_points(
        config.boundary_points_per_side, config.t_min, config.t_max, device
    )
    return {
        "x_ic_left": x_ic_left,
        "x_ic_right": x_ic_right,
        "t_bc_left": t_bc_left,
        "t_bc_right": t_bc_right,
        "x_pde": uniform_points(config.pde_points, config.x_min, config.x_max, device),
        "t_pde": uniform_points(config.pde_points, config.t_min, config.t_max, device),
    }


def fixed_epoch_permutations(
    points: dict[str, Tensor], device: torch.device
) -> dict[str, Tensor]:
    """Shuffle fixed-pool indices without resampling any physical coordinates."""
    return {
        "ic_left": torch.randperm(points["x_ic_left"].shape[0], device=device),
        "ic_right": torch.randperm(points["x_ic_right"].shape[0], device=device),
        "bc_left": torch.randperm(points["t_bc_left"].shape[0], device=device),
        "bc_right": torch.randperm(points["t_bc_right"].shape[0], device=device),
        "pde": torch.randperm(points["x_pde"].shape[0], device=device),
    }


def fixed_training_batch(
    points: dict[str, Tensor],
    permutations: dict[str, Tensor],
    batch_index: int,
    config: TrainingConfig,
) -> dict[str, Tensor]:
    """Return one balanced batch; all fixed points occur exactly once per epoch."""
    ic_start = batch_index * config.initial_points_per_side_batch
    bc_start = batch_index * config.boundary_points_per_side_batch
    pde_start = batch_index * config.pde_batch
    ic_left = permutations["ic_left"][
        ic_start : ic_start + config.initial_points_per_side_batch
    ]
    ic_right = permutations["ic_right"][
        ic_start : ic_start + config.initial_points_per_side_batch
    ]
    bc_left = permutations["bc_left"][
        bc_start : bc_start + config.boundary_points_per_side_batch
    ]
    bc_right = permutations["bc_right"][
        bc_start : bc_start + config.boundary_points_per_side_batch
    ]
    pde = permutations["pde"][pde_start : pde_start + config.pde_batch]

    x_ic_left = points["x_ic_left"][ic_left]
    x_ic_right = points["x_ic_right"][ic_right]
    t_bc_left = points["t_bc_left"][bc_left]
    t_bc_right = points["t_bc_right"][bc_right]
    return {
        "x_ic": torch.cat((x_ic_left, x_ic_right), dim=0),
        "t_ic": torch.zeros(
            (2 * config.initial_points_per_side_batch, 1),
            device=x_ic_left.device,
            dtype=x_ic_left.dtype,
        ),
        "u_ic": torch.cat(
            (torch.ones_like(x_ic_left), torch.zeros_like(x_ic_right)), dim=0
        ),
        "x_bc": torch.cat(
            (-torch.ones_like(t_bc_left), torch.ones_like(t_bc_right)), dim=0
        ),
        "t_bc": torch.cat((t_bc_left, t_bc_right), dim=0),
        "u_bc": torch.cat(
            (torch.ones_like(t_bc_left), torch.zeros_like(t_bc_right)), dim=0
        ),
        "x_pde": points["x_pde"][pde],
        "t_pde": points["t_pde"][pde],
    }


def exact_entropy_solution(x: Tensor, t: Tensor, config: TrainingConfig) -> Tensor:
    """Analytical entropy solution, reserved exclusively for evaluation."""
    return torch.where(
        x < config.shock_speed * t, torch.ones_like(x), torch.zeros_like(x)
    )


@torch.no_grad()
def predict_in_batches(
    model: nn.Module, x: Tensor, t: Tensor, batch_size: int
) -> Tensor:
    chunks: list[Tensor] = []
    for start in range(0, x.shape[0], batch_size):
        stop = min(start + batch_size, x.shape[0])
        chunks.append(model(x[start:stop], t[start:stop]))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def relative_l2_on_fixed_grid(
    model: nn.Module, config: TrainingConfig, device: torch.device
) -> float:
    """A fixed-grid diagnostic; it never enters the optimizer or checkpoint rule."""
    x_values = torch.linspace(config.x_min, config.x_max, 401, device=device)
    t_values = torch.linspace(config.t_min, config.t_max, 81, device=device)
    t_grid, x_grid = torch.meshgrid(t_values, x_values, indexing="ij")
    x = x_grid.reshape(-1, 1)
    t = t_grid.reshape(-1, 1)
    prediction = predict_in_batches(model, x, t, config.inference_batch_size)
    truth = exact_entropy_solution(x, t, config)
    numerator = torch.linalg.vector_norm(prediction - truth)
    denominator = torch.linalg.vector_norm(truth)
    return float((numerator / denominator).item())


def cosine_learning_rate(optimizer_step: int, config: TrainingConfig) -> float:
    """Return the rate for one update, including eta_min at the final update."""
    if not 1 <= optimizer_step <= config.total_optimizer_steps:
        raise ValueError(
            f"Optimizer step {optimizer_step} is outside [1, {config.total_optimizer_steps}]"
        )
    progress = (optimizer_step - 1) / max(config.total_optimizer_steps - 1, 1)
    return config.learning_rate_min + 0.5 * (
        config.learning_rate_start - config.learning_rate_min
    ) * (1.0 + math.cos(math.pi * progress))


def estimate_level_crossing(x: np.ndarray, values: np.ndarray, level: float) -> float:
    """Find the leftmost downward level crossing, with a transparent fallback."""
    shifted = values - level
    crossings = np.flatnonzero((shifted[:-1] >= 0.0) & (shifted[1:] <= 0.0))
    if crossings.size == 0:
        return float(x[int(np.argmin(np.abs(shifted)))])
    index = int(crossings[0])
    left_value, right_value = values[index], values[index + 1]
    if math.isclose(left_value, right_value):
        return float(x[index])
    fraction = (level - left_value) / (right_value - left_value)
    return float(x[index] + fraction * (x[index + 1] - x[index]))


@torch.no_grad()
def evaluate_final_model(
    model: nn.Module, config: TrainingConfig, device: torch.device
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate only after training; exact data are not fed back into training."""
    x_values = torch.linspace(
        config.x_min, config.x_max, config.evaluation_x_points, device=device
    )
    t_values = torch.linspace(
        config.t_min, config.t_max, config.evaluation_t_points, device=device
    )
    t_grid, x_grid = torch.meshgrid(t_values, x_values, indexing="ij")
    x_flat = x_grid.reshape(-1, 1)
    t_flat = t_grid.reshape(-1, 1)
    prediction = predict_in_batches(model, x_flat, t_flat, config.inference_batch_size)
    truth = exact_entropy_solution(x_flat, t_flat, config)
    prediction_grid = prediction.reshape(
        config.evaluation_t_points, config.evaluation_x_points
    )
    truth_grid = truth.reshape(config.evaluation_t_points, config.evaluation_x_points)

    prediction_np = prediction_grid.detach().cpu().numpy()
    truth_np = truth_grid.detach().cpu().numpy()
    x_np = x_values.detach().cpu().numpy()
    t_np = t_values.detach().cpu().numpy()
    absolute_error = np.abs(prediction_np - truth_np)
    relative_l2 = float(
        np.linalg.norm(prediction_np - truth_np) / np.linalg.norm(truth_np)
    )

    shock_location_errors: list[float] = []
    shock_thicknesses: list[float] = []
    for t_value, profile in zip(t_np[1:], prediction_np[1:], strict=True):
        predicted_shock = estimate_level_crossing(x_np, profile, 0.5)
        shock_location_errors.append(
            abs(predicted_shock - config.shock_speed * float(t_value))
        )
        x_at_09 = estimate_level_crossing(x_np, profile, 0.9)
        x_at_01 = estimate_level_crossing(x_np, profile, 0.1)
        shock_thicknesses.append(abs(x_at_01 - x_at_09))

    metrics: dict[str, Any] = {
        "evaluation_grid": {
            "x_points": config.evaluation_x_points,
            "t_points": config.evaluation_t_points,
            "domain": [[config.x_min, config.x_max], [config.t_min, config.t_max]],
        },
        "analytical_solution_usage": "evaluation_only; excluded from training loss",
        "relative_l2_fixed_grid": relative_l2,
        "mean_absolute_error_fixed_grid": float(np.mean(absolute_error)),
        "maximum_absolute_error_fixed_grid": float(np.max(absolute_error)),
        "shock_location_definition": "leftmost downward crossing of u_hat=0.5; nearest-value fallback if absent",
        "mean_shock_location_absolute_error": float(np.mean(shock_location_errors)),
        "maximum_shock_location_absolute_error": float(np.max(shock_location_errors)),
        "shock_thickness_definition": "absolute distance between u_hat=0.9 and u_hat=0.1 crossings",
        "mean_predicted_shock_thickness": float(np.mean(shock_thicknesses)),
        "maximum_predicted_shock_thickness": float(np.max(shock_thicknesses)),
    }
    arrays = {
        "x": x_np,
        "t": t_np,
        "prediction": prediction_np,
        "truth": truth_np,
        "absolute_error": absolute_error,
    }
    return metrics, arrays


def write_history_csv(history: list[dict[str, float]], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def read_history_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def scalar_residual(
    model: nn.Module, x: Tensor, t: Tensor, viscosity: float
) -> Tensor:
    """Strong residual away from shocks; viscosity=0 gives Vanilla PINN."""
    x_coordinate = x.detach().requires_grad_(True)
    t_coordinate = t.detach().requires_grad_(True)
    prediction = model(x_coordinate, t_coordinate)
    u_t = torch.autograd.grad(
        prediction,
        t_coordinate,
        grad_outputs=torch.ones_like(prediction),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_x = torch.autograd.grad(
        prediction,
        x_coordinate,
        grad_outputs=torch.ones_like(prediction),
        create_graph=True,
        retain_graph=True,
    )[0]
    residual = u_t + prediction * u_x
    if viscosity == 0.0:
        return residual
    u_xx = torch.autograd.grad(
        u_x,
        x_coordinate,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0]
    return residual - viscosity * u_xx


def train_scalar(
    config: TrainingConfig, device: torch.device, viscosity: float
) -> tuple[VanillaBurgersPINN, list[dict[str, float]], dict[str, Tensor]]:
    """Train one controlled scalar model with identical pools and budget."""
    model = VanillaBurgersPINN(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate_start)
    pools = build_fixed_training_points(config, device)
    history: list[dict[str, float]] = []
    step = 0
    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        permutations = fixed_epoch_permutations(pools, device)
        sums = {key: 0.0 for key in ("total", "ic", "bc", "pde")}
        for batch_index in range(config.batches_per_epoch):
            step += 1
            rate = cosine_learning_rate(step, config)
            optimizer.param_groups[0]["lr"] = rate
            item = fixed_training_batch(pools, permutations, batch_index, config)
            optimizer.zero_grad(set_to_none=True)
            ic = torch.mean((model(item["x_ic"], item["t_ic"]) - item["u_ic"]).square())
            bc = torch.mean((model(item["x_bc"], item["t_bc"]) - item["u_bc"]).square())
            pde = torch.mean(
                scalar_residual(model, item["x_pde"], item["t_pde"], viscosity).square()
            )
            total = ic + bc + pde
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {step}")
            total.backward()
            optimizer.step()
            for key, value in (("total", total), ("ic", ic), ("bc", bc), ("pde", pde)):
                sums[key] += float(value.detach())
        relative_l2 = float("nan")
        if epoch == 1 or epoch % config.evaluation_interval_epochs == 0 or epoch == config.epochs:
            relative_l2 = relative_l2_on_fixed_grid(model, config, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        history.append(
            {
                "epoch": float(epoch),
                "optimizer_step": float(step),
                "epoch_elapsed_seconds": time.perf_counter() - started,
                "total_loss": sums["total"] / config.batches_per_epoch,
                "initial_condition_loss": sums["ic"] / config.batches_per_epoch,
                "boundary_condition_loss": sums["bc"] / config.batches_per_epoch,
                "pde_residual_loss": sums["pde"] / config.batches_per_epoch,
                "relative_l2_fixed_grid": relative_l2,
                "learning_rate": rate,
            }
        )
        if epoch == 1 or epoch % config.evaluation_interval_epochs == 0 or epoch == config.epochs:
            print(
                f"Epoch {epoch}/{config.epochs} | updates {step}/{config.total_optimizer_steps} | "
                f"loss {sums['total'] / config.batches_per_epoch:.6g} | "
                f"diagnostic relative L2 {relative_l2:.6g} | lr {rate:.3g}",
                flush=True,
            )
    return model, history, pools
