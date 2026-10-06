"""GPU-only mini-batch baseline PINN for the inviscid Burgers shock.

This is deliberately a *vanilla* strong-form PINN baseline.  It learns one
fixed Riemann problem, not a neural operator, and it does not use the
analytical solution in its training objective.  The analytical entropy
solution is reserved for held-out evaluation and presentation.

Training contract (fixed for the first research baseline):
    * PDE: u_t + u u_x = 0 on [-1, 1] x [0, 1].
    * Network: [2, 64 x 8, 1] with Tanh activations.
    * Fixed uniform pools are shuffled into 32 mini-batches per epoch:
      128 initial, 128 boundary, and 512 PDE points per update.
    * 1,000 epochs x 32 mini-batches = 32,000 Adam updates.
    * Per-update cosine learning rate 1e-3 -> 1e-5.

This program intentionally refuses CPU execution.  It must be run in a
Slurm GPU allocation, never on a login node.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from argparse import ArgumentParser
from dataclasses import asdict, dataclass, fields
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

# ruff: noqa: I001
# Matplotlib must finish importing before PyMuPDF/fitz; see AGENTS.md import gate.
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation, PillowWriter
import fitz
import numpy as np
import torch
from torch import Tensor, nn
from PIL import Image


PINN_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PINN_DIR / "results"
CANDIDATE_DIR = RESULTS_DIR / "vanilla_burgers_uniform_long_8x64_candidate"
ANIMATION_FRAME_COUNT = 61
ANIMATION_FPS = 10
ANIMATION_DPI = 120


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


CONFIG = TrainingConfig()


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


def require_cuda() -> torch.device:
    """Refuse silent CPU fallback and record the scheduler-visible device."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this PINN baseline. Run inside a Slurm GPU "
            "allocation; CPU fallback is intentionally disabled."
        )
    torch.cuda.init()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "This baseline requires exactly one Slurm-visible GPU; "
            f"found {torch.cuda.device_count()}."
        )
    return torch.device("cuda:0")


def set_reproducible_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def uniform_points(count: int, low: float, high: float, device: torch.device) -> Tensor:
    return low + (high - low) * torch.rand((count, 1), device=device)


def build_fixed_training_points(
    config: TrainingConfig, device: torch.device
) -> dict[str, Tensor]:
    """Create the fixed IC, BC, and interior pools once on the GPU."""
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


def burgers_residual(model: nn.Module, x: Tensor, t: Tensor) -> Tensor:
    """Return r_theta = u_t + u u_x using automatic differentiation."""
    x_coordinate = x.detach().requires_grad_(True)
    t_coordinate = t.detach().requires_grad_(True)
    u_prediction = model(x_coordinate, t_coordinate)
    u_t = torch.autograd.grad(
        u_prediction,
        t_coordinate,
        grad_outputs=torch.ones_like(u_prediction),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_x = torch.autograd.grad(
        u_prediction,
        x_coordinate,
        grad_outputs=torch.ones_like(u_prediction),
        create_graph=True,
        retain_graph=True,
    )[0]
    return u_t + u_prediction * u_x


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


def train_baseline(
    config: TrainingConfig, device: torch.device
) -> tuple[VanillaBurgersPINN, list[dict[str, float]], dict[str, Tensor]]:
    model = VanillaBurgersPINN(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate_start)
    history: list[dict[str, float]] = []
    fixed_points = build_fixed_training_points(config, device)

    torch.cuda.synchronize(device)
    optimizer_step = 0
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        permutations = fixed_epoch_permutations(fixed_points, device)
        loss_sums = {"total": 0.0, "ic": 0.0, "bc": 0.0, "pde": 0.0}
        for batch_index in range(config.batches_per_epoch):
            optimizer_step += 1
            model.train()
            learning_rate = cosine_learning_rate(optimizer_step, config)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            batch = fixed_training_batch(
                fixed_points, permutations, batch_index, config
            )

            ic_loss = torch.mean(
                (model(batch["x_ic"], batch["t_ic"]) - batch["u_ic"]) ** 2
            )
            bc_loss = torch.mean(
                (model(batch["x_bc"], batch["t_bc"]) - batch["u_bc"]) ** 2
            )
            pde_loss = torch.mean(
                burgers_residual(model, batch["x_pde"], batch["t_pde"]) ** 2
            )
            total_loss = ic_loss + bc_loss + pde_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch}, optimizer step {optimizer_step}"
                )

            total_loss.backward()
            optimizer.step()
            loss_sums["total"] += float(total_loss.detach().item())
            loss_sums["ic"] += float(ic_loss.detach().item())
            loss_sums["bc"] += float(bc_loss.detach().item())
            loss_sums["pde"] += float(pde_loss.detach().item())

        model.eval()
        relative_l2 = float("nan")
        if (
            epoch == 1
            or epoch % config.evaluation_interval_epochs == 0
            or epoch == config.epochs
        ):
            relative_l2 = relative_l2_on_fixed_grid(model, config, device)
        torch.cuda.synchronize(device)
        history.append(
            {
                "epoch": float(epoch),
                "optimizer_step": float(optimizer_step),
                "epoch_elapsed_seconds": time.perf_counter() - epoch_start,
                "total_loss": loss_sums["total"] / config.batches_per_epoch,
                "initial_condition_loss": loss_sums["ic"] / config.batches_per_epoch,
                "boundary_condition_loss": loss_sums["bc"] / config.batches_per_epoch,
                "pde_residual_loss": loss_sums["pde"] / config.batches_per_epoch,
                "relative_l2_fixed_grid": relative_l2,
                "learning_rate": learning_rate,
            }
        )
    return model, history, fixed_points


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


def configure_typography() -> dict[str, str]:
    """Require real Times New Roman faces and STIX mathtext before plotting."""
    resolved: dict[str, str] = {}
    for label, style, weight in (
        ("regular", "normal", "normal"),
        ("bold", "normal", "bold"),
        ("italic", "italic", "normal"),
        ("bold_italic", "italic", "bold"),
    ):
        path = font_manager.findfont(
            font_manager.FontProperties(
                family="Times New Roman", style=style, weight=weight
            ),
            fallback_to_default=False,
        )
        resolved[label] = str(Path(path).resolve())
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return resolved


def style_history_axis(axis: plt.Axes) -> None:
    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=5.0,
        width=1.0,
        labelsize=13,
        pad=5.0,
    )
    axis.minorticks_off()
    axis.grid(True, alpha=0.25, linewidth=0.7)
    for spine in axis.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")


def validate_pdf(path: Path) -> dict[str, Any]:
    """Check saved-page identity, real fonts, and the 12 pt text safety margin."""
    forbidden = ("dejavu", "liberation", "nimbus")
    with fitz.open(path) as document:
        if len(document) != 1:
            raise RuntimeError(f"Expected exactly one PDF page in {path.name}")
        page = document[0]
        font_names = sorted({entry[3] for entry in page.get_fonts(full=True)})
        compact_names = [
            name.replace("-", "").replace(" ", "").lower() for name in font_names
        ]
        if not any("timesnewroman" in name for name in compact_names):
            raise RuntimeError(
                f"Times New Roman was not embedded in {path.name}: {font_names}"
            )
        if any(any(bad in name.lower() for bad in forbidden) for name in font_names):
            raise RuntimeError(f"Fallback font found in {path.name}: {font_names}")
        text_blocks = [
            block for block in page.get_text("blocks") if str(block[4]).strip()
        ]
        if not text_blocks:
            raise RuntimeError(f"No visible text blocks found in {path.name}")
        minimum_margin = min(
            min(
                block[0],
                block[1],
                page.rect.width - block[2],
                page.rect.height - block[3],
            )
            for block in text_blocks
        )
        if minimum_margin < 12.0:
            raise RuntimeError(
                f"PDF text margin in {path.name} is {minimum_margin:.2f} pt; at least 12 pt is required"
            )
        # Exercise the post-save rasterization path used for human candidate review.
        page.get_pixmap(matrix=fitz.Matrix(144 / 72, 144 / 72), alpha=False)
        return {
            "page_count": len(document),
            "fonts": font_names,
            "minimum_text_margin_pt": float(minimum_margin),
        }


def save_training_history_pdf(
    history: list[dict[str, float]], output: Path, method_label: str | None = None
) -> dict[str, Any]:
    records = {key: np.asarray([row[key] for row in history]) for key in history[0]}
    figure, axes = plt.subplots(2, 2, figsize=(16.5, 11.0), facecolor="white")
    figure.suptitle(
        "Training History"
        if method_label is None
        else f"{method_label} Training History",
        fontsize=18,
        fontweight="bold",
        y=0.965,
    )
    panel_specs = (
        ("(a)", "Total Loss", [("total_loss", "Total loss", "#6A4C93")], "Loss"),
        (
            "(b)",
            "MSE Loss Components",
            [
                ("initial_condition_loss", r"$L_{\mathrm{IC}}$ (IC MSE)", "#4267AC"),
                ("boundary_condition_loss", r"$L_{\mathrm{BC}}$ (BC MSE)", "#FF924C"),
                ("pde_residual_loss", r"$L_{\mathrm{PDE}}$ (residual MSE)", "#FF595E"),
            ],
            "MSE loss",
        ),
        (
            "(c)",
            r"Relative $L_2$ Error",
            [("relative_l2_fixed_grid", r"Relative $L_2$ Error", "#FFCA3A")],
            r"Relative $L_2$ Error",
        ),
        (
            "(d)",
            "Learning Rate",
            [("learning_rate", "Learning rate", "#8AC926")],
            "Learning rate",
        ),
    )
    for axis, (panel_label, title, curves, y_label) in zip(
        axes.flat, panel_specs, strict=True
    ):
        for key, label, color in curves:
            linestyle = "--" if key == "learning_rate" else "-"
            finite = np.isfinite(records[key])
            axis.plot(
                records["epoch"][finite],
                records[key][finite],
                label=label,
                color=color,
                linewidth=2.0,
                linestyle=linestyle,
            )
        axis.set_yscale("log")
        axis.set_title(title, fontsize=17, pad=12)
        axis.set_xlabel("Epoch", fontsize=15, labelpad=7)
        axis.set_ylabel(y_label, fontsize=15, labelpad=7)
        axis.legend(fontsize=10, frameon=True, facecolor="white", edgecolor="#666666")
        axis.text(
            -0.14,
            1.04,
            panel_label,
            transform=axis.transAxes,
            fontsize=15,
            fontweight="bold",
        )
        style_history_axis(axis)
    figure.subplots_adjust(
        left=0.09, right=0.985, bottom=0.09, top=0.89, wspace=0.25, hspace=0.32
    )
    figure.savefig(output, format="pdf", facecolor="white", bbox_inches=None)
    plt.close(figure)
    return validate_pdf(output)


def save_solution_comparison_pdf(
    arrays: dict[str, np.ndarray],
    output: Path,
    method_label: str = "Vanilla PINN",
    equation_label: str = r"Inviscid Burgers equation: $u_t+u u_x=0$; analytical solution is used for evaluation only",
) -> dict[str, Any]:
    requested_times = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0])
    time_indices = [
        int(np.argmin(np.abs(arrays["t"] - value))) for value in requested_times
    ]
    figure, axes = plt.subplots(1, 5, figsize=(23.0, 5.8), facecolor="white")
    figure.suptitle(
        f"{method_label} and Analytical Entropy Solution",
        fontsize=18,
        fontweight="bold",
        y=0.950,
    )
    figure.text(
        0.5,
        0.905,
        equation_label,
        ha="center",
        va="center",
        fontsize=14,
    )
    for axis, time_index, requested_time in zip(
        axes, time_indices, requested_times, strict=True
    ):
        axis.plot(
            arrays["x"],
            arrays["truth"][time_index],
            color="#4267AC",
            linewidth=2.5,
            drawstyle="steps-post",
            label="Analytical entropy solution",
        )
        axis.plot(
            arrays["x"],
            arrays["prediction"][time_index],
            color="#FF595E",
            linewidth=2.0,
            label=method_label,
        )
        axis.set_title(rf"$t={requested_time:0.2f}$", fontsize=16, pad=10)
        axis.set_xlim(-1.0, 1.0)
        axis.set_ylim(-0.15, 1.15)
        axis.set_xticks(np.linspace(-1.0, 1.0, 5))
        axis.set_yticks(np.linspace(0.0, 1.0, 5))
        axis.set_xlabel(r"Position, $x$ (dimensionless)", fontsize=11, labelpad=6)
        if axis is axes[0]:
            axis.set_ylabel(r"State, $u(x,t)$ (dimensionless)", fontsize=11, labelpad=6)
            axis.legend(
                loc="lower left",
                fontsize=10,
                frameon=True,
                facecolor="white",
                edgecolor="#666666",
            )
        axis.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=4.0,
            width=1.0,
            labelsize=10,
            pad=4.0,
        )
        axis.grid(True, alpha=0.25, linewidth=0.7)
        for spine in axis.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.2)
    figure.subplots_adjust(left=0.055, right=0.975, bottom=0.19, top=0.76, wspace=0.28)
    figure.savefig(output, format="pdf", facecolor="white", bbox_inches=None)
    plt.close(figure)
    return validate_pdf(output)


def _representative_subset(values: np.ndarray, count: int) -> np.ndarray:
    """Select evenly indexed members of an existing fixed pool for display only."""
    if values.size <= count:
        return values
    return values[np.linspace(0, values.size - 1, count, dtype=int)]


def save_training_point_layout_pdf(
    fixed_points: dict[str, Tensor],
    config: TrainingConfig,
    output: Path,
    pde_display_count: int = 2400,
    equation_label: str = r"Inviscid Burgers PINN: $u_t+u u_x=0$ on $[-1,1]\times[0,1]$",
) -> dict[str, Any]:
    """Render a readable subset of the actual fixed training-coordinate pools."""

    def cpu_values(key: str) -> np.ndarray:
        return fixed_points[key].detach().cpu().numpy().reshape(-1)

    x_ic_left = _representative_subset(cpu_values("x_ic_left"), 80)
    x_ic_right = _representative_subset(cpu_values("x_ic_right"), 80)
    t_bc_left = _representative_subset(cpu_values("t_bc_left"), 80)
    t_bc_right = _representative_subset(cpu_values("t_bc_right"), 80)
    x_pde = _representative_subset(cpu_values("x_pde"), pde_display_count)
    t_pde = _representative_subset(cpu_values("t_pde"), pde_display_count)

    # A 16:9 composition keeps the figure legible when placed directly on a
    # presentation slide.  IC and BC markers are grouped by geometric role;
    # their prescribed values belong in the accompanying problem statement.
    figure, axis = plt.subplots(figsize=(13.333, 7.5), facecolor="white")
    figure.suptitle(
        "PINN Training Points in Space–Time", fontsize=20, fontweight="bold", y=0.965
    )
    figure.text(
        0.5,
        0.865,
        equation_label,
        ha="center",
        va="center",
        fontsize=14.5,
    )
    axis.scatter(
        x_pde,
        t_pde,
        s=14,
        marker="o",
        color="#6C8EBF",
        alpha=0.52,
        linewidths=0.0,
        label="PDE collocation points",
        zorder=1,
    )
    axis.scatter(
        np.concatenate((x_ic_left, x_ic_right)),
        np.zeros(x_ic_left.size + x_ic_right.size),
        s=34,
        marker="s",
        color="#E07A1F",
        edgecolors="white",
        linewidths=0.45,
        label=r"Initial-condition points ($t=0$)",
        zorder=4,
    )
    axis.scatter(
        np.concatenate(
            (
                np.full_like(t_bc_left, config.x_min),
                np.full_like(t_bc_right, config.x_max),
            )
        ),
        np.concatenate((t_bc_left, t_bc_right)),
        s=36,
        marker="^",
        color="#2A9D8F",
        edgecolors="white",
        linewidths=0.45,
        label=r"Boundary-condition points ($x=\pm1$)",
        zorder=4,
    )
    shock_times = np.linspace(config.t_min, config.t_max, 301)
    axis.plot(
        config.shock_speed * shock_times,
        shock_times,
        color="#D1495B",
        linestyle="--",
        linewidth=2.8,
        label=r"Shock path: $x_s(t)=0.5t$ (reference only)",
        zorder=3,
    )
    axis.set_xlim(config.x_min - 0.07, config.x_max + 0.07)
    axis.set_ylim(config.t_min - 0.04, config.t_max + 0.07)
    axis.set_xticks(np.linspace(config.x_min, config.x_max, 5))
    axis.set_yticks(np.linspace(config.t_min, config.t_max, 5))
    axis.set_xlabel(r"$x$", fontsize=15, labelpad=8)
    axis.set_ylabel(r"$t$", fontsize=15, labelpad=8)
    axis.set_aspect("equal", adjustable="box")
    style_history_axis(axis)
    axis.set_facecolor("#FBFCFE")
    axis.tick_params(axis="both", labelsize=12.5)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=2,
        fontsize=11.5,
        frameon=True,
        facecolor="white",
        edgecolor="#B7C0CC",
        columnspacing=1.8,
        handletextpad=0.55,
    )
    figure.text(
        0.5,
        0.055,
        f"Displayed: 160 IC + 160 BC + {x_pde.size:,} PDE points   |   "
        "Fixed pools: 4,096 IC + 4,096 BC + 16,384 PDE points   |   "
        "Shuffled, not resampled, each epoch",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#344054",
    )
    figure.subplots_adjust(left=0.085, right=0.975, bottom=0.145, top=0.70)
    figure.savefig(output, format="pdf", facecolor="white", bbox_inches=None)
    plt.close(figure)
    return validate_pdf(output)


def save_solution_animation_gif(
    arrays: dict[str, np.ndarray],
    config: TrainingConfig,
    output: Path,
    method_label: str = "Standard PINN",
    equation_label: str = r"$\frac{\partial u}{\partial t}+u\frac{\partial u}{\partial x}=0$",
    main_title: str = "Inviscid Burgers Equation",
) -> dict[str, Any]:
    """Render the final-model shock evolution without changing numerical results.

    The prediction and entropy solution arrays are evaluated before this CPU-only
    Matplotlib step.  The GIF is a candidate artifact for one trained model, not
    an animation of optimizer iterations.
    """
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing animation candidate: {output}"
        )
    frame_times = np.linspace(config.t_min, config.t_max, ANIMATION_FRAME_COUNT)
    time_indices = [
        int(np.argmin(np.abs(arrays["t"] - value))) for value in frame_times
    ]
    figure, profile_axis = plt.subplots(
        1, 1, figsize=(8.0, 7.5), facecolor="white"
    )
    figure.suptitle(
        main_title, fontsize=24, fontweight="bold", y=0.982
    )
    figure.text(
        0.5,
        0.875,
        equation_label,
        ha="center",
        va="center",
        fontsize=26 if r"\partial^2" in equation_label else 30,
    )

    profile_axis.set_title(f"{method_label} Prediction", fontsize=20, pad=34)
    profile_axis.set_xlim(config.x_min, config.x_max)
    profile_axis.set_ylim(-0.1, 1.1)
    profile_axis.set_xticks(np.linspace(config.x_min, config.x_max, 5))
    profile_axis.set_yticks(np.linspace(0.0, 1.0, 5))
    profile_axis.set_xlabel(r"$x$", fontsize=17, labelpad=8)
    profile_axis.set_ylabel(r"$u$", fontsize=17, labelpad=8)
    style_history_axis(profile_axis)
    profile_axis.tick_params(axis="both", which="major", labelsize=14)
    (exact_line,) = profile_axis.plot(
        arrays["x"],
        arrays["truth"][time_indices[0]],
        color="#2F6BBD",
        linewidth=4.0,
        drawstyle="steps-post",
        label="Exact solution",
        zorder=2,
    )
    (prediction_line,) = profile_axis.plot(
        arrays["x"],
        arrays["prediction"][time_indices[0]],
        color="#D1495B",
        linewidth=3.5,
        label=f"{method_label} prediction",
        zorder=3,
    )
    profile_axis.legend(
        loc="center left",
        fontsize=14,
        frameon=True,
        facecolor="white",
        edgecolor="#AAB4C0",
    )
    time_text = profile_axis.text(
        0.5,
        1.018,
        "",
        transform=profile_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=15,
    )
    profile_axis.set_position([0.145, 0.115, 0.82, 0.585])

    def update(frame: int) -> tuple[Any, ...]:
        time_index = time_indices[frame]
        time_value = frame_times[frame]
        exact_line.set_ydata(arrays["truth"][time_index])
        prediction_line.set_ydata(arrays["prediction"][time_index])
        time_text.set_text(rf"$t={time_value:0.2f}$")
        return exact_line, prediction_line, time_text

    animation = FuncAnimation(
        figure,
        update,
        frames=ANIMATION_FRAME_COUNT,
        interval=1000 / ANIMATION_FPS,
        blit=False,
        repeat=True,
    )
    animation.save(output, writer=PillowWriter(fps=ANIMATION_FPS), dpi=ANIMATION_DPI)
    plt.close(figure)
    with Image.open(output) as image:
        frame_count = int(getattr(image, "n_frames", 1))
        if frame_count != ANIMATION_FRAME_COUNT:
            raise RuntimeError(
                f"Expected {ANIMATION_FRAME_COUNT} GIF frames, found {frame_count}"
            )
        if image.info.get("loop") != 0:
            raise RuntimeError("GIF must loop continuously")
        return {
            "frame_count": frame_count,
            "size_px": list(image.size),
            "loop": image.info.get("loop"),
            "duration_ms": image.info.get("duration"),
            "quantity": "final-model prediction and analytical entropy solution over time",
        }


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


def build_run_metadata(
    config: TrainingConfig, model: nn.Module, device: torch.device
) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "completed": False,
        "scientific_contract": {
            "equation": "u_t + u u_x = 0",
            "domain": {
                "x": [config.x_min, config.x_max],
                "t": [config.t_min, config.t_max],
            },
            "initial_condition": "u(x, 0)=1 for x<0 and 0 for x>=0",
            "boundary_condition": "u(-1,t)=1; u(1,t)=0",
            "analytical_entropy_shock": "x_s(t)=0.5 t; evaluation only",
            "training_method": "vanilla strong-form PINN with automatic differentiation",
            "sampling": {
                "mode": "fixed uniform coordinate pools, shuffled mini-batch Adam",
                "fixed_initial_condition_points": 2 * config.initial_points_per_side,
                "fixed_boundary_condition_points": 2 * config.boundary_points_per_side,
                "fixed_pde_points": config.pde_points,
                "total_fixed_points": config.total_fixed_points,
                "points_per_batch": {
                    "initial_condition": 2 * config.initial_points_per_side_batch,
                    "boundary_condition": 2 * config.boundary_points_per_side_batch,
                    "pde": config.pde_batch,
                    "total": config.total_batch_size,
                },
                "batches_per_epoch": config.batches_per_epoch,
                "coverage": "every fixed IC, BC, and PDE point occurs exactly once per epoch",
            },
        },
        "configuration": asdict(config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": {
            "type": device.type,
            "visible_device_count": torch.cuda.device_count(),
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": int(properties.total_memory),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "candidate_only": True,
        "candidate_promotion_requirement": "manual saved-PDF and visual review before formal publication",
    }


def main() -> None:
    if CANDIDATE_DIR.exists():
        raise FileExistsError(
            f"Candidate directory already exists: {CANDIDATE_DIR}. Review or recoverably archive it before rerunning."
        )
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=False)
    device = require_cuda()
    set_reproducible_seed(CONFIG.seed)
    model, history, fixed_points = train_baseline(CONFIG, device)
    metrics, arrays = evaluate_final_model(model, CONFIG, device)

    # Preserve the numerical candidate before presentation validation. A figure
    # failure must not erase an otherwise inspectable training result.
    write_history_csv(history, CANDIDATE_DIR / "training_history.csv")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "configuration": asdict(CONFIG),
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        },
        CANDIDATE_DIR / "checkpoint_final.pt",
    )

    resolved_fonts = configure_typography()
    history_pdf = CANDIDATE_DIR / "training_loss_curves.pdf"
    comparison_pdf = CANDIDATE_DIR / "burgers_solution_comparison.pdf"
    point_layout_pdf = CANDIDATE_DIR / "training_point_layout.pdf"
    animation_gif = CANDIDATE_DIR / "burgers_shock_motion.gif"
    plot_validation = {
        "training_loss_curves.pdf": save_training_history_pdf(history, history_pdf),
        "burgers_solution_comparison.pdf": save_solution_comparison_pdf(
            arrays, comparison_pdf
        ),
        "training_point_layout.pdf": save_training_point_layout_pdf(
            fixed_points, CONFIG, point_layout_pdf
        ),
    }
    animation_validation = save_solution_animation_gif(arrays, CONFIG, animation_gif)

    metadata = build_run_metadata(CONFIG, model, device)
    metadata["resolved_fonts"] = resolved_fonts
    metadata["metrics"] = metrics
    metadata["plot_validation"] = plot_validation
    metadata["animation"] = {"file": animation_gif.name, **animation_validation}
    metadata["completed"] = True
    with (CANDIDATE_DIR / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metadata, indent=2), flush=True)


def rerender_history_candidate() -> None:
    """Repair only the candidate history PDF from saved logs; never retrain."""
    history_path = CANDIDATE_DIR / "training_history.csv"
    result_path = CANDIDATE_DIR / "result.json"
    if not history_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(
            "Candidate history/result metadata is required for a rendering-only repair."
        )
    history = read_history_csv(history_path)
    if not history:
        raise RuntimeError("Candidate training history is empty.")
    resolved_fonts = configure_typography()
    validation = save_training_history_pdf(
        history, CANDIDATE_DIR / "training_loss_curves.pdf"
    )
    with result_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata["resolved_fonts"] = resolved_fonts
    metadata["plot_validation"]["training_loss_curves.pdf"] = validation
    metadata["history_pdf_rendering"] = (
        "finite Relative L2 samples are connected across evaluation epochs"
    )
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"training_loss_curves.pdf": validation}, indent=2), flush=True)


def render_candidate_animation(candidate_dir: Path) -> None:
    """Add a GIF to a completed candidate in a Slurm GPU allocation only."""
    result_path = candidate_dir / "result.json"
    checkpoint_path = candidate_dir / "checkpoint_final.pt"
    if not result_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            "A completed candidate must contain result.json and checkpoint_final.pt."
        )
    with result_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not metadata.get("completed"):
        raise RuntimeError(f"Candidate is not marked completed: {candidate_dir}")
    config_field_names = {field.name for field in fields(TrainingConfig)}
    config = TrainingConfig(
        **{
            key: value
            for key, value in metadata["configuration"].items()
            if key in config_field_names
        }
    )
    device = require_cuda()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = VanillaBurgersPINN(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _, arrays = evaluate_final_model(model, config, device)
    configure_typography()
    viscosity = float(metadata["configuration"].get("global_viscosity", 0.0))
    if viscosity > 0.0:
        method_label = "Artificial-Viscosity PINN"
        equation_label = (
            r"$\frac{\partial u}{\partial t}+u\frac{\partial u}{\partial x}"
            r"-\nu\frac{\partial^2u}{\partial x^2}=0,\quad \nu=" + f"{viscosity:g}" + "$"
        )
        main_title = "Regularized Burgers Equation"
    else:
        method_label = "Standard PINN"
        equation_label = (
            r"$\frac{\partial u}{\partial t}+u\frac{\partial u}{\partial x}=0$"
        )
        main_title = "Inviscid Burgers Equation"
    animation_path = candidate_dir / "burgers_shock_motion.gif"
    animation_validation = save_solution_animation_gif(
        arrays,
        config,
        animation_path,
        method_label=method_label,
        equation_label=equation_label,
        main_title=main_title,
    )
    metadata["animation"] = {
        "file": animation_path.name,
        **animation_validation,
        "layout": "single-panel portrait",
        "comparison": f"exact solution versus final {method_label} prediction",
    }
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {"candidate": str(candidate_dir), "animation": animation_validation},
            indent=2,
        ),
        flush=True,
    )


def render_candidate_training_point_layout(candidate_dir: Path) -> None:
    """Reconstruct the saved run's deterministic fixed pools and render their layout."""
    result_path = candidate_dir / "result.json"
    checkpoint_path = candidate_dir / "checkpoint_final.pt"
    if not result_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            "A completed candidate must contain result.json and checkpoint_final.pt."
        )
    with result_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not metadata.get("completed"):
        raise RuntimeError(f"Candidate is not marked completed: {candidate_dir}")
    config_names = {field.name for field in fields(TrainingConfig)}
    config = TrainingConfig(**{key: value for key, value in metadata["configuration"].items() if key in config_names})
    device = require_cuda()
    # The original sequence seeds, initializes the model, then samples fixed pools.
    # Replaying that exact sequence recovers the training coordinates without retraining.
    set_reproducible_seed(config.seed)
    VanillaBurgersPINN(config).to(device)
    fixed_points = build_fixed_training_points(config, device)
    configure_typography()
    layout_path = candidate_dir / "training_point_layout.pdf"
    validation = save_training_point_layout_pdf(fixed_points, config, layout_path)
    print(
        json.dumps(
            {"candidate": str(candidate_dir), "training_point_layout.pdf": validation},
            indent=2,
        ),
        flush=True,
    )


def render_candidate_dense_training_point_layout(candidate_dir: Path) -> None:
    """Add a denser, separately named view without overwriting an existing layout."""
    result_path = candidate_dir / "result.json"
    checkpoint_path = candidate_dir / "checkpoint_final.pt"
    if not result_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            "A completed candidate must contain result.json and checkpoint_final.pt."
        )
    with result_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not metadata.get("completed"):
        raise RuntimeError(f"Candidate is not marked completed: {candidate_dir}")
    config_names = {field.name for field in fields(TrainingConfig)}
    config = TrainingConfig(**{key: value for key, value in metadata["configuration"].items() if key in config_names})
    device = require_cuda()
    set_reproducible_seed(config.seed)
    VanillaBurgersPINN(config).to(device)
    fixed_points = build_fixed_training_points(config, device)
    configure_typography()
    layout_path = candidate_dir / "training_point_layout_dense.pdf"
    validation = save_training_point_layout_pdf(
        fixed_points, config, layout_path, pde_display_count=2400
    )
    print(
        json.dumps(
            {
                "candidate": str(candidate_dir),
                "training_point_layout_dense.pdf": validation,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--rerender-history-candidate",
        action="store_true",
        help="Re-render the existing candidate history PDF without training or GPU use.",
    )
    mode.add_argument(
        "--render-candidate-animation",
        type=Path,
        metavar="CANDIDATE_DIRECTORY",
        help="Render a new final-model GIF inside one completed candidate directory on a GPU node.",
    )
    mode.add_argument(
        "--render-candidate-training-point-layout",
        type=Path,
        metavar="CANDIDATE_DIRECTORY",
        help="Reconstruct and plot one completed candidate's fixed training coordinates on a GPU node.",
    )
    mode.add_argument(
        "--render-candidate-dense-training-point-layout",
        type=Path,
        metavar="CANDIDATE_DIRECTORY",
        help="Add a denser separately named fixed-point-layout PDF on a GPU node.",
    )
    arguments = parser.parse_args()
    if arguments.rerender_history_candidate:
        rerender_history_candidate()
    elif arguments.render_candidate_animation:
        render_candidate_animation(arguments.render_candidate_animation)
    elif arguments.render_candidate_training_point_layout:
        render_candidate_training_point_layout(
            arguments.render_candidate_training_point_layout
        )
    elif arguments.render_candidate_dense_training_point_layout:
        render_candidate_dense_training_point_layout(
            arguments.render_candidate_dense_training_point_layout
        )
    else:
        main()
