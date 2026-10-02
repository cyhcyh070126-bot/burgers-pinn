"""Portable execution and checkpoint helpers for the two published methods."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .core import TrainingConfig, VanillaBurgersPINN


def select_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu.")
    return torch.device(value)


def require_new_directory(path: Path) -> None:
    """Never reuse an output directory, even if it is currently empty."""
    path.mkdir(parents=True, exist_ok=False)


def validate_config(config: TrainingConfig) -> None:
    positive = (
        "hidden_width", "hidden_layers", "initial_points_per_side",
        "boundary_points_per_side", "pde_points", "initial_points_per_side_batch",
        "boundary_points_per_side_batch", "pde_batch", "epochs",
        "evaluation_interval_epochs", "inference_batch_size",
    )
    if any(not isinstance(getattr(config, name), int) or getattr(config, name) <= 0 for name in positive):
        raise ValueError("Network, sampling and training counts must be positive integers.")
    if config.evaluation_x_points < 2 or config.evaluation_t_points < 2:
        raise ValueError("Evaluation grids need at least two points per axis.")
    if (config.x_min, config.x_max, config.t_min, config.t_max, config.shock_speed) != (-1.0, 1.0, 0.0, 1.0, 0.5):
        raise ValueError("This implementation supports only the published Riemann problem.")
    if config.activation != "Tanh":
        raise ValueError("The published model uses Tanh activations.")
    if not (0 < config.learning_rate_min <= config.learning_rate_start < float("inf")):
        raise ValueError("Learning rates must be finite and positive with min <= start.")
    if config.batches_per_epoch < 1:
        raise ValueError("Fixed pools must provide at least one complete batch.")


def smoke_config() -> TrainingConfig:
    """A tiny, explicitly non-scientific execution check, never a benchmark."""
    return replace(
        TrainingConfig(), hidden_width=8, hidden_layers=2, epochs=2,
        initial_points_per_side=8, boundary_points_per_side=8, pde_points=32,
        initial_points_per_side_batch=4, boundary_points_per_side_batch=4,
        pde_batch=16, evaluation_x_points=101, evaluation_t_points=61,
        inference_batch_size=2048, evaluation_interval_epochs=1,
    )


def resolve_viscosity(method: str, viscosity: float) -> float:
    if not math.isfinite(viscosity) or viscosity <= 0:
        raise ValueError("--viscosity must be a finite positive value.")
    return 0.0 if method == "standard" else viscosity


def scientific_contract(viscosity: float) -> dict[str, Any]:
    return {
        "equation": "u_t + u u_x = 0" if viscosity == 0 else f"u_t + u u_x - {viscosity:g} u_xx = 0",
        "domain": {"x": [-1.0, 1.0], "t": [0.0, 1.0]},
        "initial_condition": "u(x,0)=1 for x<0; 0 for x>=0",
        "boundary_condition": "u(-1,t)=1; u(1,t)=0",
        "loss": "IC MSE + BC MSE + PDE residual MSE; equal weights",
        "analytical_solution_usage": "evaluation only; excluded from optimizer loss and checkpoint selection",
        "reference_shock": "x_s(t)=0.5t",
        "interpretation": (
            "Direct inviscid strong-form baseline." if viscosity == 0 else
            "Global artificial-viscosity regularization changes the PDE. Inviscid entropy-reference errors are diagnostics, not same-PDE accuracy rankings."
        ),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def device_metadata(device: torch.device) -> dict[str, str]:
    return {
        "type": device.type,
        "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": str(torch.__version__),
    }


def save_checkpoint(path: Path, model: VanillaBurgersPINN, config: TrainingConfig,
                    viscosity: float, pools: dict[str, torch.Tensor], smoke: bool) -> None:
    torch.save({
        "format_version": 1,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "configuration": asdict(config),
        "global_viscosity": viscosity,
        "training_points": {key: value.detach().cpu() for key, value in pools.items()},
        "smoke_test": smoke,
    }, path)


def load_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not {"model_state_dict", "configuration", "global_viscosity"} <= payload.keys():
        raise ValueError("Expected a Burgers checkpoint with model state, configuration and viscosity.")
    names = {field.name for field in fields(TrainingConfig)}
    values = payload["configuration"]
    if not isinstance(values, dict) or set(values) - names:
        raise ValueError("Checkpoint contains unsupported configuration fields.")
    config = TrainingConfig(**values)
    validate_config(config)
    viscosity = float(payload["global_viscosity"])
    if not math.isfinite(viscosity) or viscosity < 0:
        raise ValueError("Checkpoint viscosity must be finite and nonnegative.")
    model = VanillaBurgersPINN(config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, config, viscosity, payload


def write_evaluation_arrays(output: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(output / "evaluation_fields.npz", **arrays)


def render_outputs(output: Path, config: TrainingConfig, viscosity: float,
                   arrays: dict[str, np.ndarray], *, history=None, pools=None) -> dict[str, Any]:
    """Call the archived renderers; only the method/equation inputs differ."""
    from . import plotting
    fonts = plotting.configure_typography()
    method = "Standard PINN" if viscosity == 0 else "Artificial-Viscosity PINN"
    equation = (
        r"Inviscid Burgers equation: $u_t+u u_x=0$; analytical solution used only for evaluation"
        if viscosity == 0 else
        rf"Regularized Burgers: $u_t+u u_x-{viscosity:g}u_{{xx}}=0$; inviscid entropy solution is a diagnostic"
    )
    validation = {"burgers_solution_comparison.pdf": plotting.save_solution_comparison_pdf(
        arrays, output / "burgers_solution_comparison.pdf", method, equation)}
    if history is not None:
        validation["training_loss_curves.pdf"] = plotting.save_training_history_pdf(
            history, output / "training_loss_curves.pdf", method)
    if pools is not None:
        validation["training_point_layout.pdf"] = plotting.save_training_point_layout_pdf(
            pools, config, output / "training_point_layout.pdf", equation_label=equation)
    animation_equation = (
        r"$\frac{\partial u}{\partial t}+u\frac{\partial u}{\partial x}=0$" if viscosity == 0 else
        (r"$\frac{\partial u}{\partial t}+u\frac{\partial u}{\partial x}"
         rf"-\nu\frac{{\partial^2u}}{{\partial x^2}}=0,\quad \nu={viscosity:g}$")
    )
    animation = plotting.save_solution_animation_gif(
        arrays, config, output / "burgers_shock_motion.gif", method, animation_equation,
        "Inviscid Burgers Equation" if viscosity == 0 else "Regularized Burgers Equation")
    # Record font identities without publishing machine-specific filesystem paths.
    return {"resolved_fonts": {key: Path(value).name for key, value in fonts.items()},
            "plot_validation": validation, "animation": animation}
