"""GPU-only prescribed-interface XPINN for the inviscid Burgers shock.

This is deliberately labelled *oracle interface*: the known analytical path
``x_s(t)=0.5t`` partitions the two smooth subdomains.  The analytical state is
never used in the training loss; only the interface path and the
Rankine--Hugoniot condition are imposed.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

# Matplotlib import must precede fitz through the shared baseline module.
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn

from vanilla_burgers_pinn_baseline import (
    TrainingConfig,
    VanillaBurgersPINN,
    burgers_residual,
    configure_typography,
    estimate_level_crossing,
    exact_entropy_solution,
    require_cuda,
    save_solution_animation_gif,
    save_solution_comparison_pdf,
    set_reproducible_seed,
    style_history_axis,
    validate_pdf,
)


PINN_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PINN_DIR / "results"
CANDIDATE_DIR = RESULTS_DIR / "xpinn_prescribed_interface_8x64_candidate"


@dataclass(frozen=True)
class XPINNConfig(TrainingConfig):
    """Shared numerical contract plus two-subdomain XPINN point pools."""

    pde_points_per_subdomain: int = 8192
    pde_batch_per_subdomain: int = 256
    interface_points: int = 1024
    interface_batch: int = 32

    @property
    def batches_per_epoch(self) -> int:
        counts = (
            self.initial_points_per_side // self.initial_points_per_side_batch,
            self.boundary_points_per_side // self.boundary_points_per_side_batch,
            self.pde_points_per_subdomain // self.pde_batch_per_subdomain,
            self.interface_points // self.interface_batch,
        )
        if len(set(counts)) != 1 or any(value != 32 for value in counts):
            raise ValueError(f"XPINN pools must form 32 complete batches, got {counts}")
        return counts[0]

    @property
    def total_optimizer_steps(self) -> int:
        return self.epochs * self.batches_per_epoch


CONFIG = XPINNConfig()


def uniform(count: int, low: float, high: float, device: torch.device) -> Tensor:
    return low + (high - low) * torch.rand((count, 1), device=device)


def build_pools(config: XPINNConfig, device: torch.device) -> dict[str, Tensor]:
    """Sample fixed coordinates in the two prescribed smooth subdomains."""
    t_left = uniform(config.pde_points_per_subdomain, 0.0, 1.0, device)
    t_right = uniform(config.pde_points_per_subdomain, 0.0, 1.0, device)
    interface_left = config.shock_speed * t_left
    interface_right = config.shock_speed * t_right
    return {
        "x_ic_left": uniform(config.initial_points_per_side, -1.0, 0.0, device),
        "x_ic_right": uniform(config.initial_points_per_side, 0.0, 1.0, device),
        "t_bc_left": uniform(config.boundary_points_per_side, 0.0, 1.0, device),
        "t_bc_right": uniform(config.boundary_points_per_side, 0.0, 1.0, device),
        "t_pde_left": t_left,
        "x_pde_left": -1.0 + (interface_left + 1.0) * torch.rand_like(t_left),
        "t_pde_right": t_right,
        "x_pde_right": interface_right
        + (1.0 - interface_right) * torch.rand_like(t_right),
        "t_interface": uniform(config.interface_points, 0.0, 1.0, device),
    }


def batch(
    pools: dict[str, Tensor], index: int, config: XPINNConfig
) -> dict[str, Tensor]:
    """Shuffle fixed pools once per epoch, then return one balanced batch."""
    keys = {
        "ic_left": "x_ic_left",
        "ic_right": "x_ic_right",
        "bc_left": "t_bc_left",
        "bc_right": "t_bc_right",
        "pde_left": "x_pde_left",
        "pde_right": "x_pde_right",
        "interface": "t_interface",
    }
    device = pools["x_ic_left"].device
    if index == 0:
        pools["_perm"] = {
            key: torch.randperm(pools[value].shape[0], device=device)
            for key, value in keys.items()
        }  # type: ignore[assignment]
    perm: dict[str, Tensor] = pools["_perm"]  # type: ignore[assignment]
    ic_start = index * config.initial_points_per_side_batch
    bc_start = index * config.boundary_points_per_side_batch
    pde_start = index * config.pde_batch_per_subdomain
    interface_start = index * config.interface_batch
    # Use explicit indexing below; this avoids any ambiguity between x/t coordinate keys.
    ic_l = pools["x_ic_left"][
        perm["ic_left"][ic_start : ic_start + config.initial_points_per_side_batch]
    ]
    ic_r = pools["x_ic_right"][
        perm["ic_right"][ic_start : ic_start + config.initial_points_per_side_batch]
    ]
    bc_l = pools["t_bc_left"][
        perm["bc_left"][bc_start : bc_start + config.boundary_points_per_side_batch]
    ]
    bc_r = pools["t_bc_right"][
        perm["bc_right"][bc_start : bc_start + config.boundary_points_per_side_batch]
    ]
    pde_l = perm["pde_left"][pde_start : pde_start + config.pde_batch_per_subdomain]
    pde_r = perm["pde_right"][pde_start : pde_start + config.pde_batch_per_subdomain]
    interface_t = pools["t_interface"][
        perm["interface"][interface_start : interface_start + config.interface_batch]
    ]
    return {
        "x_ic_left": ic_l,
        "x_ic_right": ic_r,
        "t_bc_left": bc_l,
        "t_bc_right": bc_r,
        "x_pde_left": pools["x_pde_left"][pde_l],
        "t_pde_left": pools["t_pde_left"][pde_l],
        "x_pde_right": pools["x_pde_right"][pde_r],
        "t_pde_right": pools["t_pde_right"][pde_r],
        "x_interface": config.shock_speed * interface_t,
        "t_interface": interface_t,
    }


def learning_rate(step: int, config: XPINNConfig) -> float:
    progress = (step - 1) / max(config.total_optimizer_steps - 1, 1)
    return config.learning_rate_min + 0.5 * (
        config.learning_rate_start - config.learning_rate_min
    ) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(
    net_left: nn.Module, net_right: nn.Module, config: XPINNConfig, device: torch.device
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    x = torch.linspace(-1.0, 1.0, config.evaluation_x_points, device=device)
    t = torch.linspace(0.0, 1.0, config.evaluation_t_points, device=device)
    t_grid, x_grid = torch.meshgrid(t, x, indexing="ij")
    x_flat, t_flat = x_grid.reshape(-1, 1), t_grid.reshape(-1, 1)
    interface = config.shock_speed * t_flat
    pred = torch.where(
        x_flat < interface, net_left(x_flat, t_flat), net_right(x_flat, t_flat)
    )
    truth = exact_entropy_solution(x_flat, t_flat, config)
    pred_np = pred.reshape(t.shape[0], x.shape[0]).cpu().numpy()
    truth_np = truth.reshape(t.shape[0], x.shape[0]).cpu().numpy()
    x_np, t_np = x.cpu().numpy(), t.cpu().numpy()
    locations, thicknesses = [], []
    for time_value, profile in zip(t_np[1:], pred_np[1:], strict=True):
        locations.append(
            abs(
                estimate_level_crossing(x_np, profile, 0.5)
                - config.shock_speed * float(time_value)
            )
        )
        thicknesses.append(
            abs(
                estimate_level_crossing(x_np, profile, 0.1)
                - estimate_level_crossing(x_np, profile, 0.9)
            )
        )
    metrics = {
        "relative_l2_fixed_grid": float(
            np.linalg.norm(pred_np - truth_np) / np.linalg.norm(truth_np)
        ),
        "mean_absolute_error_fixed_grid": float(
            np.mean(np.abs(pred_np - truth_np)),
        ),
        "maximum_absolute_error_fixed_grid": float(np.max(np.abs(pred_np - truth_np))),
        "mean_shock_location_absolute_error": float(np.mean(locations)),
        "mean_predicted_shock_thickness": float(np.mean(thicknesses)),
    }
    return metrics, {
        "x": x_np,
        "t": t_np,
        "prediction": pred_np,
        "truth": truth_np,
        "absolute_error": np.abs(pred_np - truth_np),
    }


def save_history(history: list[dict[str, float]], output: Path) -> dict[str, Any]:
    records = {key: np.asarray([row[key] for row in history]) for key in history[0]}
    figure, axes = plt.subplots(2, 2, figsize=(16.5, 11.0), facecolor="white")
    figure.suptitle(
        "Prescribed-Interface XPINN Training History",
        fontsize=18,
        fontweight="bold",
        y=0.965,
    )
    specs = (
        ("Total Loss", ["total_loss"]),
        ("Constraint MSE", ["ic_loss", "bc_loss", "pde_loss", "rh_loss"]),
        (r"Relative $L_2$ Error", ["relative_l2"]),
        ("Learning Rate", ["learning_rate"]),
    )
    colors = {
        "total_loss": "#6A4C93",
        "ic_loss": "#4267AC",
        "bc_loss": "#FF924C",
        "pde_loss": "#FF595E",
        "rh_loss": "#00A6A6",
        "relative_l2": "#FFCA3A",
        "learning_rate": "#8AC926",
    }
    labels = {
        "total_loss": "Total loss",
        "ic_loss": r"$L_{\rm IC}$",
        "bc_loss": r"$L_{\rm BC}$",
        "pde_loss": r"$L_{\rm PDE}$",
        "rh_loss": r"$L_{\rm RH}$",
        "relative_l2": r"Relative $L_2$ Error",
        "learning_rate": "Learning rate",
    }
    for axis, (title, names) in zip(axes.flat, specs, strict=True):
        for name in names:
            finite = np.isfinite(records[name])
            axis.plot(
                records["epoch"][finite],
                records[name][finite],
                color=colors[name],
                label=labels[name],
                linewidth=2.0,
                linestyle="--" if name == "learning_rate" else "-",
            )
        axis.set_yscale("log")
        axis.set_title(title, fontsize=17, pad=12)
        axis.set_xlabel("Epoch", fontsize=15)
        axis.legend(fontsize=10, frameon=True)
        style_history_axis(axis)
    figure.subplots_adjust(
        left=0.085, right=0.985, bottom=0.09, top=0.89, wspace=0.24, hspace=0.32
    )
    figure.savefig(output, format="pdf", facecolor="white")
    plt.close(figure)
    return validate_pdf(output)


def save_layout(
    pools: dict[str, Tensor], config: XPINNConfig, output: Path
) -> dict[str, Any]:
    def subset(key: str, count: int) -> np.ndarray:
        values = pools[key].detach().cpu().numpy().reshape(-1)
        return values[
            np.linspace(0, values.size - 1, min(count, values.size), dtype=int)
        ]

    xl, tl = subset("x_pde_left", 1200), subset("t_pde_left", 1200)
    xr, tr = subset("x_pde_right", 1200), subset("t_pde_right", 1200)
    ti = np.linspace(0.0, 1.0, 301)
    figure, axis = plt.subplots(figsize=(14.5, 9.0), facecolor="white")
    figure.suptitle(
        "Prescribed-Interface XPINN Training-Point Layout",
        fontsize=19,
        fontweight="bold",
        y=0.965,
    )
    figure.text(
        0.5,
        0.915,
        r"Two subdomains separated by the prescribed interface $x_s(t)=0.5t$",
        ha="center",
        fontsize=14,
    )
    axis.scatter(
        xl,
        tl,
        s=14,
        color="#4267AC",
        alpha=0.38,
        linewidths=0,
        label=r"Left-subdomain PDE points",
        zorder=1,
    )
    axis.scatter(
        xr,
        tr,
        s=14,
        color="#FF924C",
        alpha=0.38,
        linewidths=0,
        label=r"Right-subdomain PDE points",
        zorder=1,
    )
    axis.plot(
        0.5 * ti,
        ti,
        color="#FF595E",
        linewidth=2.6,
        label=r"Interface $x_s(t)=0.5t$",
        zorder=3,
    )
    axis.set(
        xlim=(-1.05, 1.05),
        ylim=(-0.04, 1.07),
        xlabel=r"Position, $x$ (dimensionless)",
        ylabel=r"Time, $t$ (dimensionless)",
    )
    axis.set_aspect("equal", adjustable="box")
    style_history_axis(axis)
    axis.legend(loc="upper right", fontsize=11, frameon=True)
    figure.text(
        0.5,
        0.055,
        "Displayed: 1,200 points per subdomain. Fixed training pools: 8,192 points per subdomain plus 1,024 interface points.",
        ha="center",
        fontsize=11,
    )
    figure.subplots_adjust(left=0.105, right=0.975, bottom=0.14, top=0.84)
    figure.savefig(output, format="pdf", facecolor="white")
    plt.close(figure)
    return validate_pdf(output)


def main() -> None:
    if CANDIDATE_DIR.exists():
        raise FileExistsError(f"Candidate directory already exists: {CANDIDATE_DIR}")
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=False)
    device = require_cuda()
    set_reproducible_seed(CONFIG.seed)
    net_left, net_right = (
        VanillaBurgersPINN(CONFIG).to(device),
        VanillaBurgersPINN(CONFIG).to(device),
    )
    optimizer = torch.optim.Adam(
        list(net_left.parameters()) + list(net_right.parameters()),
        lr=CONFIG.learning_rate_start,
    )
    pools = build_pools(CONFIG, device)
    history: list[dict[str, float]] = []
    step = 0
    for epoch in range(1, CONFIG.epochs + 1):
        sums = {name: 0.0 for name in ("total", "ic", "bc", "pde", "rh")}
        for batch_index in range(CONFIG.batches_per_epoch):
            step += 1
            value = learning_rate(step, CONFIG)
            for group in optimizer.param_groups:
                group["lr"] = value
            item = batch(pools, batch_index, CONFIG)
            optimizer.zero_grad(set_to_none=True)
            ic = torch.mean(
                (net_left(item["x_ic_left"], torch.zeros_like(item["x_ic_left"])) - 1.0)
                ** 2
            ) + torch.mean(
                (
                    net_right(item["x_ic_right"], torch.zeros_like(item["x_ic_right"]))
                    - 0.0
                )
                ** 2
            )
            bc = torch.mean(
                (net_left(-torch.ones_like(item["t_bc_left"]), item["t_bc_left"]) - 1.0)
                ** 2
            ) + torch.mean(
                (
                    net_right(torch.ones_like(item["t_bc_right"]), item["t_bc_right"])
                    - 0.0
                )
                ** 2
            )
            pde = torch.mean(
                burgers_residual(net_left, item["x_pde_left"], item["t_pde_left"]) ** 2
            ) + torch.mean(
                burgers_residual(net_right, item["x_pde_right"], item["t_pde_right"])
                ** 2
            )
            ul, ur = (
                net_left(item["x_interface"], item["t_interface"]),
                net_right(item["x_interface"], item["t_interface"]),
            )
            rh = torch.mean(
                (0.5 * ul**2 - 0.5 * ur**2 - CONFIG.shock_speed * (ul - ur)) ** 2
            )
            total = ic + bc + pde + rh
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}, step {step}"
                )
            total.backward()
            optimizer.step()
            for name, loss in (
                ("total", total),
                ("ic", ic),
                ("bc", bc),
                ("pde", pde),
                ("rh", rh),
            ):
                sums[name] += float(loss.detach())
        relative = float("nan")
        if (
            epoch == 1
            or epoch % CONFIG.evaluation_interval_epochs == 0
            or epoch == CONFIG.epochs
        ):
            metrics, _ = evaluate(net_left, net_right, CONFIG, device)
            relative = metrics["relative_l2_fixed_grid"]
        history.append(
            {
                "epoch": float(epoch),
                "total_loss": sums["total"] / 32,
                "ic_loss": sums["ic"] / 32,
                "bc_loss": sums["bc"] / 32,
                "pde_loss": sums["pde"] / 32,
                "rh_loss": sums["rh"] / 32,
                "relative_l2": relative,
                "learning_rate": value,
            }
        )
    metrics, arrays = evaluate(net_left, net_right, CONFIG, device)
    with (CANDIDATE_DIR / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    torch.save(
        {
            "net_left_state_dict": net_left.state_dict(),
            "net_right_state_dict": net_right.state_dict(),
            "configuration": asdict(CONFIG),
        },
        CANDIDATE_DIR / "checkpoint_final.pt",
    )
    configure_typography()
    validations = {
        "training_loss_curves.pdf": save_history(
            history, CANDIDATE_DIR / "training_loss_curves.pdf"
        ),
        "burgers_solution_comparison.pdf": save_solution_comparison_pdf(
            arrays,
            CANDIDATE_DIR / "burgers_solution_comparison.pdf",
            "Prescribed-Interface XPINN",
        ),
        "training_point_layout.pdf": save_layout(
            pools, CONFIG, CANDIDATE_DIR / "training_point_layout.pdf"
        ),
    }
    animation = save_solution_animation_gif(
        arrays,
        CONFIG,
        CANDIDATE_DIR / "burgers_shock_motion.gif",
        "Prescribed-Interface XPINN",
    )
    metadata = {
        "completed": True,
        "method": "prescribed-interface XPINN",
        "scientific_contract": {
            "equation": "u_t+(u^2/2)_x=0",
            "interface": "x_s(t)=0.5t is prescribed from the analytical Riemann solution; oracle interface",
            "interface_loss": "Rankine--Hugoniot residual",
        },
        "configuration": asdict(CONFIG),
        "metrics": metrics,
        "plot_validation": validations,
        "animation": {"file": "burgers_shock_motion.gif", **animation},
    }
    with (CANDIDATE_DIR / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
