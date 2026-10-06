"""Formal, controlled rerun for the inviscid Burgers Riemann shock.

Only generic, already-validated infrastructure is reused from the clean
baseline module: the MLP, GPU guard, fixed-pool batching, diagnostics, and
publication plotting.  The scientific contracts below are declared here:
one direct strong-form baseline, a fixed-viscosity sensitivity study, and a
known-interface domain-decomposition sanity check.

The analytical entropy solution is never used in an optimizer loss.  In the
domain-decomposition run its *interface path* is explicitly prescribed and is
therefore labelled as an oracle-interface experiment.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

import pinn_burgers_common_plotting as common
import pinn_burgers_dd_plotting as prescribed_dd


PINN_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PINN_DIR / "results"
SCALAR_CONFIG = common.TrainingConfig()
GLOBAL_VISCOSITIES = (1.0e-3, 3.0e-3, 1.0e-2)
EXPECTED_ARTIFACTS = frozenset(
    {
        "checkpoint_final.pt",
        "training_history.csv",
        "result.json",
        "training_loss_curves.pdf",
        "burgers_solution_comparison.pdf",
        "training_point_layout.pdf",
        "burgers_shock_motion.gif",
    }
)


def require_new_directory(directory: Path) -> None:
    """Create exactly one fresh method directory; never overwrite evidence."""
    if directory.exists():
        raise FileExistsError(f"Formal result directory already exists: {directory}")
    directory.mkdir(parents=True, exist_ok=False)


def completed_directory(directory: Path) -> bool:
    """Skip only a formally completed directory with the exact artifact contract."""
    if not directory.exists():
        return False
    actual = {item.name for item in directory.iterdir() if item.is_file()}
    metadata_path = directory / "result.json"
    metadata_complete = False
    if metadata_path.is_file():
        metadata_complete = bool(json.loads(metadata_path.read_text(encoding="utf-8")).get("completed"))
    if actual == EXPECTED_ARTIFACTS and metadata_complete:
        print(json.dumps({"skipped_complete": directory.name}), flush=True)
        return True
    raise RuntimeError(
        f"Refusing partial or contaminated result directory {directory}: "
        f"expected={sorted(EXPECTED_ARTIFACTS)}, actual={sorted(actual)}"
    )


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
    config: common.TrainingConfig, device: torch.device, viscosity: float
) -> tuple[common.VanillaBurgersPINN, list[dict[str, float]], dict[str, Tensor]]:
    """Train one controlled scalar model with identical pools and budget."""
    model = common.VanillaBurgersPINN(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate_start)
    pools = common.build_fixed_training_points(config, device)
    history: list[dict[str, float]] = []
    step = 0
    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        permutations = common.fixed_epoch_permutations(pools, device)
        sums = {key: 0.0 for key in ("total", "ic", "bc", "pde")}
        for batch_index in range(config.batches_per_epoch):
            step += 1
            rate = common.cosine_learning_rate(step, config)
            optimizer.param_groups[0]["lr"] = rate
            item = common.fixed_training_batch(pools, permutations, batch_index, config)
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
            relative_l2 = common.relative_l2_on_fixed_grid(model, config, device)
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
    return model, history, pools


def scalar_metadata(
    *, method: str, viscosity: float, metrics: dict[str, Any], fonts: dict[str, str],
    validation: dict[str, Any], animation: dict[str, Any], device: torch.device,
) -> dict[str, Any]:
    is_vanilla = viscosity == 0.0
    equation = "u_t + (u^2/2)_x = 0" if is_vanilla else f"u_t + (u^2/2)_x - {viscosity:g} u_xx = 0"
    return {
        "completed": True,
        "formal_rebuild": True,
        "method": method,
        "scientific_contract": {
            "equation": equation,
            "domain": {"x": [-1.0, 1.0], "t": [0.0, 1.0]},
            "initial_condition": "u(x,0)=1 for x<0 and 0 for x>=0",
            "boundary_condition": "u(-1,t)=1; u(1,t)=0",
            "training_physics": "strong residual u_t+u u_x" if is_vanilla else "strong residual u_t+u u_x-nu u_xx",
            "analytical_entropy_solution": "x_s(t)=0.5t; excluded from optimizer losses",
            "interpretation": (
                "Direct inviscid strong-form baseline."
                if is_vanilla
                else "Regularized-model sensitivity run; inviscid entropy metrics are diagnostics, not same-PDE ranking metrics."
            ),
        },
        "configuration": {**asdict(SCALAR_CONFIG), "global_viscosity": viscosity},
        "parameter_count": sum(parameter.numel() for parameter in common.VanillaBurgersPINN(SCALAR_CONFIG).parameters()),
        "device": {
            "type": "cuda",
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "torch_version": torch.__version__,
        },
        "metrics": metrics,
        "resolved_fonts": fonts,
        "plot_validation": validation,
        "animation": {"file": "burgers_shock_motion.gif", **animation},
    }


def run_scalar(method: str, viscosity: float, directory_name: str, device: torch.device) -> None:
    output = RESULTS_DIR / directory_name
    require_new_directory(output)
    common.set_reproducible_seed(SCALAR_CONFIG.seed)
    model, history, pools = train_scalar(SCALAR_CONFIG, device, viscosity)
    metrics, arrays = common.evaluate_final_model(model, SCALAR_CONFIG, device)
    common.write_history_csv(history, output / "training_history.csv")
    torch.save(
        {"model_state_dict": model.state_dict(), "configuration": asdict(SCALAR_CONFIG), "global_viscosity": viscosity},
        output / "checkpoint_final.pt",
    )
    fonts = common.configure_typography()
    equation_label = (
        r"Inviscid Burgers equation: $u_t+u u_x=0$; analytical solution used only for evaluation"
        if viscosity == 0.0
        else rf"Regularized Burgers: $u_t+u u_x-{viscosity:g}u_{{xx}}=0$; inviscid entropy solution is a diagnostic"
    )
    validation = {
        "training_loss_curves.pdf": common.save_training_history_pdf(history, output / "training_loss_curves.pdf", method),
        "burgers_solution_comparison.pdf": common.save_solution_comparison_pdf(
            arrays, output / "burgers_solution_comparison.pdf", method, equation_label
        ),
        "training_point_layout.pdf": common.save_training_point_layout_pdf(
            pools, SCALAR_CONFIG, output / "training_point_layout.pdf", equation_label=equation_label
        ),
    }
    animation = common.save_solution_animation_gif(
        arrays, SCALAR_CONFIG, output / "burgers_shock_motion.gif", method, equation_label
    )
    metadata = scalar_metadata(
        method=method, viscosity=viscosity, metrics=metrics, fonts=fonts,
        validation=validation, animation=animation, device=device,
    )
    (output / "result.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed": directory_name, "metrics": metrics}, indent=2), flush=True)


def run_prescribed_interface(device: torch.device) -> None:
    """Controlled oracle-interface sanity check; no automatic shock discovery claim."""
    config = prescribed_dd.XPINNConfig()
    output = RESULTS_DIR / "formal_prescribed_interface_dd_8x64"
    require_new_directory(output)
    common.set_reproducible_seed(config.seed)
    left, right = common.VanillaBurgersPINN(config).to(device), common.VanillaBurgersPINN(config).to(device)
    optimizer = torch.optim.Adam(list(left.parameters()) + list(right.parameters()), lr=config.learning_rate_start)
    pools = prescribed_dd.build_pools(config, device)
    history: list[dict[str, float]] = []
    step = 0
    for epoch in range(1, config.epochs + 1):
        sums = {key: 0.0 for key in ("total", "ic", "bc", "pde", "rh")}
        for batch_index in range(config.batches_per_epoch):
            step += 1
            optimizer.param_groups[0]["lr"] = prescribed_dd.learning_rate(step, config)
            item = prescribed_dd.batch(pools, batch_index, config)
            optimizer.zero_grad(set_to_none=True)
            ic = torch.mean((left(item["x_ic_left"], torch.zeros_like(item["x_ic_left"])) - 1.0).square()) + torch.mean((right(item["x_ic_right"], torch.zeros_like(item["x_ic_right"])) - 0.0).square())
            bc = torch.mean((left(-torch.ones_like(item["t_bc_left"]), item["t_bc_left"]) - 1.0).square()) + torch.mean((right(torch.ones_like(item["t_bc_right"]), item["t_bc_right"]) - 0.0).square())
            pde = torch.mean(common.burgers_residual(left, item["x_pde_left"], item["t_pde_left"]).square()) + torch.mean(common.burgers_residual(right, item["x_pde_right"], item["t_pde_right"]).square())
            u_left, u_right = left(item["x_interface"], item["t_interface"]), right(item["x_interface"], item["t_interface"])
            rh = torch.mean((0.5 * u_left.square() - 0.5 * u_right.square() - config.shock_speed * (u_left - u_right)).square())
            total = ic + bc + pde + rh
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite prescribed-interface loss at epoch {epoch}, step {step}")
            total.backward()
            optimizer.step()
            for key, value in (("total", total), ("ic", ic), ("bc", bc), ("pde", pde), ("rh", rh)):
                sums[key] += float(value.detach())
        relative = float("nan")
        if epoch == 1 or epoch % config.evaluation_interval_epochs == 0 or epoch == config.epochs:
            relative = prescribed_dd.evaluate(left, right, config, device)[0]["relative_l2_fixed_grid"]
        history.append({
            "epoch": float(epoch), "total_loss": sums["total"] / config.batches_per_epoch,
            "ic_loss": sums["ic"] / config.batches_per_epoch,
            "bc_loss": sums["bc"] / config.batches_per_epoch,
            "pde_loss": sums["pde"] / config.batches_per_epoch,
            "rh_loss": sums["rh"] / config.batches_per_epoch,
            "relative_l2": relative, "learning_rate": optimizer.param_groups[0]["lr"],
        })
    metrics, arrays = prescribed_dd.evaluate(left, right, config, device)
    common.write_history_csv(history, output / "training_history.csv")
    torch.save({"net_left_state_dict": left.state_dict(), "net_right_state_dict": right.state_dict(), "configuration": asdict(config)}, output / "checkpoint_final.pt")
    fonts = common.configure_typography()
    equation_label = r"Inviscid Burgers equation with prescribed interface $x_s(t)=0.5t$; entropy state values excluded from training"
    validation = {
        "training_loss_curves.pdf": prescribed_dd.save_history(history, output / "training_loss_curves.pdf"),
        "burgers_solution_comparison.pdf": common.save_solution_comparison_pdf(arrays, output / "burgers_solution_comparison.pdf", "Prescribed-Interface DD PINN", equation_label),
        "training_point_layout.pdf": prescribed_dd.save_layout(pools, config, output / "training_point_layout.pdf"),
    }
    animation = common.save_solution_animation_gif(arrays, config, output / "burgers_shock_motion.gif", "Prescribed-Interface DD PINN", equation_label)
    metadata = {
        "completed": True, "formal_rebuild": True, "method": "prescribed-interface domain-decomposition PINN",
        "scientific_contract": {
            "equation": "u_t+(u^2/2)_x=0", "interface": "x_s(t)=0.5t prescribed from analytical Riemann solution",
            "interface_loss": "Rankine--Hugoniot residual [u^2/2]-s[u]", "interpretation": "Oracle-interface sanity check; not an unknown-shock solver.",
        },
        "configuration": asdict(config), "parameter_count": sum(p.numel() for p in left.parameters()) + sum(p.numel() for p in right.parameters()),
        "device": {"type": "cuda", "name": torch.cuda.get_device_name(device), "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory, "torch_version": torch.__version__},
        "metrics": metrics, "resolved_fonts": fonts, "plot_validation": validation,
        "animation": {"file": "burgers_shock_motion.gif", **animation},
    }
    (output / "result.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed": output.name, "metrics": metrics}, indent=2), flush=True)


def main() -> None:
    device = common.require_cuda()
    vanilla_directory = RESULTS_DIR / "formal_vanilla_strongform_8x64"
    if not completed_directory(vanilla_directory):
        run_scalar("Formal Vanilla PINN", 0.0, vanilla_directory.name, device)
    for viscosity, directory in zip(
        GLOBAL_VISCOSITIES,
        ("formal_global_viscosity_nu_1e-3_8x64", "formal_global_viscosity_nu_3e-3_8x64", "formal_global_viscosity_nu_1e-2_8x64"),
        strict=True,
    ):
        if not completed_directory(RESULTS_DIR / directory):
            run_scalar(rf"Global Artificial-Viscosity PINN ($\nu={viscosity:g}$)", viscosity, directory, device)
    if not completed_directory(RESULTS_DIR / "formal_prescribed_interface_dd_8x64"):
        run_prescribed_interface(device)


if __name__ == "__main__":
    main()
