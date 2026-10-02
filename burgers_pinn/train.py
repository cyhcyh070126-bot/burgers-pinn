"""Train the standard or fixed global artificial-viscosity Burgers PINN."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

from .core import TrainingConfig, evaluate_final_model, set_reproducible_seed, train_scalar, write_history_csv
from .runtime import (device_metadata, render_outputs, require_new_directory,
                      resolve_viscosity, save_checkpoint, scientific_contract,
                      select_device, smoke_config, validate_config,
                      write_evaluation_arrays, write_json)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("standard", "viscosity"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory; existing paths are refused.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--viscosity", type=float, default=1e-3, help="Positive global nu for the viscosity method (default 0.001).")
    parser.add_argument("--epochs", type=int, help="Override the original 1,000 epochs (changes training budget).")
    parser.add_argument("--seed", type=int, help="Override the original random seed 1234.")
    parser.add_argument("--smoke-test", action="store_true", help="Tiny 2-epoch/4-update check; reduced model and pools; no research plots.")
    parser.add_argument("--skip-plots", action="store_true", help="Save numerical outputs without requiring fonts or generating PDF/GIF.")
    args = parser.parse_args()
    config = smoke_config() if args.smoke_test else TrainingConfig()
    if args.smoke_test and args.epochs is not None:
        parser.error("--smoke-test has a fixed tiny budget; do not combine it with --epochs.")
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    validate_config(config)
    viscosity = resolve_viscosity(args.method, args.viscosity)
    device = select_device(args.device)
    plots_enabled = not (args.skip_plots or args.smoke_test)
    if plots_enabled:
        from .plotting import configure_typography
        configure_typography()  # Fail before training if the original font contract cannot be met.
    require_new_directory(args.output_dir)
    metadata = {
        "completed": False, "method": args.method, "smoke_test": args.smoke_test,
        "configuration": {**asdict(config), "global_viscosity": viscosity},
        "scientific_contract": scientific_contract(viscosity),
        "device": device_metadata(device),
        "plots_enabled": plots_enabled,
        "checkpoint_selection": "Final optimizer step; no evaluation-based model selection.",
    }
    write_json(args.output_dir / "result.json", metadata)
    print(f"Training {args.method} on {device}: {config.epochs} epochs, {config.total_optimizer_steps} updates; smoke_test={args.smoke_test}", flush=True)
    set_reproducible_seed(config.seed)
    model, history, pools = train_scalar(config, device, viscosity)
    model.eval()
    metrics, arrays = evaluate_final_model(model, config, device)
    write_history_csv(history, args.output_dir / "training_history.csv")
    save_checkpoint(args.output_dir / "checkpoint_final.pt", model, config, viscosity, pools, args.smoke_test)
    write_evaluation_arrays(args.output_dir, arrays)
    metadata.update({"metrics": metrics, "parameter_count": sum(p.numel() for p in model.parameters()),
                     "optimizer_steps": config.total_optimizer_steps})
    if plots_enabled:
        metadata.update(render_outputs(args.output_dir, config, viscosity, arrays, history=history, pools=pools))
    metadata["completed"] = True
    write_json(args.output_dir / "result.json", metadata)
    print(f"Completed: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
