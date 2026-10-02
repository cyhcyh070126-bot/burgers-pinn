"""Evaluate a saved final checkpoint without further optimization."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .core import evaluate_final_model
from .runtime import (device_metadata, load_checkpoint, prepare_plotting, render_outputs,
                      require_new_directory, scientific_contract, select_device,
                      write_evaluation_arrays, write_json)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not already exist.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-plots", action="store_true", help="Write metrics and arrays only.")
    args = parser.parse_args()
    device = select_device(args.device)
    model, config, viscosity, payload = load_checkpoint(args.checkpoint, device)
    smoke = bool(payload.get("smoke_test", False))
    plots_enabled = not (args.skip_plots or smoke)
    if plots_enabled:
        prepare_plotting()
    require_new_directory(args.output_dir)
    metrics, arrays = evaluate_final_model(model, config, device)
    write_evaluation_arrays(args.output_dir, arrays)
    metadata = {
        "completed": True, "evaluation_only": True, "smoke_test": smoke,
        "source_checkpoint": args.checkpoint.name,
        "method": "standard" if viscosity == 0 else "viscosity",
        "configuration": {**asdict(config), "global_viscosity": viscosity},
        "scientific_contract": scientific_contract(viscosity),
        "device": device_metadata(device), "metrics": metrics,
        "plots_enabled": plots_enabled,
    }
    if plots_enabled:
        metadata.update(render_outputs(args.output_dir, config, viscosity, arrays,
                                       pools=payload.get("training_points")))
    write_json(args.output_dir / "result.json", metadata)
    print(f"Evaluated {args.checkpoint.name}: {args.output_dir}; smoke_test={smoke}", flush=True)


if __name__ == "__main__":
    main()
