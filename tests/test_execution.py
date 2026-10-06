"""Exercise the public entry points without plots or any full training run."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from burgers_pinn import evaluate, train
from burgers_pinn.runtime import smoke_config


class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # patch.dict restores its complete sys.modules snapshot on exit. Load
        # Adam's lazy backend first so subsequent tests do not re-register its
        # native operators after the Python module cache is restored.
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([parameter])
        parameter.sum().backward()
        optimizer.step()

    def test_training_and_evaluation_work_without_plotting_and_preserve_metadata(self):
        # Block the plotting module entirely: numerical use must not need its
        # font, Matplotlib, or PDF imports. Only the full-run budget is reduced.
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            sys.modules, {"burgers_pinn.plotting": None}
        ), contextlib.redirect_stdout(io.StringIO()):
            for method, flags, smoke in (
                ("standard", ["--smoke-test"], True),
                ("viscosity", ["--skip-plots"], False),
            ):
                with self.subTest(method=method):
                    output = Path(directory) / method
                    argv = ["train", "--method", method, "--output-dir", str(output),
                            "--device", "cpu", *flags]
                    with patch.object(sys, "argv", argv), patch.object(
                        train, "TrainingConfig", return_value=smoke_config()
                    ):
                        train.main()
                    trained = json.loads((output / "result.json").read_text(encoding="utf-8"))
                    self.assertTrue(trained["completed"])
                    self.assertEqual(trained["smoke_test"], smoke)
                    self.assertEqual(trained["optimizer_steps"], 4)
                    self.assertFalse(trained["plots_enabled"])

                    evaluated_dir = Path(directory) / f"{method}-evaluation"
                    argv = ["evaluate", "--checkpoint", str(output / "checkpoint_final.pt"),
                            "--output-dir", str(evaluated_dir), "--device", "cpu"]
                    if not smoke:
                        argv.append("--skip-plots")
                    with patch.object(sys, "argv", argv):
                        evaluate.main()
                    evaluated = json.loads((evaluated_dir / "result.json").read_text(encoding="utf-8"))
                    for key in ("method", "smoke_test", "configuration", "metrics"):
                        self.assertEqual(trained[key], evaluated[key])
                    with np.load(output / "evaluation_fields.npz") as trained_arrays, np.load(
                        evaluated_dir / "evaluation_fields.npz"
                    ) as evaluated_arrays:
                        for key in trained_arrays.files:
                            np.testing.assert_array_equal(trained_arrays[key], evaluated_arrays[key])
                    for path in (output, evaluated_dir):
                        self.assertFalse(list(path.glob("*.pdf")))
                        self.assertFalse(list(path.glob("*.gif")))

                    original_report = (evaluated_dir / "result.json").read_bytes()
                    with patch.object(sys, "argv", argv), self.assertRaises(FileExistsError):
                        evaluate.main()
                    self.assertEqual((evaluated_dir / "result.json").read_bytes(), original_report)

    def test_invalid_seed_fails_before_creating_an_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid-seed"
            argv = ["train", "--method", "standard", "--output-dir", str(output),
                    "--device", "cpu", "--smoke-test", "--seed", "-1"]
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "seed"):
                train.main()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
