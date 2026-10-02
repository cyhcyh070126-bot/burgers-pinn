"""Checks for PDE differentiation, sampling contracts and checkpoint integrity."""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch import nn

from burgers_pinn.core import (
    TrainingConfig, VanillaBurgersPINN, build_fixed_training_points,
    cosine_learning_rate, exact_entropy_solution, fixed_epoch_permutations,
    fixed_training_batch, scalar_residual, set_reproducible_seed,
)
from burgers_pinn.runtime import (
    load_checkpoint, require_new_directory, save_checkpoint,
    smoke_config, validate_config,
)


class PolynomialField(nn.Module):
    def forward(self, x, t):
        return x.square() + t


class NumericalContractTests(unittest.TestCase):
    def test_residual_matches_analytic_first_and_second_derivatives(self):
        x = torch.tensor([[-0.7], [0.0], [0.4]], dtype=torch.float64)
        t = torch.tensor([[0.2], [0.5], [0.9]], dtype=torch.float64)
        inviscid = 1.0 + (x.square() + t) * (2.0 * x)
        for nu in (0.0, 1e-3, 0.05):
            torch.testing.assert_close(scalar_residual(PolynomialField(), x, t, nu),
                                       inviscid - 2.0 * nu, rtol=1e-12, atol=1e-12)

    def test_fixed_pools_are_swept_once_and_conditions_are_exact(self):
        config = smoke_config()
        set_reproducible_seed(config.seed)
        pools = build_fixed_training_points(config, torch.device("cpu"))
        original = {name: value.clone() for name, value in pools.items()}
        permutations = fixed_epoch_permutations(pools, torch.device("cpu"))
        batches = [fixed_training_batch(pools, permutations, index, config)
                   for index in range(config.batches_per_epoch)]
        for batch in batches:
            self.assertTrue(torch.equal(batch["t_ic"], torch.zeros_like(batch["t_ic"])))
            self.assertTrue(torch.equal(batch["u_ic"], (batch["x_ic"] < 0).float()))
            self.assertTrue(torch.equal(batch["u_bc"], (batch["x_bc"] < 0).float()))
            self.assertTrue(torch.all(batch["x_bc"].abs() == 1))
        pde_values = torch.cat([batch["x_pde"] for batch in batches]).sort(dim=0).values
        torch.testing.assert_close(pde_values, pools["x_pde"].sort(dim=0).values, rtol=0, atol=0)
        for name in pools:
            torch.testing.assert_close(pools[name], original[name], rtol=0, atol=0)
        # Each permutation is exactly one traversal of its pool, without replacement.
        for indices in permutations.values():
            torch.testing.assert_close(indices.sort().values, torch.arange(indices.numel()))

    def test_formal_model_shape_and_parameter_count(self):
        model = VanillaBurgersPINN(TrainingConfig())
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 29377)
        self.assertEqual(tuple(model.layers[0].weight.shape), (64, 2))
        self.assertEqual(tuple(model.layers[-1].weight.shape), (1, 64))
        self.assertEqual(len(model.layers), 9)
        self.assertEqual(model(torch.zeros(3, 1), torch.ones(3, 1)).shape, (3, 1))

    def test_checkpoint_roundtrip_preserves_prediction_viscosity_and_points(self):
        config = smoke_config()
        model = VanillaBurgersPINN(config)
        pools = build_fixed_training_points(config, torch.device("cpu"))
        x, t = torch.rand(5, 1), torch.rand(5, 1)
        expected = model(x, t)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, config, 1e-3, pools, True)
            loaded, loaded_config, nu, payload = load_checkpoint(path, torch.device("cpu"))
            torch.testing.assert_close(loaded(x, t), expected, rtol=0, atol=0)
            self.assertEqual(asdict(config), asdict(loaded_config))
            self.assertEqual(nu, 1e-3)
            self.assertTrue(payload["smoke_test"])
            for name in pools:
                torch.testing.assert_close(payload["training_points"][name], pools[name], rtol=0, atol=0)

    def test_reference_shock_and_learning_rate_endpoints(self):
        config = TrainingConfig()
        x = torch.tensor([[-0.1], [0.0], [0.49], [0.5], [0.8]])
        t = torch.tensor([[0.0], [0.0], [1.0], [1.0], [1.0]])
        torch.testing.assert_close(exact_entropy_solution(x, t, config),
                                   torch.tensor([[1.0], [0.0], [1.0], [0.0], [0.0]]))
        self.assertEqual(config.total_optimizer_steps, 32000)
        self.assertEqual(config.total_fixed_points, 24576)
        self.assertAlmostEqual(cosine_learning_rate(1, config), 1e-3)
        self.assertAlmostEqual(cosine_learning_rate(32000, config), 1e-5)

    def test_invalid_configuration_and_existing_output_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_config(replace(TrainingConfig(), pde_points=16385))
        with self.assertRaises(ValueError):
            validate_config(replace(TrainingConfig(), epochs=0))
        with self.assertRaises(ValueError):
            validate_config(replace(TrainingConfig(), x_min=-2.0))
        for seed in (-1, 2**32):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "seed"):
                validate_config(replace(TrainingConfig(), seed=seed))
        with self.assertRaises(ValueError):
            validate_config(replace(TrainingConfig(), evaluation_x_points=1001.5))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            require_new_directory(path)
            (path / "sentinel.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                require_new_directory(path)
            self.assertEqual((path / "sentinel.txt").read_text(encoding="utf-8"), "keep")

    def test_checkpoint_rejects_incomplete_or_unknown_metadata(self):
        config = smoke_config()
        model = VanillaBurgersPINN(config)
        pools = build_fixed_training_points(config, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, config, 0.0, pools, True)
            payload = torch.load(path, weights_only=True)
            for name in ("format_version", "smoke_test", "evaluation_x_points"):
                broken = dict(payload)
                if name == "evaluation_x_points":
                    broken["configuration"] = dict(payload["configuration"])
                    del broken["configuration"][name]
                else:
                    del broken[name]
                torch.save(broken, path)
                with self.subTest(missing=name), self.assertRaises(ValueError):
                    load_checkpoint(path, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
