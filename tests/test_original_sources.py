"""Compare published numerical code to preserved original Burgers sources."""
import ast
import dataclasses
import hashlib
import importlib.util
import json
import sys
import unittest
import contextlib
import io
import tempfile
from unittest.mock import patch
from pathlib import Path

import torch
from burgers_pinn import core

ROOT = Path(__file__).resolve().parents[1]


def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class OriginalSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.original = load('vanilla_burgers_pinn_baseline', ROOT/'source_archive/pinn_burgers_common_plotting.py')
        cls.dd_original = load('xpinn_burgers_suite', ROOT/'source_archive/pinn_burgers_dd_plotting.py')
        cls.formal = load('source_original_formal', ROOT/'source_archive/pinn_burgers_formal_rebuild.py')
        cls.common_copy = load('pinn_burgers_common_plotting',ROOT/'research_scripts/pinn_burgers_common_plotting.py')
        cls.dd_copy = load('pinn_burgers_dd_plotting',ROOT/'research_scripts/pinn_burgers_dd_plotting.py')
        cls.formal_copy = load('source_research_formal',ROOT/'research_scripts/pinn_burgers_formal_rebuild.py')

    def test_source_bytes_and_all_four_imports(self):
        manifest = json.loads((ROOT/'docs/complete-source-manifest.json').read_text(encoding='utf8'))
        self.assertEqual(len(manifest['files']),4)
        for entry in manifest['files']:
            self.assertEqual(hashlib.sha256((ROOT/entry['archive']).read_bytes()).hexdigest(),entry['sha256'])
            self.assertEqual(hashlib.sha256((ROOT/entry['research_copy']).read_bytes()).hexdigest(),entry['research_sha256'])
        load('research_shock_reference',ROOT/'research_scripts/pinn_burgers_shock_reference.py')

    def test_formal_configuration_pools_batches_and_model(self):
        config_a,config_b = self.original.TrainingConfig(),core.TrainingConfig()
        self.assertEqual(dataclasses.asdict(config_a),dataclasses.asdict(config_b))
        torch.manual_seed(1234)
        a = self.original.VanillaBurgersPINN(config_a)
        torch.manual_seed(1234)
        b = core.VanillaBurgersPINN(config_b)
        for key,value in a.state_dict().items():
            self.assertTrue(torch.equal(value,b.state_dict()[key]))
        device = torch.device('cpu')
        torch.manual_seed(21); pool_a = self.original.build_fixed_training_points(config_a,device)
        torch.manual_seed(21); pool_b = core.build_fixed_training_points(config_b,device)
        for key in pool_a:
            self.assertTrue(torch.equal(pool_a[key],pool_b[key]))
        torch.manual_seed(22); permutation_a = self.original.fixed_epoch_permutations(pool_a,device)
        torch.manual_seed(22); permutation_b = core.fixed_epoch_permutations(pool_b,device)
        for index in range(config_a.batches_per_epoch):
            ba = self.original.fixed_training_batch(pool_a,permutation_a,index,config_a)
            bb = core.fixed_training_batch(pool_b,permutation_b,index,config_b)
            for key in ba:
                self.assertTrue(torch.equal(ba[key],bb[key]))

    def test_residuals_gradients_and_adam_update_equal_original(self):
        config = core.TrainingConfig(hidden_width=8,hidden_layers=2)
        for viscosity in (0.,1e-3,3e-3,1e-2):
            torch.manual_seed(7)
            a,b = self.original.VanillaBurgersPINN(config),core.VanillaBurgersPINN(config)
            b.load_state_dict(a.state_dict())
            x,t = torch.rand(19,1)*2-1,torch.rand(19,1)
            ra,rb = self.formal.scalar_residual(a,x,t,viscosity),core.scalar_residual(b,x,t,viscosity)
            self.assertTrue(torch.equal(ra,rb))
            ra.square().mean().backward();rb.square().mean().backward()
            for pa,pb in zip(a.parameters(),b.parameters()):
                self.assertTrue(torch.equal(pa.grad,pb.grad))
            torch.optim.Adam(a.parameters(),lr=1e-3).step()
            torch.optim.Adam(b.parameters(),lr=1e-3).step()
            for pa,pb in zip(a.parameters(),b.parameters()):
                self.assertTrue(torch.equal(pa,pb))

    def test_research_dd_rejects_partial_pool_without_changing_defaults(self):
        self.assertEqual(self.dd_copy.XPINNConfig().batches_per_epoch,32)
        with self.assertRaises(ValueError):
            _ = self.dd_copy.XPINNConfig(interface_points=1025).batches_per_epoch
        device = torch.device('cpu')
        torch.manual_seed(42);a = self.dd_original.build_pools(self.dd_original.XPINNConfig(),device)
        torch.manual_seed(42);b = self.dd_copy.build_pools(self.dd_copy.XPINNConfig(),device)
        for key in a:
            self.assertTrue(torch.equal(a[key],b[key]))

    def test_saved_viscosity_metadata_and_animation_label(self):
        common = self.common_copy
        config = common.TrainingConfig(hidden_width=8,hidden_layers=2)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory/'result.json').write_text(json.dumps({'completed':True,'configuration':{**dataclasses.asdict(config),'global_viscosity':.003}}),encoding='utf8')
            model = common.VanillaBurgersPINN(config)
            torch.save({'model_state_dict':model.state_dict()},directory/'checkpoint_final.pt')
            with patch.object(common,'require_cuda',return_value=torch.device('cpu')), patch.object(common,'configure_typography',return_value={}), patch.object(common,'save_training_point_layout_pdf',return_value={}) as layout, contextlib.redirect_stdout(io.StringIO()):
                common.render_candidate_training_point_layout(directory)
                common.render_candidate_dense_training_point_layout(directory)
                self.assertEqual(layout.call_count,2)
                with patch.object(common,'evaluate_final_model',return_value=({},{})), patch.object(common,'save_solution_animation_gif',return_value={}) as animation:
                    common.render_candidate_animation(directory)
                    self.assertIn('0.003',animation.call_args.kwargs['equation_label'])


if __name__ == '__main__':
    unittest.main()
