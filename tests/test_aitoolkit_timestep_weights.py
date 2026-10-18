import ast
import hashlib
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.util.aitoolkit_timestep_weights import aitoolkit_bell_weights, aitoolkit_timestep_weights
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.LossWeight import LossWeight

import torch


class AiToolkitTimestepWeightsTest(unittest.TestCase):
    def test_upstream_table_full_schedule(self):
        # Read the independent source checkout to detect any transcription or order error.
        source = Path(__file__).resolve().parents[2] / "ai-toolkit/toolkit/timestep_weighing/default_weighing_scheme.py"
        if not source.is_file():
            self.skipTest("Optional ai-toolkit reference checkout is absent; standalone digest/gradient fixtures remain active")
        spec = importlib.util.spec_from_file_location("upstream_weights", source)
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        expected = torch.tensor(upstream.default_weighing_scheme).flip(0)
        actual = aitoolkit_timestep_weights(torch.arange(1000), 1000)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.numel(), 1000)

    def test_other_schedule_lengths(self):
        canonical = aitoolkit_timestep_weights(torch.arange(1000), 1000)
        # sigma .25, .5, .75, 1 correspond to descending table indices 750, 500, 250, 0.
        torch.testing.assert_close(
            aitoolkit_timestep_weights(torch.arange(4), 4), canonical[torch.tensor([249, 499, 749, 999])],
            rtol=0, atol=0,
        )
        torch.testing.assert_close(aitoolkit_timestep_weights(torch.tensor([0]), 1), canonical[-1:])

    def test_bell_source_tables(self):
        source = Path(__file__).resolve().parents[2] / "ai-toolkit/toolkit/samplers/custom_flowmatch_sampler.py"
        if not source.is_file():
            self.skipTest("Optional ai-toolkit reference checkout is absent; standalone bell fixtures remain active")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        scheduler = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        init = next(node for node in scheduler.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        weight_block = next(node for node in init.body if isinstance(node, ast.With))
        upstream = SimpleNamespace()
        exec(compile(ast.Module(body=[weight_block], type_ignores=[]), str(source), "exec"),
             {"torch": torch, "self": upstream})
        for half, mode, expected in [
            (False, LossWeight.AI_TOOLKIT_BELL, upstream.linear_timesteps_weights),
            (True, LossWeight.AI_TOOLKIT_HALF_BELL, upstream.linear_timesteps_weights2),
        ]:
            with self.subTest(mode=mode):
                torch.testing.assert_close(aitoolkit_bell_weights(torch.arange(1000), 1000, half),
                                           expected.flip(0), rtol=0, atol=0)

    def test_canonical_table_digest(self):
        # SHA256 of upstream descending IEEE754 float32 bytes, b36bb3998ae596a566d85513299696a3a78f0dcb.
        values = aitoolkit_timestep_weights(torch.arange(1000), 1000).flip(0)
        digest = hashlib.sha256(values.numpy().astype("<f4").tobytes()).hexdigest()
        self.assertEqual(digest, "485f5c69b93051c920b737f4ab87a90b9cc362c4501e422579ef6fedffc070fb")

    def test_bell_tables_and_actual_loss_gradients(self):
        # Independent source fixture: upstream descending indices 0, 500, 999.
        for half, mode, fixture in [
            (False, LossWeight.AI_TOOLKIT_BELL, [0.0, 1.5796048641204834, 0.004869713447988033]),
            (True, LossWeight.AI_TOOLKIT_HALF_BELL, [0.0, 1.5796048641204834, 1.5796048641204834]),
        ]:
            with self.subTest(mode=mode):
                config = TrainConfig.default_values()
                config.loss_weight_fn = mode
                predicted = torch.tensor([[2.0], [3.0], [4.0]], requires_grad=True)
                indices = torch.tensor([999, 499, 0])
                weights = torch.tensor(fixture)
                torch.testing.assert_close(aitoolkit_bell_weights(indices, 1000, half), weights)
                loss = ModelSetupDiffusionLossMixin()._flow_matching_losses(
                    {"loss_weight": torch.ones(3)},
                    {"loss_type": "target", "predicted": predicted, "target": torch.zeros_like(predicted),
                     "timestep": indices}, config, torch.device("cpu"), torch.linspace(0.001, 1, 1000),
                )
                torch.testing.assert_close(loss, torch.tensor([4., 9., 16.]) * weights)
                loss.sum().backward()
                torch.testing.assert_close(predicted.grad.flatten(), torch.tensor([4., 6., 8.]) * weights)

    def test_invalid_indices(self):
        for timesteps, count in [(torch.tensor([0]), 0), (torch.tensor([-1]), 1000),
                                  (torch.tensor([1000]), 1000), (torch.tensor([0.5]), 1000)]:
            with self.subTest(timesteps=timesteps, count=count), self.assertRaises(ValueError):
                aitoolkit_timestep_weights(timesteps, count)

    def test_actual_flow_loss_and_gradients(self):
        config = TrainConfig.default_values()
        config.loss_weight_fn = LossWeight.AI_TOOLKIT_WEIGHTED
        predicted = torch.tensor([[2.0], [3.0]], requires_grad=True)
        timesteps = torch.tensor([999, 0])
        data = {"loss_type": "target", "predicted": predicted,
                "target": torch.zeros_like(predicted), "timestep": timesteps}
        batch = {"loss_weight": torch.tensor([1.0, 2.0])}
        loss = ModelSetupDiffusionLossMixin()._flow_matching_losses(
            batch, data, config, torch.device("cpu"), torch.linspace(0.001, 1, 1000),
        )
        # Independently recorded endpoints from the upstream empirical table.
        weights = torch.tensor([0.905706524848938, 0.4472601115703583])
        torch.testing.assert_close(loss, torch.tensor([4.0, 18.0]) * weights)
        loss.sum().backward()
        torch.testing.assert_close(predicted.grad.flatten(), torch.tensor([4.0, 12.0]) * weights)
        self.assertTrue(LossWeight.AI_TOOLKIT_WEIGHTED.supports_flow_matching())


if __name__ == "__main__":
    unittest.main()
