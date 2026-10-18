import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from modules.modelSaver.mixin.LoRASaverMixin import LoRASaverMixin
from modules.module.LoRAModule import FullDeltaModule, FullRankModule, LoConModule, LoRAModule, LoRAModuleWrapper
from modules.util.config.TrainConfig import TrainConfig
from modules.util.load_lora_util import normalize_various, reverse_comfy

import torch
from torch import nn

from safetensors.torch import load_file, save_file


class LoRANetworkOptionsTest(unittest.TestCase):
    def make_adapter(self, layer=None):
        adapter = LoRAModule("test", layer if layer is not None else nn.Linear(2, 2, bias=False), 2, 2)
        adapter.hook_to_module()
        with torch.no_grad():
            adapter.lora_down.weight.fill_(1)
            adapter.lora_up.weight.fill_(1)
        return adapter

    def test_rank_dropout_matches_toolkit_reference(self):
        for x in (torch.ones(2, 2), torch.ones(2, 3, 2), torch.ones(2, 2, 3, 3)):
            layer = nn.Conv2d(2, 2, 1, bias=False) if x.ndim == 4 else None
            adapter = self.make_adapter(layer)
            adapter.rank_dropout = 0.5
            mask = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
            with patch("torch.rand", return_value=mask):
                actual = adapter.delta_forward(x)
            hidden = adapter.lora_down(x)
            reference_mask = mask > 0.5
            if x.ndim == 3:
                reference_mask = reference_mask.unsqueeze(1)
            elif x.ndim == 4:
                reference_mask = reference_mask.unsqueeze(-1).unsqueeze(-1)
            expected = adapter.lora_up(hidden * reference_mask) * 2
            torch.testing.assert_close(actual, expected)
            actual.sum().backward()
            self.assertIsNotNone(adapter.lora_up.weight.grad)

    def test_module_dropout_and_eval(self):
        adapter = self.make_adapter()
        adapter.module_dropout = 1.0
        x = torch.ones(2, 2)
        torch.testing.assert_close(adapter(x), adapter.orig_forward(x))
        adapter.eval()
        torch.testing.assert_close(adapter.delta_forward(x), torch.full_like(x, 4))
        adapter.rank_dropout = 0.9
        torch.testing.assert_close(adapter.delta_forward(x), torch.full_like(x, 4))

    def test_dropped_only_adapter_keeps_backward_with_frozen_base(self):
        for adapter_type in (LoRAModule, LoConModule, FullRankModule):
            with self.subTest(adapter_type=adapter_type.__name__):
                layer = nn.Linear(2, 2, bias=False).requires_grad_(False)
                adapter = adapter_type('test', layer, 2, 2)
                adapter.hook_to_module()
                adapter.module_dropout = 0.5
                with patch('torch.rand', return_value=torch.tensor([0.1])):
                    result = layer(torch.ones(1, 2))
                torch.testing.assert_close(result, adapter.orig_forward(torch.ones(1, 2)))
                result.square().mean().backward()
                for projection in (adapter.lora_down, adapter.lora_up):
                    for parameter in projection.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertEqual(parameter.grad.count_nonzero().item(), 0)

    def test_wrapper_filters_and_convolution_parameters(self):
        model = nn.ModuleDict({"linear": nn.Linear(4, 4), "tiny": nn.Linear(1, 1),
                               "conv": nn.Conv2d(4, 4, 3), "point": nn.Conv2d(4, 4, 1)})
        config = TrainConfig.default_values()
        config.lora_rank = 4
        config.lora_alpha = 4
        config.lora_conv_rank = 2
        config.lora_conv_alpha = 3.0
        config.lora_exclude = " tiny, "
        config.lora_rank_dropout = 0.2
        wrapper = LoRAModuleWrapper(model, "test", config)
        self.assertNotIn("tiny", wrapper.lora_modules)
        self.assertEqual(wrapper.lora_modules["conv"].rank, 2)
        self.assertEqual(wrapper.lora_modules["conv"].alpha.item(), 3)
        self.assertEqual(wrapper.lora_modules["point"].rank, 4)
        self.assertEqual(wrapper.lora_modules["linear"].rank_dropout, 0.2)
        config.lora_conv_rank = 0
        config.lora_include = " conv, linear, "
        config.lora_parameter_threshold = 21
        self.assertEqual(LoRAModuleWrapper(model, "test", config).lora_modules, {})

    def test_fullrank_dense_delta_and_roundtrip(self):
        layer = nn.Linear(2, 3, bias=False)
        adapter = FullRankModule("test", layer, 2, 2)
        adapter.hook_to_module()
        x = torch.ones(1, 2)
        self.assertEqual(adapter.lora_down.weight.shape, layer.weight.shape)
        self.assertNotEqual(adapter.lora_down.weight.abs().sum().item(), 0)
        expected = adapter(x).detach()
        adapter(x).sum().backward()
        self.assertIsNotNone(adapter.lora_down.weight.grad)
        fresh = FullRankModule("test", nn.Linear(2, 3, bias=False), 2, 2)
        fresh.orig_module.weight.data.copy_(layer.weight)
        fresh.hook_to_module()
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / "fullrank.safetensors")
            save_file(adapter.state_dict(prefix="test."), filename)
            fresh.load_state_dict(load_file(filename))
        torch.testing.assert_close(fresh(x), expected)

    def test_locon_zero_gate_gradients_cp_and_roundtrip(self):
        for layer, x, use_cp in ((nn.Linear(2, 3), torch.ones(1, 2), False),
                                 (nn.Conv2d(2, 3, 3), torch.ones(1, 2, 5, 5), True)):
            adapter = LoConModule("test", layer, 2, 2, use_cp=use_cp)
            adapter.hook_to_module()
            torch.testing.assert_close(adapter(x), adapter.orig_forward(x))
            adapter(x).sum().backward()
            self.assertIsNotNone(adapter.scalar.grad)
            self.assertEqual(adapter.lora_down.weight.grad.abs().sum().item(), 0)
            with torch.no_grad():
                adapter.scalar.fill_(0.75)
            state = adapter.state_dict(prefix="test.")
            self.assertEqual(state["test.scalar"].item(), 0.75)
            fresh_layer = nn.Conv2d(2, 3, 3) if use_cp else nn.Linear(2, 3)
            fresh_layer.load_state_dict(layer.state_dict())
            fresh = LoConModule("test", fresh_layer, 2, 2, use_cp=use_cp)
            fresh.hook_to_module()
            with tempfile.TemporaryDirectory() as directory:
                filename = str(Path(directory) / "locon.safetensors")
                save_file(state, filename)
                fresh.load_state_dict(load_file(filename))
            torch.testing.assert_close(fresh(x), adapter(x))
            before = adapter(x).detach()
            adapter.apply_to_module()
            torch.testing.assert_close(adapter.orig_forward(x), before)

    def test_locon_internal_optimizer_resume_and_external_export(self):
        adapter = LoConModule("test", nn.Linear(2, 3), 2, 1.0)
        adapter.hook_to_module()
        optimizer = torch.optim.Adam(adapter.parameters(), lr=0.01)
        x = torch.ones(2, 2)

        def update(current, optim):
            optim.zero_grad()
            current(x).square().mean().backward()
            optim.step()

        update(adapter, optimizer)
        update(adapter, optimizer)
        checkpoint = copy.deepcopy(adapter.state_dict(prefix="test."))
        optimizer_checkpoint = copy.deepcopy(optimizer.state_dict())
        fresh = LoConModule("test", nn.Linear(2, 3), 2, 1.0)
        fresh.orig_module.load_state_dict(adapter.orig_module.state_dict())
        fresh.hook_to_module()
        fresh.load_state_dict(checkpoint)
        resumed_optimizer = torch.optim.Adam(fresh.parameters(), lr=0.01)
        resumed_optimizer.load_state_dict(optimizer_checkpoint)
        update(adapter, optimizer)
        update(fresh, resumed_optimizer)
        for original, resumed in zip(adapter.parameters(), fresh.parameters(), strict=True):
            torch.testing.assert_close(original, resumed)

        class Saver(LoRASaverMixin):
            def _get_state_dict(self, _model):
                return fresh.state_dict(prefix="test.")

            def _create_safetensors_header(self, _model, _state):
                return {}

        saver = Saver()
        with tempfile.TemporaryDirectory() as directory:
            saver._save_internal(None, directory)
            internal = load_file(str(Path(directory) / "lora" / "lora.safetensors"))
            self.assertIn("test.scalar", internal)
            self.assertIn("test.scalar", normalize_various(internal))
            exported_path = str(Path(directory) / "external.safetensors")
            saver._save_diffusers(SimpleNamespace(adapters=list), exported_path, None)
            external = load_file(exported_path)
            self.assertNotIn("test.scalar", external)
            restored = LoConModule("test", nn.Linear(2, 3), 2, 1.0)
            restored.orig_module.load_state_dict(fresh.orig_module.state_dict())
            restored.hook_to_module()
            restored.load_state_dict(normalize_various(external))
            torch.testing.assert_close(restored(x), fresh(x))

    def test_full_delta_weight_bias_gradients_and_parameter_restore(self):
        for layer, x in ((nn.Linear(2, 3), torch.ones(2, 2)),
                         (nn.Conv2d(2, 3, 3), torch.ones(2, 2, 5, 5)),
                         (nn.LayerNorm(2), torch.tensor([[1.0, 2.0]])),
                         (nn.Embedding(3, 2), torch.tensor([0, 1]))):
            adapter = FullDeltaModule("test", layer)
            adapter.hook_to_module()
            weight, bias = layer.weight, getattr(layer, "bias", None)
            expected = adapter.orig_forward(x).detach()
            torch.testing.assert_close(adapter(x), expected)
            adapter(x).sum().backward()
            self.assertIsNotNone(adapter.diff.grad)
            if bias is not None:
                self.assertIsNotNone(adapter.diff_b.grad)
            self.assertIs(layer.weight, weight)
            self.assertIs(getattr(layer, "bias", None), bias)
            with torch.no_grad():
                adapter.diff.fill_(0.1)
                if adapter.diff_b is not None:
                    adapter.diff_b.fill_(0.2)
            predicted = adapter(x).detach()
            adapter.apply_to_module()
            torch.testing.assert_close(adapter.orig_forward(x), predicted)

    def test_full_delta_restores_base_after_forward_exception(self):
        class FailingLayer(nn.Linear):
            def forward(self, _x):
                raise RuntimeError("expected failure")

        layer = FailingLayer(2, 3)
        weight, bias = layer.weight, layer.bias
        adapter = FullDeltaModule("test", layer)
        adapter.hook_to_module()
        with self.assertRaisesRegex(RuntimeError, "expected failure"):
            adapter(torch.ones(1, 2))
        self.assertIs(layer.weight, weight)
        self.assertIs(layer.bias, bias)

    def test_mixed_full_delta_selection_and_export_roundtrip(self):
        config = TrainConfig.default_values()
        config.lora_all_layers = True
        config.lora_full_include = "linear"
        def make_model():
            return nn.ModuleDict({"linear": nn.Linear(2, 2), "normal": nn.Linear(2, 2),
                                  "norm": nn.LayerNorm(2), "embedding": nn.Embedding(3, 2)})
        model = make_model()
        wrapper = LoRAModuleWrapper(model, "transformer", config)
        self.assertIsInstance(wrapper.lora_modules["linear"], FullDeltaModule)
        self.assertIsInstance(wrapper.lora_modules["normal"], LoRAModule)
        self.assertIsInstance(wrapper.lora_modules["norm"], FullDeltaModule)
        self.assertIsInstance(wrapper.lora_modules["embedding"], FullDeltaModule)
        wrapper.hook_to_module()
        with torch.no_grad():
            wrapper.lora_modules["linear"].diff.fill_(0.25)
            wrapper.lora_modules["norm"].diff_b.fill_(0.5)
        class Saver(LoRASaverMixin):
            def _get_state_dict(self, _model):
                return wrapper.state_dict()
            def _create_safetensors_header(self, _model, _state):
                return {}
        with tempfile.TemporaryDirectory() as directory:
            Saver()._save_internal(None, directory)
            state = normalize_various(load_file(str(Path(directory) / "lora" / "lora.safetensors")))
            self.assertIn("transformer.linear.diff", state)
            self.assertIn("transformer.linear.diff_b", state)
            fresh_model = make_model()
            fresh_model.load_state_dict(model.state_dict())
            fresh = LoRAModuleWrapper(fresh_model, "transformer", config)
            fresh.load_state_dict(state)
            fresh.hook_to_module()
            x = torch.ones(1, 2)
            torch.testing.assert_close(fresh_model["linear"](x), model["linear"](x))
            torch.testing.assert_close(fresh_model["norm"](x), model["norm"](x))
            export_model = SimpleNamespace(
                model_type=SimpleNamespace(denoising_model_part=lambda: "transformer"),
                lora_diffusers_to_comfy=lambda: None, lora_text_encoders=list, adapters=lambda: [wrapper])
            comfy_file = str(Path(directory) / "comfy.safetensors")
            Saver()._save_comfy(export_model, comfy_file, None)
            exported = load_file(comfy_file)
            self.assertIn("diffusion_model.linear.diff", exported)
            canonical = reverse_comfy(normalize_various(exported), "transformer", None, [], {})
            fresh.load_state_dict(canonical)
            torch.testing.assert_close(fresh_model["linear"](x), model["linear"](x))

    def test_full_delta_rejects_quantized_weight_dtype(self):
        layer = nn.Linear(2, 3)
        layer.weight = nn.Parameter(layer.weight.to(torch.float8_e4m3fn), requires_grad=False)
        with self.assertRaisesRegex(ValueError, "quantized"):
            FullDeltaModule("test", layer)

    def test_fullrank_comfy_dense_delta_export_roundtrip(self):
        config = TrainConfig.default_values()
        config.lora_full_rank = True
        config.lora_rank = 2
        config.lora_alpha = 0.75
        model = nn.ModuleDict({"linear": nn.Linear(2, 3)})
        wrapper = LoRAModuleWrapper(model, "transformer", config)
        wrapper.hook_to_module()
        class Saver(LoRASaverMixin):
            def _get_state_dict(self, _model):
                return wrapper.state_dict()
            def _create_safetensors_header(self, _model, _state):
                return {}
        export_model = SimpleNamespace(
            model_type=SimpleNamespace(denoising_model_part=lambda: "transformer"),
            lora_diffusers_to_comfy=lambda: None, lora_text_encoders=list,
            adapters=lambda: [wrapper])
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / "fullrank-comfy.safetensors")
            Saver()._save_comfy(export_model, filename, None)
            exported = load_file(filename)
            self.assertEqual(set(exported), {"diffusion_model.linear.diff"})
            adapter = wrapper.lora_modules["linear"]
            torch.testing.assert_close(exported["diffusion_model.linear.diff"],
                                       adapter.lora_down.weight * 0.75 / 2)
            canonical = reverse_comfy(normalize_various(exported), "transformer", None, [], {})
            fresh_model = nn.ModuleDict({"linear": nn.Linear(2, 3)})
            fresh_model.load_state_dict(model.state_dict())
            fresh = LoRAModuleWrapper(fresh_model, "transformer", config)
            fresh.load_state_dict(canonical)
            fresh.hook_to_module()
            torch.testing.assert_close(fresh_model["linear"](torch.ones(1, 2)),
                                       model["linear"](torch.ones(1, 2)))
            self.assertEqual(set(fresh.state_dict()), set(wrapper.state_dict()))
            self.assertIn("transformer.linear.lora_down.weight", fresh.state_dict())
            self.assertNotIn("transformer.linear.diff", fresh.state_dict())

    def test_default_config_retains_convolution_rank(self):
        config = TrainConfig.default_values()
        model = nn.Conv2d(2, 2, 3)
        adapter = next(iter(LoRAModuleWrapper(model, "test", config).lora_modules.values()))
        self.assertEqual(adapter.rank, config.lora_rank)


if __name__ == "__main__":
    unittest.main()
