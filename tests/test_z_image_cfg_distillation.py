import tempfile
import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn


def _install_external_stubs():
    try:
        import diffusers  # noqa: F401
    except ImportError:
        class _Stub:
            _keep_in_fp32_modules = []

        diffusers = types.ModuleType("diffusers")
        for name in [
            "AutoencoderKL",
            "DiffusionPipeline",
            "FlowMatchEulerDiscreteScheduler",
            "GGUFQuantizationConfig",
            "ZImagePipeline",
            "ZImageTransformer2DModel",
        ]:
            setattr(diffusers, name, _Stub)
        sys.modules["diffusers"] = diffusers

        gguf_utils = types.ModuleType("diffusers.quantizers.gguf.utils")
        gguf_utils.GGUFLinear = _Stub
        gguf_utils.dequantize_gguf_tensor = lambda tensor: tensor
        sys.modules["diffusers.quantizers"] = types.ModuleType("diffusers.quantizers")
        sys.modules["diffusers.quantizers.gguf"] = types.ModuleType("diffusers.quantizers.gguf")
        sys.modules["diffusers.quantizers.gguf.utils"] = gguf_utils

    try:
        import transformers  # noqa: F401
    except ImportError:
        class _Stub:
            _keep_in_fp32_modules = []

        transformers = types.ModuleType("transformers")
        transformers.Qwen2Tokenizer = _Stub
        transformers.Qwen3ForCausalLM = _Stub
        sys.modules["transformers"] = transformers

        conversion_mapping = types.ModuleType("transformers.conversion_mapping")
        conversion_mapping.get_checkpoint_conversion_mapping = lambda _model_type: None
        sys.modules["transformers.conversion_mapping"] = conversion_mapping

        core_model_loading = types.ModuleType("transformers.core_model_loading")
        core_model_loading.rename_source_key = lambda key, *_args, **_kwargs: (key, None)
        sys.modules["transformers.core_model_loading"] = core_model_loading

    try:
        import accelerate  # noqa: F401
    except ImportError:
        accelerate = types.ModuleType("accelerate")
        accelerate.init_empty_weights = lambda: nullcontext()
        sys.modules["accelerate"] = accelerate


_install_external_stubs()

from modules.model.ZImageModel import ZImageModel
from modules.modelLoader.ZImageModelLoader import ZImageModelLoader
from modules.modelSetup.z_image_distillation_util import apply_z_image_differential_guidance
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.ModelType import ModelType


def _linear_lora(linear: nn.Linear) -> LoRAModuleWrapper:
    config = TrainConfig.default_values()
    config.lora_rank = 1
    config.lora_alpha = 1.0
    wrapper = LoRAModuleWrapper(linear, "transformer", config)
    module = wrapper.lora_modules[""]
    module.lora_down.weight.data.fill_(2.0)
    module.lora_up.weight.data.fill_(3.0)
    return wrapper


class ZImageCfgDistillationTest(unittest.TestCase):
    def test_differential_guidance_scales_target(self):
        predicted = torch.tensor([[1.0, 2.0]])
        target = torch.tensor([[3.0, 5.0]])

        guided = apply_z_image_differential_guidance(predicted, target, 3.0)

        self.assertTrue(torch.equal(guided, predicted + 3.0 * (target - predicted)))

    def test_differential_guidance_leaves_prior_prediction_targets_unchanged(self):
        predicted = torch.tensor([[1.0, 2.0], [10.0, 20.0]])
        target = torch.tensor([[3.0, 5.0], [30.0, 50.0]])

        guided = apply_z_image_differential_guidance(
            predicted,
            target,
            3.0,
            [ConceptType.STANDARD.value, ConceptType.PRIOR_PREDICTION.value],
        )

        self.assertTrue(torch.equal(guided[0], predicted[0] + 3.0 * (target[0] - predicted[0])))
        self.assertTrue(torch.equal(guided[1], target[1]))

    def test_differential_guidance_disabled_is_old_target_path(self):
        target = torch.tensor([[3.0, 5.0]])
        config = TrainConfig.default_values()

        output_target = target
        if config.differential_guidance:
            output_target = apply_z_image_differential_guidance(
                torch.zeros_like(target),
                target,
                config.differential_guidance_scale,
            )

        self.assertIs(output_target, target)

    def test_assistant_lora_merge_and_inverse_hook_matches_baseline(self):
        linear = nn.Linear(1, 1, bias=False)
        linear.weight.data.fill_(5.0)
        x = torch.tensor([[2.0]])
        baseline = linear(x)
        wrapper = _linear_lora(linear)
        model = ZImageModel(ModelType.Z_IMAGE)

        wrapper.apply_to_module(1.0)
        merged = linear(x)
        model.transformer = linear
        model.transformer_assistant_lora = wrapper
        model.assistant_lora_merged = True
        with model.inverse_assistant_lora_for_sampling():
            restored = linear(x)

        self.assertFalse(torch.equal(merged, baseline))
        self.assertTrue(torch.allclose(restored, baseline))
        self.assertFalse(wrapper.is_applied())
        self.assertEqual(wrapper.multiplier(), 1.0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_delta_forward_syncs_lora_device_to_input_on_every_call(self):
        # Regression: lora_down/lora_up aren't registered submodules of orig_module, so
        # per-layer offloading (moving orig_module to/from its compute device just-in-time
        # around a call, e.g. for the assistant LoRA's unfiltered wrapper) never reaches
        # them. A one-time sync when the LoRA hook is (re-)attached isn't enough -- observed
        # in practice as: sampling prompt 1 succeeds, then offloading moves orig_module
        # again before prompt 2's forward pass, which then crashes on a CPU/CUDA mismatch.
        # So delta_forward must check on every call against where the input actually is.
        linear = nn.Linear(1, 1, bias=False)
        linear.weight.data.fill_(5.0)
        wrapper = _linear_lora(linear)
        wrapper.hook_to_module()
        module = wrapper.lora_modules[""]

        x_cuda = torch.tensor([[2.0]], device="cuda")
        module.delta_forward(x_cuda)
        self.assertEqual(module.lora_down.weight.device.type, "cuda")

        # Simulates the offloader moving orig_module (and thus this call's input) back to
        # CPU for a later call -- must re-sync rather than staying pinned to "cuda".
        x_cpu = torch.tensor([[2.0]])
        module.delta_forward(x_cpu)
        self.assertEqual(module.lora_down.weight.device.type, "cpu")

    def test_delta_forward_does_not_change_lora_dtype_for_quantized_orig_module(self):
        # Regression: an earlier version of the device-sync fix above also synced dtype,
        # which breaks LoRA math against a quantized orig_module -- e.g. torch has no CUDA
        # kernel for dividing Float8_e4m3fn tensors, which this alpha/rank scaling does.
        # delta_forward reads device from its own input, not from orig_module.weight, so it
        # never observes the quantized dtype at all; assert the dtype-preservation directly.
        linear = nn.Linear(1, 1, bias=False)
        linear.weight.data.fill_(5.0)
        wrapper = _linear_lora(linear)
        wrapper.hook_to_module()
        module = wrapper.lora_modules[""]

        x = torch.tensor([[2.0]])
        module.delta_forward(x)

        self.assertEqual(module.lora_down.weight.dtype, torch.float32)

    def test_assistant_lora_parameters_are_absent_from_optimizer_parameters(self):
        model = ZImageModel(ModelType.Z_IMAGE)
        model.transformer_lora = _linear_lora(nn.Linear(1, 1, bias=False))
        model.transformer_assistant_lora = _linear_lora(nn.Linear(1, 1, bias=False))
        model.assistant_lora_merged = True

        optimizer_param_ids = {id(parameter) for adapter in model.adapters() for parameter in adapter.parameters()}
        assistant_param_ids = {id(parameter) for parameter in model.transformer_assistant_lora.parameters()}

        self.assertFalse(optimizer_param_ids & assistant_param_ids)

    def test_assistant_lora_partial_module_coverage_does_not_raise(self):
        # Real assistant-LoRA files (e.g. ostris/zimage_turbo_training_adapter) only cover a
        # subset of the transformer's LoRA-eligible modules (they skip the embedders), while
        # the wrapper built for merging is unfiltered ("full"). Regression for the crash this
        # caused: LoRAModuleWrapper.load_state_dict used strict=True, so modules with no
        # matching keys at all (not just individual missing tensors) raised instead of
        # keeping their zero-initialized (no-op) lora_up.
        class Transformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj_a = nn.Linear(1, 1, bias=False)
                self.proj_b = nn.Linear(1, 1, bias=False)

        transformer = Transformer()
        config = TrainConfig.default_values()
        config.lora_rank = 1
        config.lora_alpha = 1.0
        wrapper = LoRAModuleWrapper(transformer, "transformer", config)

        partial_state_dict = {
            "transformer.proj_a.alpha": torch.tensor(1.0),
            "transformer.proj_a.lora_down.weight": torch.full((1, 1), 2.0),
            "transformer.proj_a.lora_up.weight": torch.full((1, 1), 3.0),
        }

        wrapper.load_state_dict(partial_state_dict, strict=False)

        self.assertTrue(torch.equal(wrapper.lora_modules["proj_a"].lora_down.weight, torch.full((1, 1), 2.0)))
        self.assertTrue(torch.equal(wrapper.lora_modules["proj_b"].lora_up.weight, torch.zeros((1, 1))))

    def test_unrecognized_assistant_lora_key_format_is_clear(self):
        model = ZImageModel(ModelType.Z_IMAGE)
        transformer = nn.Linear(1, 1, bias=False)
        state_dict = {
            "unrecognized.layer.lora_down.weight": torch.ones(1, 1),
            "unrecognized.layer.lora_up.weight": torch.ones(1, 1),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.safetensors"
            save_file(state_dict, str(path))
            with self.assertRaisesRegex(RuntimeError, "Assistant LoRA contains keys that do not match"):
                ZImageModelLoader()._ZImageModelLoader__load_assistant_lora(model, transformer, str(path))


if __name__ == "__main__":
    unittest.main()
