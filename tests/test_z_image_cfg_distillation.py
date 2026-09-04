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
        model.transformer_assistant_lora = wrapper
        model.assistant_lora_merged = True
        with model.inverse_assistant_lora_for_sampling():
            restored = linear(x)

        self.assertFalse(torch.equal(merged, baseline))
        self.assertTrue(torch.allclose(restored, baseline))
        self.assertFalse(wrapper.is_applied())
        self.assertEqual(wrapper.multiplier(), 1.0)

    def test_assistant_lora_parameters_are_absent_from_optimizer_parameters(self):
        model = ZImageModel(ModelType.Z_IMAGE)
        model.transformer_lora = _linear_lora(nn.Linear(1, 1, bias=False))
        model.transformer_assistant_lora = _linear_lora(nn.Linear(1, 1, bias=False))
        model.assistant_lora_merged = True

        optimizer_param_ids = {id(parameter) for adapter in model.adapters() for parameter in adapter.parameters()}
        assistant_param_ids = {id(parameter) for parameter in model.transformer_assistant_lora.parameters()}

        self.assertFalse(optimizer_param_ids & assistant_param_ids)

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
