import tempfile
import unittest
from pathlib import Path

from modules.model.ZImageModel import ZImageModel
from modules.modelLoader.ZImageModelLoader import ZImageModelLoader
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod

import torch
from torch import nn

from safetensors.torch import save_file


class ToolkitValidationTest(unittest.TestCase):
    def test_guidance_rejects_nonfinite_scales(self):
        config = TrainConfig.default_values()
        config.differential_guidance = True
        for scale in (float('nan'), float('inf'), float('-inf'), 0):
            config.differential_guidance_scale = scale
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                config.validate()

    def test_preservation_guards_and_roundtrip(self):
        config = TrainConfig.default_values()
        config.model_type = ModelType.Z_IMAGE
        config.training_method = TrainingMethod.LORA
        config.text_encoder.train = False
        config.diff_output_preservation = True
        with self.assertRaisesRegex(ValueError, 'trigger'):
            config.validate()
        config.diff_output_preservation_trigger = 'sks'
        config.validate()
        restored = TrainConfig.default_values().from_dict(config.to_dict())
        restored.validate()
        self.assertEqual(restored.diff_output_preservation_trigger, 'sks')
        restored.diff_output_preservation_blank = True
        with self.assertRaisesRegex(ValueError, 'together'):
            restored.validate()

    def test_full_rank_export_and_dropout_guards(self):
        config = TrainConfig.default_values()
        config.lora_full_rank = True
        config.output_model_format = ModelFormat.DIFFUSERS_LORA
        with self.assertRaisesRegex(ValueError, 'native'):
            config.validate()
        config.output_model_format = ModelFormat.KOHYA_LORA
        config.lora_rank_dropout = .1
        with self.assertRaisesRegex(ValueError, 'rank dropout'):
            config.validate()

    def test_update_budget_includes_resume_and_accumulation(self):
        config = TrainConfig.default_values()
        config.max_train_steps = 380
        config.gradient_accumulation_steps = 4
        self.assertFalse(config.reached_max_train_steps(379))
        self.assertTrue(config.reached_max_train_steps(380))
        config.gradient_accumulation_steps = 2
        self.assertTrue(config.reached_max_train_steps(380))
        with self.assertRaisesRegex(ValueError, 'Legacy backup'):
            config.reached_max_train_steps(None)
        config.max_train_steps = 0
        self.assertFalse(config.reached_max_train_steps(100000))

    def test_incomplete_and_nonfinite_assistants_do_not_merge(self):
        class Transformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(1, 1, bias=False)

        for state in (
            {'transformer.proj.lora_down.weight': torch.ones(1, 1)},
            {'transformer.proj.lora_up.weight': torch.ones(1, 1)},
            {'transformer.proj.lora_down.weight': torch.ones(1, 1),
             'transformer.proj.lora_up.weight': torch.full((1, 1), float('nan'))},
        ):
            model = ZImageModel(ModelType.Z_IMAGE)
            transformer = Transformer()
            original = transformer.proj.weight.detach().clone()
            with tempfile.TemporaryDirectory() as directory:
                path = str(Path(directory) / 'assistant.safetensors')
                save_file(state, path)
                with self.assertRaises(RuntimeError):
                    ZImageModelLoader()._ZImageModelLoader__load_assistant_lora(model, transformer, path)
            torch.testing.assert_close(transformer.proj.weight, original)
            self.assertFalse(model.assistant_lora_merged)


if __name__ == '__main__':
    unittest.main()
