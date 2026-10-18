import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from modules.module.LoRAModule import LoRAModuleWrapper
from modules.module.quantized.LinearFp8 import LinearFp8
from modules.util.aitoolkit_config_import import import_aitoolkit_config
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.LossWeight import LossWeight
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TimestepDistribution import TimestepDistribution
from modules.util.quantization_util import replace_linear_with_quantized_layers

from torch import nn

import yaml


def recipe():
    return {"job": "extension", "config": {"name": "screen_i", "process": [{
        "type": "diffusion_trainer", "training_folder": "D:/original/output", "device": "cpu",
        "network": {"type": "lora", "linear": 32, "linear_alpha": 8,
                    "network_kwargs": {"ignore_if_contains": ["feed_forward"]}},
        "model": {"name_or_path": "Tongyi-MAI/Z-Image-Turbo", "arch": "zimage:turbo",
                  "assistant_lora_path": "ostris/zimage_turbo_training_adapter/adapter.safetensors",
                  "quantize": True, "qtype": "qfloat8", "quantize_te": True, "qtype_te": "qfloat8"},
        "train": {"batch_size": 1, "steps": 380, "gradient_accumulation_steps": 1,
                  "train_unet": True, "train_text_encoder": False, "gradient_checkpointing": True,
                  "noise_scheduler": "flowmatch", "optimizer": "adamw8bit", "timestep_type": "weighted",
                  "content_or_style": "style", "loss_type": "mse", "lr": 0.0001,
                  "optimizer_params": {"weight_decay": 0.0001}, "do_differential_guidance": True,
                  "differential_guidance_scale": 3, "dtype": "bf16"},
        "datasets": [{"folder_path": "D:/original/dataset", "caption_ext": "txt",
                      "caption_dropout_rate": 0.05, "shuffle_tokens": False,
                      "cache_latents_to_disk": True, "resolution": [512, 768, 1024]}],
        "save": {"dtype": "bf16", "save_every": 95, "max_step_saves_to_keep": 4,
                 "save_format": "diffusers", "push_to_hub": False},
        "sample": {"sampler": "flowmatch", "sample_every": 95, "sample_start_step": 0,
                   "width": 512, "height": 512, "neg": "", "seed": 42, "walk_seed": False,
                   "guidance_scale": 1, "sample_steps": 8, "samples": [{"prompt": "black material dress"}]},
    }]}, "meta": {"name": "screen_i", "version": "1.0"}}


class AiToolkitConfigImportTest(unittest.TestCase):
    def test_effective_screen_i_module_set_and_recipe(self):
        source = recipe()
        preserved = copy.deepcopy(source)
        config = import_aitoolkit_config(source)
        self.assertEqual(source, preserved)
        self.assertEqual((config.lora_rank, config.lora_alpha), (32, 32))
        self.assertEqual(config.max_train_steps, 380)
        self.assertEqual(config.optimizer.optimizer, Optimizer.ADAMW_8BIT)
        self.assertEqual(config.optimizer.eps, 1e-6)
        self.assertEqual(config.optimizer.weight_decay, 1e-4)
        self.assertEqual(config.transformer.weight_dtype, DataType.FLOAT_8)
        self.assertEqual(config.text_encoder.weight_dtype, DataType.FLOAT_8)
        self.assertEqual(config.loss_weight_fn, LossWeight.AI_TOOLKIT_WEIGHTED)
        self.assertEqual(config.timestep_distribution, TimestepDistribution.STYLE_BIASED)
        self.assertEqual(config.concepts[0].text.keep_tags_count, 0)
        self.assertEqual(config.concepts[0].text.tag_dropout_probability, 0.05)
        self.assertEqual(config.concepts[0].image.resolution_override, "512,768,1024")
        self.assertEqual(config.samples[0].cfg_scale, 1)
        self.assertEqual(config.samples[0].diffusion_steps, 8)
        self.assertTrue(config.aitoolkit_guidance_semantics)
        self.assertTrue(config.aitoolkit_timestep_semantics)
        self.assertFalse(config.workspace_dir.startswith("D:"))

        transformer = nn.Module()
        transformer.layers = nn.ModuleList()
        expected = set()
        for layer in range(30):
            block = nn.Module()
            block.attention = nn.ModuleDict({name: nn.Linear(2, 2) for name in ["to_q", "to_k", "to_v", "to_out"]})
            block.adaLN_modulation = nn.Sequential(nn.Linear(2, 2))
            block.feed_forward = nn.Sequential(nn.Linear(2, 2))
            transformer.layers.append(block)
            expected.update(f"layers.{layer}.attention.{name}" for name in ["to_q", "to_k", "to_v", "to_out"])
            expected.add(f"layers.{layer}.adaLN_modulation.0")
        transformer.noise_refiner = nn.Sequential(nn.Linear(2, 2))
        transformer.context_refiner = nn.Sequential(nn.Linear(2, 2))
        wrapper = LoRAModuleWrapper(transformer, "transformer", config, module_filter=[config.layer_filter])
        names = {id(module): name for name, module in transformer.named_modules()}
        actual = {names[id(adapter.orig_module)] for adapter in wrapper.lora_modules.values()}
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 150)

    def test_quantization_footprint_keeps_sensitive_projections(self):
        config = import_aitoolkit_config(recipe())
        transformer = nn.Module()
        for name in ["layers", "noise_refiner", "context_refiner"]:
            block = nn.Module()
            block.projection = nn.Linear(2, 2)
            setattr(transformer, name, nn.ModuleList([block]))
        for name in ["t_embedder", "cap_embedder", "all_x_embedder", "all_final_layer", "siglip_embedder"]:
            setattr(transformer, name, nn.Sequential(nn.Linear(2, 2)))
        replace_linear_with_quantized_layers(transformer, DataType.FLOAT_8, keep_in_fp32_modules=[], quantization=config.quantization)
        for name in ["layers", "noise_refiner", "context_refiner"]:
            self.assertIsInstance(getattr(transformer, name)[0].projection, LinearFp8)
        for name in ["t_embedder", "cap_embedder", "all_x_embedder", "all_final_layer", "siglip_embedder"]:
            self.assertIs(type(getattr(transformer, name)[0]), nn.Linear)
        text_encoder = nn.Module()
        text_encoder.model = nn.Module()
        block = nn.Module()
        block.projection = nn.Linear(2, 2)
        text_encoder.model.layers = nn.ModuleList([block])
        text_encoder.lm_head = nn.Linear(2, 2)
        replace_linear_with_quantized_layers(text_encoder, DataType.FLOAT_8, keep_in_fp32_modules=[], quantization=config.quantization)
        self.assertIsInstance(text_encoder.model.layers[0].projection, LinearFp8)
        self.assertIs(type(text_encoder.lm_head), nn.Linear)
        self.assertTrue(config.workspace_dir.endswith("-aitoolkit-parity"))
        self.assertTrue(config.cache_dir.endswith("-aitoolkit-parity"))
        self.assertTrue(config.output_model_destination.endswith("-aitoolkit-parity.safetensors"))

    def test_serialized_native_roundtrip_keeps_inline_concepts_and_samples(self):
        config = import_aitoolkit_config(recipe())
        exported = config.to_dict()
        exported.pop("secrets", None)
        serialized = json.dumps(exported, allow_nan=False)
        restored = TrainConfig.default_values().from_dict(json.loads(serialized))
        restored.validate()
        roundtripped = restored.to_dict()
        roundtripped.pop("secrets", None)
        self.assertEqual(roundtripped, exported)
        self.assertEqual(len(restored.concepts), 1)
        self.assertEqual(len(restored.samples), 1)

    def test_unknown_execution_and_unsupported_modes_fail_loudly(self):
        mutations = [
            ("train", "unknown_scheduler_control", True), ("train", "timestep_type", "next_sample"),
            ("train", "train_text_encoder", True), ("train", "loss_type", "pixelspace"),
            ("network", "type", "lokr"), ("network", "type", "locon"), ("network", "all_layers", True),
            ("model", "arch", "flux"), ("model", "qtype", "qint4"), ("save", "push_to_hub", True),
            ("sample", "walk_seed", True), ("train", "steps", 0),
            ("model", "quantize", "true"), ("train", "guidance_loss_target", [3, 1]),
            ("train", "optimizer_params", {"betas": [float("nan"), .9]}),
        ]
        for section, option, value in mutations:
            with self.subTest(section=section, option=option, value=value):
                source = recipe()
                source["config"]["process"][0][section][option] = value
                with self.assertRaisesRegex(ValueError, "Unsupported|must |requires"):
                    import_aitoolkit_config(source)

    def test_linear_weighting_flags_and_guidance_range(self):
        for flag, expected in [(None, LossWeight.CONSTANT), ("linear_timesteps", LossWeight.AI_TOOLKIT_BELL),
                               ("linear_timesteps2", LossWeight.AI_TOOLKIT_HALF_BELL)]:
            source = recipe()
            train = source["config"]["process"][0]["train"]
            train["timestep_type"] = "linear"
            train["guidance_loss_target"] = [1.0, 3.0]
            train["pred_scaler"] = 0.8
            if flag:
                train[flag] = True
            config = import_aitoolkit_config(source)
            self.assertEqual(config.loss_weight_fn, expected)
            self.assertEqual((config.guidance_loss_target_min, config.guidance_loss_target_max), (1.0, 3.0))
            self.assertEqual(config.pred_scaler, 0.8)

    def test_class_preservation_trigger_import_and_roundtrip(self):
        source = recipe()
        process = source['config']['process'][0]
        process['trigger_word'] = 'sks'
        process['train']['diff_output_preservation'] = True
        process['train']['diff_output_preservation_class'] = 'person'
        process['sample']['samples'] = [{'prompt': '[trigger] portrait of [name]'}, {'prompt': 'a person'}]
        config = import_aitoolkit_config(source)
        restored = TrainConfig.default_values().from_dict(config.to_dict())
        restored.validate()
        self.assertEqual(restored.aitoolkit_trigger_word, 'sks')
        self.assertEqual(restored.diff_output_preservation_trigger, 'sks')
        self.assertEqual(restored.samples[0].prompt, 'sks portrait of sks')
        self.assertEqual(restored.samples[1].prompt, 'a person')
        for invalid in (True, False, 0, 1, [], '  '):
            with self.subTest(invalid=invalid):
                process['trigger_word'] = invalid
                with self.assertRaisesRegex(ValueError, 'trigger_word'):
                    import_aitoolkit_config(source)

    def test_cli_does_not_overwrite_or_read_secrets(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts/import_aitoolkit_config.py"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.yaml"
            destination = Path(directory) / "output.json"
            source.write_text(yaml.safe_dump(recipe()), encoding="utf-8")
            original = source.read_bytes()
            result = subprocess.run([sys.executable, str(script), str(source), str(destination)],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("secrets", json.loads(destination.read_text(encoding="utf-8")))
            exported = destination.read_bytes()
            second = subprocess.run([sys.executable, str(script), str(source), str(destination)],
                                    cwd=root, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(destination.read_bytes(), exported)
            self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
