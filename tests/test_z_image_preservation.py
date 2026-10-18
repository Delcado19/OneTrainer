import unittest
from contextlib import nullcontext
from types import SimpleNamespace

from modules.modelSetup.BaseZImageSetup import BaseZImageSetup
from modules.modelSetup.z_image_distillation_util import apply_z_image_guidance_loss, z_image_preservation_prompts
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.TrainProgress import TrainProgress

import torch
from torch import nn


class _Adapter:
    active = True

    def is_applied(self):
        return self.active

    def remove_hook_from_module(self):
        self.active = False

    def hook_to_module(self):
        self.active = True


class _Transformer(nn.Module):
    def __init__(self, adapter):
        super().__init__()
        self.base = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.lora = nn.Parameter(torch.tensor(0.25))
        self.adapter = adapter
        self.calls = []
        self.fail = False

    def forward(self, latents, timestep, embeds, return_dict):
        self.calls.append((torch.stack(latents).detach().clone(), timestep.detach().clone(), embeds, self.adapter.active))
        if self.fail:
            raise RuntimeError("teacher failure")
        weight = self.base + (self.lora if self.adapter.active else 0)
        return SimpleNamespace(sample=[latent * weight + embed for latent, embed in zip(latents, embeds, strict=True)])


class _Model:
    def __init__(self):
        self.adapter = _Adapter()
        self.transformer = _Transformer(self.adapter)
        self.autocast_context = nullcontext()
        self.train_dtype = DataType.FLOAT_32
        self.noise_scheduler = SimpleNamespace(config={'num_train_timesteps': 1000}, timesteps=torch.arange(1000), sigmas=torch.ones(1000))
        self.encoded = []
        self.moves = []

    def adapters(self):
        return [self.adapter]

    def scale_latents(self, latents):
        return latents

    def calculate_timestep_shift(self, *args):
        return 1

    def encode_text(self, train_device, text=None, **kwargs):
        if text is None:
            return [torch.tensor(9.0)]
        self.encoded.append(list(text))
        return [torch.tensor(float(len(prompt))) for prompt in text]

    def materialize(self, part):
        self.moves.append(('materialize', part))

    def evict(self, part):
        self.moves.append(('evict', part))


class _Setup(BaseZImageSetup):
    def create_parameters(self, *args):
        pass

    def setup_model(self, *args):
        pass

    def setup_train_device(self, *args):
        pass

    def after_optimizer_step(self, *args):
        pass

    def _create_noise(self, latent, *args):
        return torch.ones_like(latent) * 2

    def _get_timestep_discrete(self, *args, **kwargs):
        return torch.tensor([500])

    def _add_noise_discrete(self, latent, noise, *args):
        return (latent + noise) / 2, torch.tensor([0.5])

    def _flow_matching_losses(self, batch, data, **kwargs):
        return (data['predicted'] - data['target']).square().mean().reshape(1)


class ZImagePreservationTest(unittest.TestCase):
    def setUp(self):
        self.config = TrainConfig.default_values()
        self.config.train_device = 'cpu'
        self.config.training_method = TrainingMethod.LORA
        self.config.diff_output_preservation_trigger = 'sks'
        self.config.diff_output_preservation_class = 'person'
        self.model = _Model()
        self.setup = _Setup(torch.device('cpu'), torch.device('cpu'), False)
        self.batch = {'latent_image': torch.ones(1, 1, 2, 2), 'prompt': ['sks portrait'],
                      'text_encoder_hidden_state': torch.ones(1), 'tokens': torch.ones(1), 'tokens_mask': torch.ones(1)}

    def test_dop_same_inputs_separate_condition_and_detached_teacher(self):
        self.config.diff_output_preservation = True
        data = self.setup.predict(self.model, self.batch, self.config, TrainProgress())
        calls = self.model.transformer.calls
        self.assertEqual(len(calls), 3)
        self.assertEqual([call[3] for call in calls], [False, True, True])
        self.assertEqual(self.model.encoded, [['person portrait']])
        for call in calls[1:]:
            torch.testing.assert_close(call[0], calls[0][0])
            torch.testing.assert_close(call[1], calls[0][1])
        loss = self.setup.calculate_loss(self.model, self.batch, data, self.config)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNone(self.model.transformer.base.grad)
        self.assertIsNotNone(self.model.transformer.lora.grad)
        self.assertTrue(self.model.adapter.active)

    def test_blank_preservation_uses_blank_caption_and_own_multiplier(self):
        self.config.diff_output_preservation_blank = True
        self.config.blank_prompt_preservation_multiplier = 4
        data = self.setup.predict(self.model, self.batch, self.config, TrainProgress())
        self.assertEqual(self.model.encoded, [['']])
        torch.testing.assert_close(data['preservation_loss'], torch.tensor(0.375 ** 2 * 4))

    def test_teacher_failure_restores_lora_hooks(self):
        self.config.diff_output_preservation = True
        self.model.transformer.fail = True
        with self.assertRaisesRegex(RuntimeError, 'teacher failure'):
            self.setup.predict(self.model, self.batch, self.config, TrainProgress())
        self.assertTrue(self.model.adapter.active)
        self.assertTrue(self.model.transformer.training)
        self.assertEqual(self.model.moves[-1], ('evict', 'text_encoder'))

    def test_dop_nested_in_prior_prediction_keeps_outer_hooks_disabled(self):
        self.config.diff_output_preservation = True
        with self.setup.prior_model(self.model, self.config), torch.no_grad():
            data = self.setup.predict(self.model, self.batch, self.config, TrainProgress())
            self.assertFalse(self.model.adapter.active)
            self.assertTrue(all(not call[3] for call in self.model.transformer.calls))
            self.assertFalse(data['predicted'].requires_grad)
        self.assertTrue(self.model.adapter.active)

    def test_stabilization_zeros_only_blank_caption_latent(self):
        self.config.do_blank_stabilization = True
        self.batch['prompt'] = ['  ']
        data = self.setup.predict(self.model, self.batch, self.config, TrainProgress())
        torch.testing.assert_close(data['target'], torch.full((1, 1, 2, 2), 2.0))
        torch.testing.assert_close(self.batch['latent_image'], torch.ones(1, 1, 2, 2))

    def test_source_guidance_semantics_and_detached_target(self):
        self.config.differential_guidance = True
        self.config.aitoolkit_guidance_semantics = True
        data = self.setup.predict(self.model, self.batch, self.config, TrainProgress())
        torch.testing.assert_close(data['target'], torch.ones(1, 1, 2, 2))
        self.config.do_guidance_loss = True
        data = self.setup.predict(self.model, self.batch, self.config, TrainProgress())
        self.assertFalse(data['target'].requires_grad)
        self.assertEqual(self.model.encoded, [['']])
        torch.testing.assert_close(self.model.transformer.calls[-1][0], self.model.transformer.calls[-2][0])

    def test_guidance_projection_sigma_and_errors(self):
        target = torch.tensor([[2.0, 4.0]], requires_grad=True)
        unconditional = torch.tensor([[1.0, 2.0]], requires_grad=True)
        result = apply_z_image_guidance_loss(target, unconditional, torch.tensor([500]), 3, schedule='sigma')
        torch.testing.assert_close(result, torch.tensor([[3.0, 6.0]]))
        self.assertFalse(result.requires_grad)
        projected = apply_z_image_guidance_loss(target, unconditional, torch.tensor([500]), 3, cfg_zero=True)
        torch.testing.assert_close(projected, target.detach())
        with self.assertRaises(ValueError):
            apply_z_image_guidance_loss(target, unconditional, torch.tensor([500]), 3, schedule='invalid')
        with self.assertRaises(ValueError):
            z_image_preservation_prompts(None, 'sks', 'person')
        with self.assertRaises(ValueError):
            z_image_preservation_prompts(['sks'], '', 'person')

    def test_guidance_range_broadcast_and_repeatability(self):
        target = torch.ones(2, 1, 2, 2)
        result = apply_z_image_guidance_loss(target, torch.zeros_like(target), torch.tensor([500, 500]), [2, 4])
        torch.testing.assert_close(result[:, 0, 0, 0], torch.tensor([2.0, 4.0]))
        self.config.do_guidance_loss = True
        self.config.guidance_loss_target_min = 2.0
        self.config.guidance_loss_target_max = 4.0
        first = self.setup.predict(self.model, self.batch, self.config, TrainProgress(global_step=5))
        second = self.setup.predict(self.model, self.batch, self.config, TrainProgress(global_step=5))
        torch.testing.assert_close(first['target'], second['target'])

    def test_t0_fft_and_prediction_scaling(self):
        self.config.pred_scaler = 2
        self.config.target_noise_multiplier = 3
        data = self.setup.predict(self.model, self.batch, self.config, TrainProgress())
        torch.testing.assert_close(data['predicted'], torch.full((1, 1, 2, 2), -21.75))
        torch.testing.assert_close(data['target'], torch.full((1, 1, 2, 2), 5.0))
        self.config.t0_loss_target = True
        self.config.t0_velocity_equiv_weight = True
        self.config.do_fft_loss = True
        self.config.do_fft_velocity_equiv_weight = True
        data = self.setup.predict(self.model, self.batch, self.config, TrainProgress())
        torch.testing.assert_close(data['predicted'], torch.full((1, 1, 2, 2), 12.375))
        torch.testing.assert_close(data['target'], torch.ones(1, 1, 2, 2))
        torch.testing.assert_close(data['element_loss_scale'], torch.full((1, 1, 1, 1), 4.0))
        expected_fft = ((torch.fft.rfft2(data['predicted'].float(), norm='ortho').abs()
                         - torch.fft.rfft2(data['target'], norm='ortho').abs()).square() * 4).mean()
        torch.testing.assert_close(data['fft_loss'], expected_fft)
        self.setup.calculate_loss(self.model, self.batch, data, self.config).backward()
        self.assertTrue(torch.isfinite(self.model.transformer.lora.grad))

    def test_spatial_norm_std_auxiliary_loss(self):
        self.config.target_norm_std = True
        self.config.target_norm_std_value = 2
        data = self.setup.predict(self.model, self.batch, self.config, TrainProgress())
        torch.testing.assert_close(data['norm_std_loss'], torch.tensor(2.0))

    def test_source_timestep_semantics_share_conditioning_and_reconstruction_sigma(self):
        def actual_noise(latent, noise, timestep, schedule):
            sigma = ((timestep + 1) / 1000).view(-1, 1, 1, 1)
            return (1 - sigma) * latent + sigma * noise, sigma

        self.setup._add_noise_discrete = actual_noise
        self.config.aitoolkit_timestep_semantics = True
        self.config.diff_output_preservation = True
        self.config.do_guidance_loss = True
        self.config.guidance_loss_schedule = 'sigma'
        self.config.t0_loss_target = True
        self.config.t0_velocity_equiv_weight = True
        data = self.setup.predict(self.model, self.batch, self.config, TrainProgress())
        for call in self.model.transformer.calls:
            torch.testing.assert_close(call[1], torch.tensor([0.499]))
        noisy = torch.full((1, 1, 2, 2), 1.501)
        expected_t0 = noisy - 0.501 * (-noisy * 1.25 - 9)
        torch.testing.assert_close(data['predicted'], expected_t0)
        torch.testing.assert_close(data['element_loss_scale'], torch.tensor([[[[0.501 ** -2]]]]))

    def test_screen_i_no_ff_preset_selects_all_150_exported_modules(self):
        def block():
            layer = nn.Module()
            layer.attention = nn.Module()
            for name in ('to_q', 'to_k', 'to_v'):
                setattr(layer.attention, name, nn.Linear(2, 2))
            layer.attention.to_out = nn.Sequential(nn.Linear(2, 2))
            layer.adaLN_modulation = nn.Sequential(nn.Linear(2, 2))
            layer.feed_forward = nn.Sequential(nn.Linear(2, 2))
            return layer

        transformer = nn.Module()
        transformer.layers = nn.ModuleList([block() for _ in range(30)])
        transformer.noise_refiner = nn.ModuleList([block()])
        self.config.lora_rank = 1
        self.config.layer_filter_regex = True
        attention = BaseZImageSetup.LAYER_PRESETS['attn-only']
        toolkit = BaseZImageSetup.LAYER_PRESETS['ai-toolkit no-FF']
        attention_wrapper = LoRAModuleWrapper(transformer, 'transformer', self.config, attention['patterns'])
        toolkit_wrapper = LoRAModuleWrapper(transformer, 'transformer', self.config, toolkit['patterns'])
        self.assertEqual(len(attention_wrapper.lora_modules), 120)
        self.assertEqual(len(toolkit_wrapper.lora_modules), 150)
        self.assertTrue(all(name.startswith('layers.') for name in toolkit_wrapper.lora_modules))
        self.assertEqual(sum('adaLN_modulation' in name for name in toolkit_wrapper.lora_modules), 30)


if __name__ == '__main__':
    unittest.main()
