import unittest
from types import SimpleNamespace

from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.util.enum.TimestepDistribution import TimestepDistribution

import torch


class _NoiseMixinHarness(ModelSetupNoiseMixin):
    """Bare instance to exercise the mixin's timestep sampling in isolation."""


def _make_config(timestep_distribution: TimestepDistribution) -> SimpleNamespace:
    return SimpleNamespace(
        timestep_shift=1.0,
        min_noising_strength=0.0,
        max_noising_strength=1.0,
        timestep_distribution=timestep_distribution,
    )


class TimestepStyleBiasedTest(unittest.TestCase):
    def test_style_biased_skews_toward_low_noise_timesteps(self):
        mixin = _NoiseMixinHarness()
        generator = torch.Generator().manual_seed(0)
        num_train_timesteps = 1000

        style_timesteps = mixin._get_timestep_discrete(
            num_train_timesteps, False, generator, 20000, _make_config(TimestepDistribution.STYLE_BIASED),
        ).float()
        uniform_timesteps = mixin._get_timestep_discrete(
            num_train_timesteps, False, generator, 20000, _make_config(TimestepDistribution.UNIFORM),
        ).float()

        # E[u**3] for u~U(0,1) is 1/4, so the mean should land near 250 (num_train_timesteps/4),
        # well below the uniform baseline's ~500 -- i.e. skewed toward low timestep values,
        # which is low noise in OneTrainer's sigma convention.
        self.assertLess(style_timesteps.mean().item(), uniform_timesteps.mean().item() * 0.6)
        self.assertAlmostEqual(style_timesteps.mean().item(), num_train_timesteps / 4, delta=15)
        self.assertGreaterEqual(style_timesteps.min().item(), 0)
        self.assertLess(style_timesteps.max().item(), num_train_timesteps)

    def test_style_biased_respects_noising_strength_window(self):
        mixin = _NoiseMixinHarness()
        generator = torch.Generator().manual_seed(1)
        num_train_timesteps = 1000
        config = _make_config(TimestepDistribution.STYLE_BIASED)
        config.min_noising_strength = 0.2
        config.max_noising_strength = 0.8

        timesteps = mixin._get_timestep_discrete(
            num_train_timesteps, False, generator, 5000, config,
        ).float()

        self.assertGreaterEqual(timesteps.min().item(), 200)
        self.assertLess(timesteps.max().item(), 800)


if __name__ == "__main__":
    unittest.main()
