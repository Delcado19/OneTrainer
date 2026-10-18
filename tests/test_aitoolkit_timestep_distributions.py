import unittest
from types import SimpleNamespace

from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.util.enum.TimestepDistribution import TimestepDistribution

import torch


def config(mode, shift=1.0, minimum=0.0, maximum=1.0):
    return SimpleNamespace(timestep_distribution=mode, timestep_shift=shift,
                           min_noising_strength=minimum, max_noising_strength=maximum)


class AiToolkitTimestepDistributionsTest(unittest.TestCase):
    def test_content_is_high_noise_and_deterministic(self):
        mixin = ModelSetupNoiseMixin()
        actual = mixin._get_timestep_discrete(
            1000, False, torch.Generator().manual_seed(71), 20000,
            config(TimestepDistribution.CONTENT_BIASED),
        )
        u = torch.rand(20000, generator=torch.Generator().manual_seed(71))
        torch.testing.assert_close(actual, ((1 - u ** 3) * 1000).clamp(max=999).int(), rtol=0, atol=0)
        self.assertAlmostEqual(actual.float().mean().item(), 750, delta=10)
        self.assertGreaterEqual(actual.min().item(), 0)
        self.assertLess(actual.max().item(), 1000)

    def test_content_respects_noise_window(self):
        actual = ModelSetupNoiseMixin()._get_timestep_discrete(
            1000, False, torch.Generator().manual_seed(71), 20000,
            config(TimestepDistribution.CONTENT_BIASED, minimum=0.2, maximum=0.8),
        )
        self.assertGreaterEqual(actual.min().item(), 200)
        self.assertLess(actual.max().item(), 800)

    def test_fixed_schedules_exact_choices_bypass_window_and_shift(self):
        source_choices = {
            TimestepDistribution.AI_TOOLKIT_ONE_STEP: [0],
            TimestepDistribution.AI_TOOLKIT_TWO_STEP: [0, 499],
            TimestepDistribution.AI_TOOLKIT_FOUR_STEP: [0, 250, 500, 750],
            TimestepDistribution.AI_TOOLKIT_EIGHT_STEP: [0, 125, 250, 375, 500, 625, 750, 875],
        }
        for mode, descending in source_choices.items():
            with self.subTest(mode=mode):
                mixin = ModelSetupNoiseMixin()
                actual = mixin._get_timestep_discrete(
                    1000, False, torch.Generator().manual_seed(42), 10000,
                    config(mode, shift=7.0, minimum=0.2, maximum=0.3),
                )
                self.assertEqual(set(actual.tolist()), {999 - index for index in descending})
                repeated = mixin._get_timestep_discrete(
                    1000, False, torch.Generator().manual_seed(42), 10000, config(mode),
                )
                torch.testing.assert_close(actual, repeated)
                # Test sigma values, not merely the enum or the branch's integer representation.
                self.assertEqual(set(((actual.double() + 1) / 1000).tolist()),
                                 {(1000 - index) / 1000 for index in descending})

    def test_fixed_schedules_reject_other_lengths(self):
        with self.assertRaisesRegex(ValueError, "1000"):
            ModelSetupNoiseMixin()._get_timestep_discrete(
                2000, False, torch.Generator().manual_seed(42), 3,
                config(TimestepDistribution.AI_TOOLKIT_TWO_STEP),
            )


if __name__ == "__main__":
    unittest.main()
