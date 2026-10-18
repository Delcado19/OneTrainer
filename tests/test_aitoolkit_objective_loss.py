import unittest

from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.util.config.TrainConfig import TrainConfig

import torch


class ToolkitObjectiveLossTest(unittest.TestCase):
    def test_pseudo_huber_mask_and_gradient(self):
        config = TrainConfig.default_values()
        config.mse_strength = 0
        config.pseudo_huber_strength = 1
        predicted = torch.tensor([[[[0.2, 0.5]]]], requires_grad=True)
        batch = {'loss_weight': torch.ones(1), 'latent_mask': torch.tensor([[[[1., 0.]]]])}
        data = {'loss_type': 'target', 'predicted': predicted, 'target': torch.zeros_like(predicted)}
        for masked in (False, True):
            config.masked_training = masked
            config.unmasked_weight = 0
            config.normalize_masked_area_loss = False
            predicted.grad = None
            loss = ModelSetupDiffusionLossMixin()._flow_matching_losses(batch, data, config, torch.device('cpu')).mean()
            expected = ((predicted.detach().square() + .0001).sqrt() - .01)
            if masked:
                expected *= batch['latent_mask']
            torch.testing.assert_close(loss.detach(), expected.mean())
            loss.backward()
            expected_gradient = predicted.detach() / (predicted.detach().square() + .0001).sqrt() / 2
            if masked:
                expected_gradient *= batch['latent_mask']
            torch.testing.assert_close(predicted.grad, expected_gradient)

    def test_reconstruction_weight_is_per_sample(self):
        config = TrainConfig.default_values()
        predicted = torch.ones(2, 1, 2, 2, requires_grad=True)
        batch = {'loss_weight': torch.ones(2)}
        data = {'loss_type': 'target', 'predicted': predicted, 'target': torch.zeros_like(predicted),
                'element_loss_scale': torch.tensor([4., 16.]).view(2, 1, 1, 1)}
        loss = ModelSetupDiffusionLossMixin()._flow_matching_losses(batch, data, config, torch.device('cpu'))
        torch.testing.assert_close(loss, torch.tensor([4., 16.]))
        loss.mean().backward()
        torch.testing.assert_close(predicted.grad[0], torch.ones_like(predicted.grad[0]))
        torch.testing.assert_close(predicted.grad[1], torch.full_like(predicted.grad[1], 4))


if __name__ == '__main__':
    unittest.main()
