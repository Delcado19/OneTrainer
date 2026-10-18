import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.create import create_lr_scheduler
from modules.util.enum.DataType import DataType
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.TimedActionMixin import TimedActionMixin
from modules.util.TrainProgress import TrainProgress

import torch


class OptimizerUpdateBudgetTest(unittest.TestCase):
    def scheduler(self, updates, global_step=12, accumulation=2, warmup=0):
        config = TrainConfig.default_values()
        config.max_train_steps = 10
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([{'params': [parameter], 'lr': 1., 'initial_lr': 1.}], lr=1.)
        scheduler = create_lr_scheduler(
            config, optimizer, LearningRateScheduler.LINEAR, warmup, 1, 0,
            num_epochs=1, batch_size=1, approximate_epoch_length=1,
            gradient_accumulation_steps=accumulation, global_step=global_step, optimizer_step=updates,
        )
        return optimizer, scheduler

    def test_resume_uses_updates_after_accumulation_change_or_amp_skip(self):
        # Twelve microsteps may mean three successful updates, or fewer after AMP overflow.
        for updates in (0, 2, 3):
            with self.subTest(updates=updates):
                optimizer, scheduler = self.scheduler(updates)
                self.assertEqual(scheduler.last_epoch, updates)
                self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 1 - updates / 10)

    def test_legacy_scheduler_fallback_and_warmup(self):
        _, scheduler = self.scheduler(None)
        self.assertEqual(scheduler.last_epoch, 6)
        optimizer, scheduler = self.scheduler(3, warmup=8)
        self.assertEqual(scheduler.last_epoch, 3)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], .75)

    def test_real_training_loop_reaches_budget_beyond_epoch_estimate(self):
        trainer = object.__new__(GenericTrainer)
        TimedActionMixin.__init__(trainer)
        config = trainer.config = TrainConfig.default_values()
        config.train_device = 'cpu'
        config.train_dtype = DataType.FLOAT_32
        config.epochs = 1
        config.max_train_steps = 3
        config.gradient_accumulation_steps = 4
        config.latent_caching = False
        config.learning_rate_warmup_steps = 0
        config.clip_grad_norm = None
        parameter = torch.nn.Parameter(torch.ones(1))
        trainer.parameters = [parameter]
        trainer.model = SimpleNamespace(train_progress=TrainProgress(),
                                        optimizer=torch.optim.SGD([parameter], lr=.01), ema=None)
        trainer.commands = TrainCommands()
        trainer.callbacks = Mock()
        trainer.tensorboard = Mock()
        trainer.model_setup = Mock()
        trainer.model_setup.calculate_loss.side_effect = lambda *args: parameter.square().mean()
        batch = {'concept_type': ['STANDARD']}
        dataset = Mock()
        dataset.approximate_length.return_value = 1
        trainer.data_loader = Mock()
        trainer.data_loader.get_data_set.return_value = dataset
        trainer.data_loader.get_data_loader.side_effect = lambda: [batch]
        for action in ('sample', 'backup', 'save', 'gc'):
            setattr(trainer, '_GenericTrainer__needs_' + action, lambda *args: False)
        trainer._GenericTrainer__execute_sample_during_training = Mock()
        with patch('modules.trainer.GenericTrainer.multi.is_master', return_value=False), \
                patch('modules.trainer.GenericTrainer.multi.sync_commands'), \
                patch('modules.trainer.GenericTrainer.multi.reduce_tensor_mean'), \
                patch('modules.trainer.GenericTrainer.multi.reduce_grads_mean'), \
                patch('modules.trainer.GenericTrainer.torch_gc'), \
                patch('modules.trainer.GenericTrainer.bf16_stochastic_rounding_set_seed'):
            trainer.train()
            self.assertEqual(trainer.model.train_progress.optimizer_step, 3)
            self.assertEqual(trainer.model.train_progress.global_step, 12)
            self.assertEqual(dataset.start_next_epoch.call_count, 12)
            dataset.approximate_length.return_value = 0
            trainer.model.train_progress = TrainProgress()
            with self.assertRaisesRegex(ValueError, 'no batches'):
                trainer.train()


if __name__ == '__main__':
    unittest.main()
