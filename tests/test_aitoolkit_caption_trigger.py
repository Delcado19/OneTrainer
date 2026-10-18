import unittest
from unittest.mock import patch

from modules.dataLoader.zImage.InjectTrigger import InjectTrigger
from modules.dataLoader.ZImageBaseDataLoader import ZImageBaseDataLoader
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.ModelType import ModelType

from mgds.pipelineModules.DropTags import DropTags
from mgds.pipelineModules.ShuffleTags import ShuffleTags


class AiToolkitCaptionTriggerTest(unittest.TestCase):
    def test_caption_placeholders_missing_and_existing_trigger(self):
        module = InjectTrigger('sks')
        for prompt, expected in [('[trigger] portrait of [name]', 'sks portrait of sks'),
                                 ('a portrait', 'sks a portrait'), ('sks portrait', 'sks portrait'),
                                 ('', 'sks ')]:
            with self.subTest(prompt=prompt):
                values = {'prompt': prompt, 'concept.type': ConceptType.STANDARD}
                with patch.object(module, '_get_previous_item', side_effect=lambda v, name, i, values=values: values[name]):
                    self.assertEqual(module.get_item(0, 0)['prompt'], expected)
        values = {'prompt': 'a person', 'concept.type': ConceptType.PRIOR_PREDICTION}
        with patch.object(module, '_get_previous_item', side_effect=lambda v, name, i: values[name]):
            self.assertEqual(module.get_item(0, 0)['prompt'], 'a person')

    def test_real_augmentation_order_and_full_caption_dropout(self):
        config = TrainConfig.default_values()
        config.model_type = ModelType.Z_IMAGE
        config.aitoolkit_trigger_word = 'sks'
        loader = object.__new__(ZImageBaseDataLoader)
        modules = loader._augmentation_modules(config)
        trigger = next(m for m in modules if isinstance(m, InjectTrigger))
        dropout = next(m for m in modules if isinstance(m, DropTags))
        shuffle = next(m for m in modules if isinstance(m, ShuffleTags))
        self.assertLess(modules.index(shuffle), modules.index(trigger))
        self.assertLess(modules.index(trigger), modules.index(dropout))
        concept = ConceptConfig.default_values()
        concept.text.keep_tags_count = 0
        concept.text.tag_dropout_enable = True
        concept.text.tag_dropout_probability = 1.0
        values = {'prompt': 'sks portrait'}

        def previous(variation, name, index):
            if name == 'prompt':
                return values[name]
            current = concept
            for part in name.split('.')[1:]:
                current = getattr(current, part)
            return current

        with patch.object(dropout, '_get_previous_item', side_effect=previous), \
                patch.object(dropout, '_get_rand') as rand:
            rand.return_value.random.return_value = 0.0
            self.assertEqual(dropout.get_item(0, 0)['prompt'], '')


if __name__ == '__main__':
    unittest.main()
