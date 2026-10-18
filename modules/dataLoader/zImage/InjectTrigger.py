from modules.util.enum.ConceptType import ConceptType

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class InjectTrigger(PipelineModule, RandomAccessPipelineModule):
    """Resolve ai-toolkit caption placeholders and prepend an absent literal trigger."""

    def __init__(self, trigger: str):
        super().__init__()
        self.trigger = trigger

    def length(self) -> int:
        return self._get_previous_length('prompt')

    def get_inputs(self) -> list[str]:
        return ['prompt', 'concept.type']

    def get_outputs(self) -> list[str]:
        return ['prompt']

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        prompt = self._get_previous_item(variation, 'prompt', index)
        concept_type = self._get_previous_item(variation, 'concept.type', index)
        prompt = prompt.replace('[name]', self.trigger).replace('[trigger]', self.trigger)
        # Regularization captions must not acquire the trained concept's trigger.
        if ConceptType(concept_type) != ConceptType.PRIOR_PREDICTION and self.trigger not in prompt:
            prompt = self.trigger + ' ' + prompt
        return {'prompt': prompt}
