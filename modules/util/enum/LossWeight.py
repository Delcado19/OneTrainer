from enum import Enum


class LossWeight(Enum):
    CONSTANT = 'CONSTANT'
    P2 = 'P2'
    MIN_SNR_GAMMA = 'MIN_SNR_GAMMA'
    DEBIASED_ESTIMATION = 'DEBIASED_ESTIMATION'
    SIGMA = 'SIGMA'
    AI_TOOLKIT_WEIGHTED = 'AI_TOOLKIT_WEIGHTED'
    AI_TOOLKIT_BELL = 'AI_TOOLKIT_BELL'
    AI_TOOLKIT_HALF_BELL = 'AI_TOOLKIT_HALF_BELL'

    def supports_flow_matching(self) -> bool:
        return self == LossWeight.CONSTANT \
            or self == LossWeight.SIGMA \
            or self == LossWeight.AI_TOOLKIT_WEIGHTED \
            or self == LossWeight.AI_TOOLKIT_BELL \
            or self == LossWeight.AI_TOOLKIT_HALF_BELL

    def __str__(self):
        return self.value
