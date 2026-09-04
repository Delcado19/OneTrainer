import torch
from torch import Tensor

from modules.util.enum.ConceptType import ConceptType


def _prior_prediction_mask(concept_types, device: torch.device) -> Tensor | None:
    if concept_types is None:
        return None

    return torch.tensor(
        [ConceptType(concept_type) == ConceptType.PRIOR_PREDICTION for concept_type in concept_types],
        dtype=torch.bool,
        device=device,
    )


def apply_z_image_differential_guidance(
        predicted: Tensor,
        target: Tensor,
        scale: float,
        concept_types=None,
) -> Tensor:
    with torch.no_grad():
        predicted_detached = predicted.detach()
        guided_target = predicted_detached + scale * (target - predicted_detached)

        prior_mask = _prior_prediction_mask(concept_types, target.device)
        if prior_mask is not None and prior_mask.any():
            # PRIOR_PREDICTION targets are replaced later by GenericTrainer; leave that mechanism isolated.
            guided_target = torch.where(prior_mask.view(-1, *[1 for _ in range(target.ndim - 1)]), target, guided_target)

        return guided_target
