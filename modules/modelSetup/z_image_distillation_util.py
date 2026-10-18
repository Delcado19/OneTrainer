from modules.util.enum.ConceptType import ConceptType

import torch
from torch import Tensor


def apply_z_image_guidance_loss(target: Tensor, unconditional: Tensor, timestep: Tensor,
                                scale: float | list[float] | Tensor, cfg_zero: bool = False,
                                schedule: str = "constant") -> Tensor:
    """Build ai-toolkit's detached contrastive target before differential guidance."""
    if schedule not in ("constant", "sigma"):
        raise ValueError(f"Unknown guidance loss schedule: {schedule}")
    with torch.no_grad():
        unconditional = unconditional.float()
        target = target.float()
        if not isinstance(scale, (int, float)):
            scale = torch.as_tensor(scale, device=target.device, dtype=target.dtype)
            if scale.numel() != target.shape[0]:
                raise ValueError("Guidance scale count must match the batch")
            scale = scale.view(-1, *[1 for _ in range(target.ndim - 1)])
        if cfg_zero:
            positive = target.flatten(1)
            negative = unconditional.flatten(1)
            alpha = (positive * negative).sum(1) / (negative.square().sum(1) + 1e-8)
            unconditional = unconditional * alpha.view(-1, *[1 for _ in range(target.ndim - 1)])
        if schedule == "sigma":
            scale = 1 + (scale - 1) * (timestep.float() / 1000).view(-1, *[1 for _ in range(target.ndim - 1)])
        return unconditional + scale * (target - unconditional)


def z_image_preservation_prompts(prompts, trigger: str, class_name: str, blank: bool = False) -> list[str]:
    """Replace the literal trigger, never reuse embeddings of the original caption."""
    if not isinstance(prompts, (list, tuple)) or not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError("Z-Image preservation requires the original batch prompts")
    if blank:
        return [""] * len(prompts)
    if not trigger:
        raise ValueError("Differential output preservation requires a non-empty trigger")
    return [prompt.replace(trigger, class_name) for prompt in prompts]


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
