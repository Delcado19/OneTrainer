from abc import ABCMeta
from contextlib import nullcontext
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.ZImageModel import ZImageModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupEmbeddingMixin import ModelSetupEmbeddingMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
from modules.modelSetup.z_image_distillation_util import (
    apply_z_image_differential_guidance,
    apply_z_image_guidance_loss,
    z_image_preservation_prompts,
)
from modules.util.checkpointing_util import (
    enable_checkpointing_for_qwen3_encoder_layers,
    enable_checkpointing_for_z_image_transformer,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BaseZImageSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupEmbeddingMixin,
    ModelSetupText2ImageMixin,
    metaclass=ABCMeta
):
    LAYER_PRESETS = {
        "full": [],
        "blocks": ["layers"],
        "attn-mlp": {'patterns': ["^(?=.*attention)(?!.*refiner).*", "^(?=.*feed_forward)(?!.*refiner).*"], 'regex': True},
        "attn-only": {'patterns': ["^(?=.*attention)(?!.*refiner).*"], 'regex': True},
        # ai-toolkit's no-FF Screen I export also trains every block's adaLN modulation.
        "ai-toolkit no-FF": {'patterns': [r"^layers\.\d+\.(attention\.|adaLN_modulation\.)"], 'regex': True},
    }

    def setup_optimizations(
            self,
            model: ZImageModel,
            config: TrainConfig,
    ):
        super().setup_optimizations(model, config)
        self._setup_model_part(model, config, "transformer", config.transformer, enable_checkpointing_for_z_image_transformer)
        self._setup_model_part(model, config, "text_encoder", config.text_encoder, enable_checkpointing_for_qwen3_encoder_layers, disable_fp16_autocast=True)
        self._setup_model_part(model, config, "vae", config.vae)

        self._set_attention_backend(model.transformer, config.attention_mechanism, mask=True)

    def predict(
            self,
            model: ZImageModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        with model.autocast_context:
            batch_seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)
            rand = Random(batch_seed)

            text_encoder_output = model.encode_text(
                train_device=self.train_device,
                batch_size=batch['latent_image'].shape[0],
                rand=rand,
                tokens=batch.get("tokens"),
                tokens_mask=batch.get("tokens_mask"),
                text_encoder_output=batch.get('text_encoder_hidden_state'),
                text_encoder_dropout_probability=config.text_encoder.dropout_probability if not deterministic else None,
            )
            scaled_latent_image = model.scale_latents(batch['latent_image'])

            preservation = config.diff_output_preservation or config.diff_output_preservation_blank
            preservation_embeds = None
            unconditional_embeds = None
            if config.do_blank_stabilization:
                prompts = batch.get('prompt')
                if (not isinstance(prompts, (list, tuple)) or len(prompts) != scaled_latent_image.shape[0]
                        or not all(isinstance(prompt, str) for prompt in prompts)):
                    raise ValueError("Blank stabilization requires the original batch prompts")
                blank = torch.tensor([not prompt.strip() for prompt in prompts], device=scaled_latent_image.device)
                scaled_latent_image = scaled_latent_image.masked_fill(blank.view(-1, 1, 1, 1), 0)
            if preservation or config.do_guidance_loss:
                # Alternate captions must be encoded anew even when the original caption is cached.
                model.materialize('text_encoder')
                try:
                    with torch.no_grad():
                        if preservation:
                            prompts = z_image_preservation_prompts(
                                batch.get('prompt'), config.diff_output_preservation_trigger,
                                config.diff_output_preservation_class, config.diff_output_preservation_blank,
                            )
                            if len(prompts) != scaled_latent_image.shape[0]:
                                raise ValueError("Preservation prompt count must match the latent batch")
                            preservation_embeds = model.encode_text(self.train_device, text=prompts)
                        if config.do_guidance_loss:
                            unconditional_embeds = model.encode_text(
                                self.train_device, text=[config.unconditional_prompt] * scaled_latent_image.shape[0],
                            )
                finally:
                    if config.latent_caching:
                        model.evict('text_encoder')

            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            shift = model.calculate_timestep_shift(scaled_latent_image.shape[-2], scaled_latent_image.shape[-1])
            timestep = self._get_timestep_discrete(
                model.noise_scheduler.config['num_train_timesteps'],
                deterministic,
                generator,
                scaled_latent_image.shape[0],
                config,
                shift = shift if config.dynamic_timestep_shifting else config.timestep_shift,
            )

            scaled_noisy_latent_image, sigma = self._add_noise_discrete(
                scaled_latent_image,
                latent_noise,
                timestep,
                model.noise_scheduler.timesteps,
            )
            latent_input = scaled_noisy_latent_image.unsqueeze(2).to(dtype=model.train_dtype.torch_dtype())
            latent_input_list = list(latent_input.unbind(dim=0))
            # OneTrainer stores a zero-based index; ai-toolkit conditions on the actual noise timestep.
            noise_timestep = timestep + 1 if config.aitoolkit_timestep_semantics else timestep

            def predict_flow(embeds):
                output = model.transformer(
                    latent_input_list, (1000 - noise_timestep) / 1000, embeds, return_dict=True,
                ).sample
                return -torch.stack(output, dim=0).squeeze(dim=2)

            preservation_target = None
            if preservation:
                # DOP is an additional class-prompt objective, not a replacement of the ordinary flow target.
                # GenericTrainer may already have disabled the adapters for a PRIOR_PREDICTION pass.
                prior_context = (self.prior_model(model, config)
                                 if any(adapter.is_applied() for adapter in model.adapters()) else nullcontext())
                with prior_context, torch.no_grad():
                    was_training = model.transformer.training
                    try:
                        model.transformer.eval()
                        preservation_target = predict_flow(preservation_embeds).detach()
                    finally:
                        model.transformer.train(was_training)

            predicted_flow = predict_flow(text_encoder_output) * config.pred_scaler


            flow = latent_noise * config.target_noise_multiplier - scaled_latent_image
            if config.do_guidance_loss:
                with torch.no_grad():
                    unconditional = predict_flow(unconditional_embeds)
                guidance_scale = config.guidance_loss_target
                if config.guidance_loss_target_min is not None and config.guidance_loss_target_max is not None:
                    guidance_scale = [rand.uniform(config.guidance_loss_target_min, config.guidance_loss_target_max)
                                      for _ in range(scaled_latent_image.shape[0])]
                flow = apply_z_image_guidance_loss(
                    flow, unconditional, noise_timestep, guidance_scale,
                    config.do_guidance_loss_cfg_zero, config.guidance_loss_schedule,
                ).to(flow.dtype)
            if config.differential_guidance and (not config.aitoolkit_guidance_semantics or config.do_guidance_loss):
                flow = apply_z_image_differential_guidance(
                    predicted_flow,
                    flow,
                    config.differential_guidance_scale,
                    batch.get("concept_type"),
                )
            model_output_data = {
                'loss_type': 'target',
                'timestep': timestep,
                'predicted': predicted_flow,
                'target': flow,
            }
            sigma_view = (noise_timestep.float() / 1000).clamp_min(0.001).view(-1, 1, 1, 1)
            if config.t0_loss_target or config.do_fft_loss:
                t0 = scaled_noisy_latent_image - sigma_view * predicted_flow
                if config.t0_loss_target:
                    model_output_data['predicted'] = t0
                    model_output_data['target'] = scaled_latent_image.detach()
                    if config.t0_velocity_equiv_weight:
                        model_output_data['element_loss_scale'] = sigma_view.clamp_min(0.1).pow(-2)
                if config.do_fft_loss:
                    target_mag = torch.fft.rfft2(scaled_latent_image.detach().float(), norm='ortho').abs()
                    predicted_mag = torch.fft.rfft2(t0.float(), norm='ortho').abs()
                    fft_loss = (predicted_mag - target_mag).square()
                    if config.do_fft_velocity_equiv_weight:
                        fft_loss = fft_loss * sigma_view.clamp_min(0.1).pow(-2)
                    model_output_data['fft_loss'] = fft_loss.mean()
            if config.target_norm_std:
                model_output_data['norm_std_loss'] = (
                    predicted_flow.std(dim=(2, 3), keepdim=True) - config.target_norm_std_value
                ).abs().mean()
            if preservation:
                preservation_predicted = predict_flow(preservation_embeds)
                multiplier = (config.blank_prompt_preservation_multiplier if config.diff_output_preservation_blank
                              else config.diff_output_preservation_multiplier)
                # ai-toolkit adds raw preservation MSE after the weighted/masked primary loss.
                model_output_data['preservation_loss'] = torch.nn.functional.mse_loss(
                    preservation_predicted.float(), preservation_target.float(),
                ) * multiplier

            if config.debug_mode:
                with torch.no_grad():
                    predicted_scaled_latent_image = scaled_noisy_latent_image - predicted_flow * sigma
                    self._save_tokens("7-prompt", batch['tokens'], model.tokenizer, config, train_progress)
                    self._save_latent("1-noise", latent_noise, config, train_progress)
                    self._save_latent("2-noisy_image", scaled_noisy_latent_image, config, train_progress)
                    self._save_latent("3-predicted_flow", predicted_flow, config, train_progress)
                    self._save_latent("4-flow", flow, config, train_progress)
                    self._save_latent("5-predicted_image", predicted_scaled_latent_image, config, train_progress)
                    self._save_latent("6-image", scaled_latent_image, config, train_progress)

        return model_output_data

    def calculate_loss(
            self,
            model: ZImageModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        loss = self._flow_matching_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            sigmas=model.noise_scheduler.sigmas,
        ).mean()
        return loss + data.get('preservation_loss', 0) + data.get('fft_loss', 0) + data.get('norm_std_loss', 0)

    def prepare_text_caching(self, model: ZImageModel, config: TrainConfig):
        model.materialize_only("text_encoder")

        model.eval()
