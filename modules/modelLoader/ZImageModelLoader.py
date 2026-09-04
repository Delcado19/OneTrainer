import os
import traceback

from modules.model.ZImageModel import ZImageModel
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.config.TrainConfig import QuantizationConfig, TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType, PeftType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    GGUFQuantizationConfig,
    ZImageTransformer2DModel,
)
from transformers import (
    Qwen2Tokenizer,
    Qwen3ForCausalLM,
)
import huggingface_hub
from safetensors.torch import load_file


class ZImageModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def __load_internal(
            self,
            model: ZImageModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            assistant_lora_model_name: str,
            quantization: QuantizationConfig,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, transformer_model_name, vae_model_name,
                assistant_lora_model_name, quantization,
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: ZImageModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            assistant_lora_model_name: str,
            quantization: QuantizationConfig,
    ):
        tokenizer = Qwen2Tokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        text_encoder = self._load_transformers_sub_module(
            Qwen3ForCausalLM,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
        )

        if vae_model_name:
            vae = self._load_diffusers_sub_module(
                AutoencoderKL,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                vae_model_name,
            )
        else:
            vae = self._load_diffusers_sub_module(
                AutoencoderKL,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                base_model_name,
                "vae",
            )

        if transformer_model_name and assistant_lora_model_name:
            raise ValueError("assistant_lora_model_name cannot be combined with a separate transformer model override")

        if transformer_model_name:
            transformer = ZImageTransformer2DModel.from_single_file(
                transformer_model_name,
                #avoid loading the transformer in float32:
                torch_dtype = torch.bfloat16 if weight_dtypes.transformer.torch_dtype() is None else weight_dtypes.transformer.torch_dtype(),
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16) if weight_dtypes.transformer.is_gguf() else None,
            )
            transformer = self._convert_diffusers_sub_module_to_dtype(
                transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
            )
        elif assistant_lora_model_name:
            transformer = self._load_diffusers_sub_module(
                ZImageTransformer2DModel,
                DataType.FLOAT_32,
                weight_dtypes.train_dtype,
                base_model_name,
                "transformer",
                None,
            )
            self.__load_assistant_lora(model, transformer, assistant_lora_model_name)
            transformer = self._convert_diffusers_sub_module_to_dtype(
                transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
            )
        else:
            transformer = self._load_diffusers_sub_module(
                ZImageTransformer2DModel,
                weight_dtypes.transformer,
                weight_dtypes.train_dtype,
                base_model_name,
                "transformer",
                quantization,
            )

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.transformer = transformer

    def __assistant_lora_path(self, assistant_lora_model_name: str) -> str:
        if os.path.isfile(assistant_lora_model_name):
            return assistant_lora_model_name
        if "/" not in assistant_lora_model_name:
            raise ValueError("assistant_lora_model_name must be a local safetensors file or repo_id/filename")

        repo_id, filename = assistant_lora_model_name.rsplit("/", 1)
        return huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)

    def __load_assistant_lora(
            self,
            model: ZImageModel,
            transformer: ZImageTransformer2DModel,
            assistant_lora_model_name: str,
    ):
        model.transformer = transformer
        state_dict = ZImageLoRALoader()._to_canonical(
            model,
            load_file(self.__assistant_lora_path(assistant_lora_model_name)),
        )
        module_names = {
            key.removeprefix("transformer.").removesuffix(".lora_down.weight")
            for key in state_dict
            if key.startswith("transformer.") and key.endswith(".lora_down.weight")
        }
        if not module_names:
            raise RuntimeError("Assistant LoRA has no Z-Image transformer LoRA keys after normalization")

        rank = next(iter(state_dict[f"transformer.{name}.lora_down.weight"].shape[0] for name in module_names))
        if any(state_dict[f"transformer.{name}.lora_down.weight"].shape[0] != rank for name in module_names):
            raise RuntimeError("Assistant LoRA uses mixed ranks, which this Z-Image MVP cannot merge")

        config = TrainConfig.default_values()
        config.peft_type = PeftType.LORA
        config.lora_rank = rank
        config.lora_alpha = float(rank)
        model.transformer_assistant_lora = LoRAModuleWrapper(transformer, "transformer", config)

        unknown = sorted(module_names - set(model.transformer_assistant_lora.lora_modules))
        if unknown:
            raise RuntimeError(f"Assistant LoRA contains keys that do not match Z-Image transformer modules: {unknown[:3]}")

        model.transformer_assistant_lora.load_state_dict(state_dict)
        model.transformer_assistant_lora.apply_to_module(1.0)
        model.assistant_lora_merged = True

    def __load_safetensors(
            self,
            model: ZImageModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            assistant_lora_model_name: str,
            quantization: QuantizationConfig,
    ):
        #no single file .safetensors for Qwen available at the time of writing this code
        raise NotImplementedError("Loading of single file Z-Image models not supported. Use the diffusers model instead. Optionally, transformer-only safetensor files can be loaded by overriding the transformer.")

    def load(
            self,
            model: ZImageModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        stacktraces = []

        if model_names.transformer_model and model_names.assistant_lora:
            raise ValueError("assistant_lora_model_name cannot be combined with a separate transformer model override")

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model,
                model_names.vae_model, model_names.assistant_lora, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model,
                model_names.vae_model, model_names.assistant_lora, quantization,
            )
            return
        except Exception:
            if model_names.assistant_lora:
                raise
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_safetensors(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model,
                model_names.vae_model, model_names.assistant_lora, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)



class ZImageLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()


    def load(
            self,
            model: ZImageModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)


ZImageLoRAModelLoader = make_lora_model_loader(
    model_spec_map={
        ModelType.Z_IMAGE: "resources/sd_model_spec/z_image-lora.json",
    },
    model_class=ZImageModel,
    model_loader_class=ZImageModelLoader,
    lora_loader_class=ZImageLoRALoader,
    embedding_loader_class=None,
)

ZImageFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={
        ModelType.Z_IMAGE: "resources/sd_model_spec/z_image.json",
    },
    model_class=ZImageModel,
    model_loader_class=ZImageModelLoader,
    embedding_loader_class=None,
)
