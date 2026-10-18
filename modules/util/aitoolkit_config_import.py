"""Strict import of supported ai-toolkit Z-Image image-LoRA recipes.

This maps recipe controls, not RNG, image preprocessing, quantizer implementation,
or historical runtime equivalence. Unknown execution controls fail before export.
"""
import math
import re

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.LossWeight import LossWeight
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TimestepDistribution import TimestepDistribution
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod

IMPORT_LIMITATIONS = (
    "Recipe mapping does not establish numerical or image-quality parity: RNG, crop/buckets, VAE posterior and quantization differ.",
    "Output/workspace paths use OneTrainer relative defaults; source training_folder is not reused.",
    "Logging/UI/database metadata and save retention are not imported; OneTrainer keeps its own logging and save lifecycle.",
    "Sampling start/order and resolution exposure may differ; inspect inline concepts and samples before training.",
)


def _section(value, path, allowed):
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path} option names must be strings")
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"Unsupported {path} option(s): {', '.join(unknown)}. Implement their semantics before importing this recipe.")
    return value


def _fixed(data, path, defaults):
    for key, default in defaults.items():
        if data.get(key, default) != default:
            raise ValueError(f"Unsupported {path}.{key}: this importer requires {default!r}")


def _positive(value, path, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{path} must be finite and positive")
    if integer and not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return value


def _dtype(value, path):
    values = {"bf16": DataType.BFLOAT_16, "bfloat16": DataType.BFLOAT_16,
              "fp16": DataType.FLOAT_16, "float16": DataType.FLOAT_16,
              "fp32": DataType.FLOAT_32, "float32": DataType.FLOAT_32}
    if value not in values:
        raise ValueError(f"Unsupported {path}: {value!r}")
    return values[value]


def _substrings(value, path):
    if value is None:
        return ""
    if not isinstance(value, list) or any(not isinstance(s, str) or not s or ',' in s for s in value):
        raise ValueError(f"{path} must be a list of nonempty substrings without commas")
    return ','.join(value)


def import_aitoolkit_config(source: dict) -> TrainConfig:
    """Build a native config with inline concepts/samples; perform no file or model reads."""
    _section(source, "root", {"job", "config", "meta"})
    _fixed(source, "root", {"job": "extension"})
    job = _section(source.get("config"), "config", {"name", "process"})
    processes = job.get("process")
    if not isinstance(processes, list) or len(processes) != 1:
        raise ValueError("config.process must contain exactly one diffusion_trainer")
    process = _section(processes[0], "process", {
        "type", "training_folder", "device", "network", "save", "datasets", "train", "model", "sample",
        "sqlite_db_path", "performance_log_every", "logging", "trigger_word",
    })
    _fixed(process, "process", {"type": "diffusion_trainer"})
    trigger = process.get("trigger_word")
    if trigger is None:
        trigger = ""
    if not isinstance(trigger, str) or (trigger and not trigger.strip()):
        raise ValueError("process.trigger_word must be a nonblank string or unset")
    model = _section(process.get("model"), "model", {
        "name_or_path", "arch", "assistant_lora_path", "quantize", "qtype", "quantize_te", "qtype_te",
    })
    if model.get("arch") not in ("zimage", "zimage:turbo"):
        raise ValueError("model.arch must be zimage or zimage:turbo; other architectures are not imported")
    if not isinstance(model.get("name_or_path"), str) or not model["name_or_path"]:
        raise ValueError("model.name_or_path must be a nonempty local path or Hub model ID")
    _fixed(model, "model", {"qtype": "qfloat8", "qtype_te": "qfloat8"})
    network = _section(process.get("network"), "network", {
        "type", "linear", "rank", "linear_alpha", "alpha", "dropout", "network_kwargs", "transformer_only",
    })
    _fixed(network, "network", {"type": "lora", "transformer_only": True})
    kwargs = _section(network.get("network_kwargs", {}), "network.network_kwargs", {
        "ignore_if_contains", "only_if_contains", "rank_dropout", "module_dropout", "parameter_threshold",
    })
    train = _section(process.get("train"), "train", {
        "batch_size", "steps", "gradient_accumulation_steps", "train_unet", "train_text_encoder",
        "gradient_checkpointing", "noise_scheduler", "optimizer", "optimizer_params", "timestep_type",
        "content_or_style", "loss_type", "lr", "dtype", "lr_scheduler", "num_train_timesteps",
        "do_differential_guidance", "differential_guidance_scale", "do_guidance_loss", "guidance_loss_target",
        "do_guidance_loss_cfg_zero", "guidance_loss_schedule", "unconditional_prompt", "do_blank_stabilization",
        "diff_output_preservation", "diff_output_preservation_class", "diff_output_preservation_multiplier",
        "blank_prompt_preservation", "blank_prompt_preservation_multiplier", "skip_first_sample",
        "linear_timesteps", "linear_timesteps2", "pred_scaler", "target_noise_multiplier",
        "target_norm_std", "target_norm_std_value", "t0_loss_target", "t0_velocity_equiv_weight",
        "do_fft_loss", "do_fft_velocity_equiv_weight",
    })
    _fixed(train, "train", {"train_unet": True, "train_text_encoder": False, "noise_scheduler": "flowmatch",
                              "lr_scheduler": "constant", "num_train_timesteps": 1000})
    if train.get("noise_scheduler", "ddpm") != "flowmatch":
        raise ValueError("train.noise_scheduler must be explicitly flowmatch")
    if train.get("loss_type", "mse") not in ("mse", "mae", "pseudo_huber"):
        raise ValueError("train.loss_type must be mse, mae or pseudo_huber; other loss execution is unsupported")
    if train.get("optimizer", "adamw") not in ("adamw8bit", "adamw"):
        raise ValueError("train.optimizer must be adamw8bit or adamw")
    optimizer = _section(train.get("optimizer_params", {}), "train.optimizer_params", {"weight_decay", "betas"})
    timestep_type = train.get("timestep_type", "sigmoid")
    distributions = {"balanced": TimestepDistribution.UNIFORM, "style": TimestepDistribution.STYLE_BIASED,
                     "content": TimestepDistribution.CONTENT_BIASED}
    fixed_distributions = {"one_step": TimestepDistribution.AI_TOOLKIT_ONE_STEP,
                           "two_step": TimestepDistribution.AI_TOOLKIT_TWO_STEP,
                           "four_step": TimestepDistribution.AI_TOOLKIT_FOUR_STEP,
                           "eight_step": TimestepDistribution.AI_TOOLKIT_EIGHT_STEP}
    if timestep_type not in ("linear", "weighted", *fixed_distributions):
        raise ValueError(f"Unsupported train.timestep_type: {timestep_type!r}; next_sample/shift schedules require different execution")
    style = train.get("content_or_style", "balanced")
    if style not in distributions:
        raise ValueError(f"Unsupported train.content_or_style: {style!r}")

    for mapping, path, fields in [
        (model, "model", ("quantize", "quantize_te")),
        (train, "train", ("gradient_checkpointing", "linear_timesteps", "linear_timesteps2", "skip_first_sample")),
    ]:
        for field in fields:
            if field in mapping and type(mapping[field]) is not bool:
                raise ValueError(f"{path}.{field} must be bool")
    config = TrainConfig.default_values()
    config.model_type = ModelType.Z_IMAGE
    config.training_method = TrainingMethod.LORA
    config.base_model_name = model["name_or_path"]
    config.assistant_lora_model_name = model.get("assistant_lora_path", "") or ""
    config.aitoolkit_guidance_semantics = True
    config.aitoolkit_timestep_semantics = True
    config.aitoolkit_trigger_word = trigger
    config.train_device = process.get("device", "cuda")
    config.train_dtype = _dtype(train.get("dtype", "fp32"), "train.dtype")
    config.transformer.weight_dtype = DataType.FLOAT_8 if model.get("quantize", False) else config.train_dtype
    config.text_encoder.weight_dtype = DataType.FLOAT_8 if model.get("quantize_te", False) else config.train_dtype
    config.vae.weight_dtype = config.train_dtype
    # toolkit/models/v2/diffusion_models/z_image.py excludes embedders/final projections,
    # while block quantization and extras include refiners; Qwen uses model.layers.
    config.quantization.layer_filter = r"(^|\.)(layers|noise_refiner|context_refiner)\."
    config.quantization.layer_filter_regex = True
    config.quantization.layer_filter_preset = "full"
    config.text_encoder.train = False
    config.transformer.train = True
    config.transformer.gradient_checkpointing = train.get("gradient_checkpointing", True)
    config.lora_rank = _positive(network.get("rank", network.get("linear", 4)), "network.linear/rank", integer=True)
    # ai-toolkit forces alpha=rank for transformer PEFT regardless of linear_alpha.
    config.lora_alpha = config.lora_rank
    config.lora_weight_dtype = config.train_dtype
    config.dropout_probability = network.get("dropout") or 0.0
    config.lora_rank_dropout = kwargs.get("rank_dropout", 0.0) or 0.0
    config.lora_module_dropout = kwargs.get("module_dropout", 0.0) or 0.0
    for field in ("dropout_probability", "lora_rank_dropout", "lora_module_dropout"):
        value = getattr(config, field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"network {field} must be between zero and one")
    config.lora_parameter_threshold = kwargs.get("parameter_threshold", 0)
    if (isinstance(config.lora_parameter_threshold, bool) or not isinstance(config.lora_parameter_threshold, int)
            or config.lora_parameter_threshold < 0):
        raise ValueError("network.network_kwargs.parameter_threshold must be a nonnegative integer")
    config.lora_include = _substrings(kwargs.get("only_if_contains"), "network.network_kwargs.only_if_contains")
    config.lora_exclude = _substrings(kwargs.get("ignore_if_contains"), "network.network_kwargs.ignore_if_contains")
    config.layer_filter_regex = True
    config.layer_filter_preset = "full"
    # Match Screen I's actual 150-module inventory, including modulation, excluding refiners.
    config.layer_filter = (r"^layers\.\d+\.(attention\.|adaLN_modulation\.)"
                           if "feed_forward" in (kwargs.get("ignore_if_contains") or []) else r"^layers\.\d+\.")
    config.max_train_steps = _positive(train.get("steps", 1000), "train.steps", integer=True)
    config.batch_size = _positive(train.get("batch_size", 1), "train.batch_size", integer=True)
    config.gradient_accumulation_steps = _positive(train.get("gradient_accumulation_steps", 1), "train.gradient_accumulation_steps", integer=True)
    # The trainer stops by successful updates; this is only the UI/cache-only epoch estimate.
    config.epochs = config.max_train_steps * config.gradient_accumulation_steps
    config.learning_rate = _positive(train.get("lr", 1e-6), "train.lr")
    config.learning_rate_warmup_steps = 0
    config.optimizer.optimizer = Optimizer.ADAMW_8BIT if train.get("optimizer", "adamw") == "adamw8bit" else Optimizer.ADAMW
    config.optimizer.eps = 1e-6  # toolkit/optimizer.py sets this explicitly, independent of PyTorch defaults.
    config.optimizer.weight_decay = optimizer.get("weight_decay", 0.01)
    betas = optimizer.get("betas", [0.9, 0.999])
    if not isinstance(betas, (list, tuple)) or len(betas) != 2 or any(
            isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or not 0 < beta < 1 for beta in betas):
        raise ValueError("train.optimizer_params.betas must contain two finite values between zero and one")
    config.optimizer.beta1, config.optimizer.beta2 = betas
    if (isinstance(config.optimizer.weight_decay, bool) or not isinstance(config.optimizer.weight_decay, (int, float))
            or not math.isfinite(config.optimizer.weight_decay) or config.optimizer.weight_decay < 0):
        raise ValueError("train.optimizer_params.weight_decay must be finite and nonnegative")
    config.loss_weight_fn = (
        LossWeight.AI_TOOLKIT_WEIGHTED if timestep_type == "weighted" else
        LossWeight.AI_TOOLKIT_HALF_BELL if train.get("linear_timesteps2", False) else
        LossWeight.AI_TOOLKIT_BELL if train.get("linear_timesteps", False) else LossWeight.CONSTANT
    )
    if train.get("loss_type", "mse") != "mse":
        config.mse_strength = 0.0
        config.mae_strength = 1.0 if train["loss_type"] == "mae" else 0.0
        config.pseudo_huber_strength = 1.0 if train["loss_type"] == "pseudo_huber" else 0.0
    config.timestep_distribution = fixed_distributions.get(timestep_type, distributions[style])
    for source_name, target_name in {
        "do_differential_guidance": "differential_guidance", "differential_guidance_scale": "differential_guidance_scale",
        "do_guidance_loss": "do_guidance_loss",
        "do_guidance_loss_cfg_zero": "do_guidance_loss_cfg_zero", "guidance_loss_schedule": "guidance_loss_schedule",
        "unconditional_prompt": "unconditional_prompt", "do_blank_stabilization": "do_blank_stabilization",
        "diff_output_preservation": "diff_output_preservation", "diff_output_preservation_class": "diff_output_preservation_class",
        "diff_output_preservation_multiplier": "diff_output_preservation_multiplier",
        "blank_prompt_preservation": "diff_output_preservation_blank",
        "blank_prompt_preservation_multiplier": "blank_prompt_preservation_multiplier",
        "pred_scaler": "pred_scaler", "target_noise_multiplier": "target_noise_multiplier",
        "target_norm_std": "target_norm_std", "target_norm_std_value": "target_norm_std_value",
        "t0_loss_target": "t0_loss_target", "t0_velocity_equiv_weight": "t0_velocity_equiv_weight",
        "do_fft_loss": "do_fft_loss", "do_fft_velocity_equiv_weight": "do_fft_velocity_equiv_weight",
    }.items():
        if source_name in train:
            value = train[source_name]
            if source_name == "target_norm_std" and value is None:
                value = False
            expected_type = config.types[target_name]
            if expected_type in (bool, str) and type(value) is not expected_type:
                raise ValueError(f"train.{source_name} must be {expected_type.__name__}")
            if expected_type is float and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f"train.{source_name} must be a finite number")
            setattr(config, target_name, value)
    guidance_target = train.get("guidance_loss_target", 3.0)
    if isinstance(guidance_target, (list, tuple)):
        if len(guidance_target) != 2:
            raise ValueError("train.guidance_loss_target range must contain exactly [min, max]")
        config.guidance_loss_target_min = _positive(guidance_target[0], "train.guidance_loss_target[0]")
        config.guidance_loss_target_max = _positive(guidance_target[1], "train.guidance_loss_target[1]")
    else:
        config.guidance_loss_target = _positive(guidance_target, "train.guidance_loss_target")
    config.diff_output_preservation_trigger = trigger

    datasets = process.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("process.datasets must contain at least one image dataset")
    config.concepts = []
    cache_modes = set()
    for index, dataset in enumerate(datasets):
        path = f"datasets[{index}]"
        _section(dataset, path, {"folder_path", "caption_ext", "caption_dropout_rate", "shuffle_tokens",
                                  "cache_latents_to_disk", "resolution", "flip_x", "flip_y"})
        _fixed(dataset, path, {"caption_ext": "txt", "flip_y": False})
        if not isinstance(dataset.get("folder_path"), str) or not dataset["folder_path"]:
            raise ValueError(f"{path}.folder_path must be nonempty")
        for field in ("shuffle_tokens", "cache_latents_to_disk", "flip_x"):
            if field in dataset and type(dataset[field]) is not bool:
                raise ValueError(f"{path}.{field} must be bool")
        concept = ConceptConfig.default_values()
        concept.name = f"Imported dataset {index + 1}"
        concept.path = dataset["folder_path"]
        concept.seed = 0
        concept.include_subdirectories = True
        concept.text.keep_tags_count = 0
        concept.text.enable_tag_shuffling = dataset.get("shuffle_tokens", False)
        dropout = dataset.get("caption_dropout_rate", 0.0)
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError(f"{path}.caption_dropout_rate must be between zero and one")
        concept.text.tag_dropout_enable = dropout > 0
        concept.text.tag_dropout_probability = dropout
        concept.text.tag_dropout_mode = "FULL"
        concept.image.enable_random_flip = dataset.get("flip_x", False)
        resolutions = dataset.get("resolution", [512])
        if isinstance(resolutions, int):
            resolutions = [resolutions]
        if not isinstance(resolutions, list) or not resolutions:
            raise ValueError(f"{path}.resolution must be a positive integer or list")
        for resolution in resolutions:
            _positive(resolution, f"{path}.resolution", integer=True)
        concept.image.enable_resolution_override = True
        concept.image.resolution_override = ','.join(str(r) for r in resolutions)
        config.concepts.append(concept)
        cache_modes.add(dataset.get("cache_latents_to_disk", False))
    if len(cache_modes) != 1:
        raise ValueError("Mixed dataset cache_latents_to_disk modes are unsupported")
    config.latent_caching = cache_modes.pop()
    config.resolution = config.concepts[0].image.resolution_override

    save = _section(process.get("save", {}), "save", {"dtype", "save_every", "max_step_saves_to_keep", "save_format", "push_to_hub"})
    _fixed(save, "save", {"save_format": "diffusers", "push_to_hub": False})
    config.output_model_format = ModelFormat.DIFFUSERS_LORA
    config.output_dtype = _dtype(save.get("dtype", train.get("dtype", "fp32")), "save.dtype")
    config.save_every = save.get("save_every", 0)
    if isinstance(config.save_every, bool) or not isinstance(config.save_every, int) or config.save_every < 0:
        raise ValueError("save.save_every must be a nonnegative integer")
    config.save_every_unit = TimeUnit.STEP if config.save_every else TimeUnit.NEVER
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(job.get("name", "imported"))).strip('.') or "imported"
    config.workspace_dir = f"workspace/{name}-aitoolkit-parity"
    config.cache_dir = f"workspace-cache/{name}-aitoolkit-parity"
    config.output_model_destination = f"models/{name}-aitoolkit-parity.safetensors"
    sample = _section(process.get("sample", {}), "sample", {
        "sampler", "sample_every", "sample_start_step", "width", "height", "neg", "seed", "walk_seed",
        "guidance_scale", "sample_steps", "samples",
    })
    _fixed(sample, "sample", {"sampler": "flowmatch", "walk_seed": False, "sample_start_step": 0})
    config.sample_after = sample.get("sample_every", 0)
    if isinstance(config.sample_after, bool) or not isinstance(config.sample_after, int) or config.sample_after < 0:
        raise ValueError("sample.sample_every must be a nonnegative integer")
    config.sample_after_unit = TimeUnit.STEP if config.sample_after else TimeUnit.NEVER
    config.sample_skip_first = config.sample_after if train.get("skip_first_sample", False) else 0
    config.samples = []
    for index, item in enumerate(sample.get("samples", [])):
        _section(item, f"sample.samples[{index}]", {"prompt"})
        if not isinstance(item.get("prompt"), str):
            raise ValueError(f"sample.samples[{index}].prompt must be a string")
        preview = SampleConfig.default_values(ModelType.Z_IMAGE)
        # ai-toolkit resolves preview placeholders without automatically prepending the trigger.
        preview.prompt = item["prompt"].replace("[name]", trigger).replace("[trigger]", trigger)
        preview.negative_prompt = sample.get("neg", "")
        preview.width = _positive(sample.get("width", 512), "sample.width", integer=True)
        preview.height = _positive(sample.get("height", 512), "sample.height", integer=True)
        preview.seed = sample.get("seed", 0)
        preview.cfg_scale = _positive(sample.get("guidance_scale", 1.0), "sample.guidance_scale")
        preview.diffusion_steps = _positive(sample.get("sample_steps", 20), "sample.sample_steps", integer=True)
        preview.noise_scheduler = NoiseScheduler.EULER
        config.samples.append(preview)
    config.validate()
    return config
