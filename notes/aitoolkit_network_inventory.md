# ai-toolkit network option inventory

Source scope: local `toolkit/config_modules.py:169`, `toolkit/lora_special.py:293`,
`toolkit/network_mixins.py:197`, `toolkit/models/lokr.py:80`,
`toolkit/lycoris_special.py:24`, and `jobs/process/BaseSDTrainProcess.py:1893`.
This is source evidence, not an end-to-end numerical training parity claim.

| ai-toolkit setting | OneTrainer mapping / status |
| --- | --- |
| `rank`, `linear` | Native `lora_rank`; aliases select the same value. |
| `alpha`, `linear_alpha` | Native `lora_alpha`; see PEFT effective-scale caveat below. |
| `conv`, `conv_alpha` | Added `lora_conv_rank`, `lora_conv_alpha` for kernels larger than 1x1. `None` retains legacy main rank/alpha; rank 0 disables these conv adapters. 1x1 kernels retain linear parameters, as in toolkit. |
| `dropout` | Native `dropout_probability` for plain LoRA. |
| `network_kwargs.rank_dropout` | Added `lora_rank_dropout` for plain LoRA: per-sample rank mask shared over tokens/pixels, scaled by inverse survival probability; inactive during evaluation. |
| `network_kwargs.module_dropout` | Added `lora_module_dropout` for plain LoRA: drop entire adapter contribution per invocation, without survival rescaling; inactive during evaluation. |
| `only_if_contains` | Added comma-separated `lora_include`; conjunctive with native module filters. |
| `ignore_if_contains` | Added comma-separated `lora_exclude`; exclusion takes precedence. |
| `parameter_threshold` | Added `lora_parameter_threshold`; counts weight and bias parameters before wrapping. |
| `transformer_only` | Partially equivalent: model-specific native layer presets/regex; match actual module lists, not preset names. |
| `target_lin_modules`, `target_conv_modules` | Model-specific traversal boundaries; native filters can select leaves but cannot add unsupported models. |
| `type=lora` | Native plain LoRA; same basic down/up architecture and Kaiming-down/zero-up initialization. |
| `type=dora` | Partial: native DoRA, but initialization, default norm axis/epsilon and dropout differ. Toolkit uses normal-distributed down weights, output-row norm, detached norm, and extra output correction. Requires numerical comparison. |
| `type=lokr`, `lokr_factor`, `lokr_full_rank` | Native LoKr, `lokr_decompose_factor`, `lokr_full_matrix` with decomposition disabled. Toolkit forces huge rank/alpha for full rank; initialization and dropout must be compared. Toolkit LoKr rank dropout is different from ordinary LoRA rank dropout. |
| `old_lokr_format` | Partial: native output-format namespaces differ; require export/import round trips for each model. |
| `pretrained_lora_path` | Native `lora_model_name`; resume/export compatibility requires matching rank and module selection. |
| `type=locon` / `lycoris` | Added `lora_locon` and `lora_use_cp`: scalar-zero gating, both projections Kaiming-initialized, optional middle convolution, raw scalar retained in internal backups, scalar folded on external export, reload and merge. CPU next-update Adam resume, actual internal/DIFFUSERS saver-file roundtrip and gradient tests pass; actual external format consumers remain unverified. |
| `type=fullrank` | Added `lora_full_rank`: dense Kaiming-initialized down and identity up, preserving source nonzero initialization and configured alpha/rank scale. Native roundtrip tested; external HF consumers may reject down-only adapters. |
| `all_layers`, `full_if_contains` | Added `lora_all_layers` and `lora_full_include`: zero-initialized `.diff`/`.diff_b` modules for non-linear weighted leaves or selected linear/conv overrides, exception-safe parameter restoration, gradients, actual INTERNAL and COMFY saver-file roundtrips through loader conversion, and merge. Quantized weights and full deltas in fused projection export groups fail explicitly; external ComfyUI application remains unverified. |
| `type=lorm`, `lorm.*` | Missing LoRM model replacement/extraction and per-module extraction modes, thresholds, convolution support and reconstruction training. This changes base-model architecture. |
| `full_train_in_out` | Missing hybrid copied full input/output training, model-specific for UNet, PixArt/AuraFlow and Wan. |
| `modules_dim`, `modules_alpha` | Missing per-module adapter rank/alpha inference and training configuration; scalar global rank is insufficient for heterogeneous imports. |
| `split_multistage_loras` | Wan 2.2-specific export behavior in `wan22_14b_model.py:243,429`; separate implementation needed. |
| `layer_offloading` in NetworkConfig | Source labels this network flag nonfunctional; do not advertise it as supported toolkit behavior. Base-model layer offloading is a separate functional option. |
| `block_dims`, `block_alphas`, `conv_block_dims`, `conv_block_alphas` | Constructor accepts/logs them, but `LoRASpecialNetwork.create_modules` does not apply per-block values. Do not infer supported parity from signature alone. |
| `attn_only` | Accepted but unused by this LoRASpecialNetwork implementation; explicit module inclusion is the real selector. |
| `use_bias` | Base process forces it for LoRM only; arbitrary duplicate kwargs conflict with constructor. Exact LoRM bias path remains missing. |
| `is_ara`, assistant flags | Internal quantization/assistant implementation flags, not standalone user LoRA features. |

## Effective scale caveat

`LoRASpecialNetwork` forces alpha to rank whenever PEFT format is active
(`lora_special.py:427-432`), including transformer models such as Z-Image.
Therefore toolkit configuration `alpha: 1` may have effective adapter scale 1,
while OneTrainer `lora_alpha: 1` with rank 16 has scale 1/16. Compare exported
alpha and actual effective scaling; copying the YAML value is insufficient.

## Verification boundaries

CPU tests cover mixed zero full deltas on Linear/Conv2d/LayerNorm/Embedding, bias/weight gradients, failure restoration, native save/load, quantized rejection, fullrank dense initialization/native roundtrip, LoCON zero gate/gradients, CP convolution/native roundtrip/merge, plain rank masks for linear/token/convolution inputs, gradients,
module dropout/eval behavior, filters, threshold and convolution parameter rules.
They do not prove quantized/offloaded GPU training, compiled stochastic paths,
fused export equivalence, optimizer trajectories or sample quality. New rank and
module dropout are restricted to plain LoRA until each other adapter-specific
algorithm is implemented and tested; rejecting unsupported combinations is
preferable to silently ignoring requested options.
