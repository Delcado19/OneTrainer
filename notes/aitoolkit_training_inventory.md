# ai-toolkit training feature inventory

Source review: 2026-10-14, local ai-toolkit checkout. Scope: ordinary diffusion LoRA training controls and adapter lifecycle. This is a source contract inventory, not a claim of complete numerical or runtime parity.

## Implemented Z-Image additions

`BaseZImageSetup.predict` now supports differential output preservation (DOP), blank prompt preservation, guidance targets, and blank stabilization. DOP replaces the literal trigger with the configured class term in the original caption, encodes that caption independently of cached original embeddings, predicts a frozen-base target with trainable adapters disabled, and adds a separate student MSE. Both passes use the exact main noisy latents and timestep. Teacher inference temporarily uses evaluation mode; hooks and mode are restored on failure. Blank preservation uses empty captions with its own multiplier. Auxiliary preservation MSE is added after the primary weighted/masked loss, as in ai-toolkit.

Guidance builds a detached contrastive target from the current network's unconditional prediction. Supported: scalar guidance target or per-example target sampled from a configured min/max range, CFG-Zero projection, constant or sigma schedule, configurable unconditional prompt. `aitoolkit_guidance_semantics` nests differential guidance under guidance loss, matching ai-toolkit; its default preserves historical OneTrainer standalone differential configs. Blank stabilization zeroes latent content for empty original captions before noise mixing; it is distinct from blank preservation.

Additional Z-Image objective controls match source formulas: `pred_scaler`, `target_noise_multiplier`, spatial `target_norm_std`/`target_norm_std_value`, `t0_loss_target` with optional `t0_velocity_equiv_weight`, and `do_fft_loss` with optional `do_fft_velocity_equiv_weight`. Reconstruction uses the scaled clean latent target; FFT compares orthonormal real-FFT magnitudes. Both velocity weights clamp sigma to at least 0.1 before taking inverse square. Loss weighting must consume the returned per-example `element_loss_scale` for the reconstruction objective.

`aitoolkit_timestep_semantics` uses the actual noise timestep (OneTrainer's zero-based index plus one) consistently for all transformer passes, guidance sigma decay, and reconstruction sigma. The default retains existing OneTrainer timestep conditioning. Loss weighting continues to receive the original zero-based schedule index.

The `ai-toolkit no-FF` layer preset includes block attention and `adaLN_modulation`, excludes feed-forward layers and refiners, and selects the 150-module layout of the Screen I export. The existing attention-only preset selects 120 modules. A synthetic 30-block test verifies the real LoRA wrapper selection. DOP inside GenericTrainer's prior-prediction pass preserves the outer disabled hook state; it still incurs redundant auxiliary forwards during that prior pass, a performance cost rather than a target change.

Alternate captions are encoded live with a frozen text encoder, including when original caption embeddings are cached. The encoder is materialized for encoding and evicted afterwards when latent caching is enabled. This is functional support, not ai-toolkit's complete alternate-embedding cache optimization. Z-Image only; wider architecture support requires model-specific execution paths.

Source: `ai-toolkit/extensions_built_in/sd_trainer/SDTrainer.py:757-854,1246-1358,1807-1818,2021-2046,2228-2248`; `ai-toolkit/jobs/process/BaseSDTrainProcess.py:1376-1383`.

## Execution contracts and remaining inventory

| Feature | Actual source execution | Boundary / additional work |
|---|---|---|
| DOP / blank preservation | `SDTrainer.py:2228-2248` | Extra student/teacher conditioned pass. OneTrainer PRIOR_PREDICTION replaces designated concept targets and is not an equivalent implementation. |
| Guidance loss | `SDTrainer.py:757-854,1989-1999` | Scalar and sampled min/max targets implemented for Z-Image. Joint audio guidance is outside Z-Image. |
| Flow weighting | `SDTrainer.py:946-967`, `custom_flowmatch_sampler.py:32-76` | Bell, half-bell, default table use scheduler-position lookup. Verify separate newly implemented weighting controls against exact source schedules. |
| Timestep bias | `BaseSDTrainProcess.py:1227-1297` | Content/style/balanced bias and separate regularization bias; combine correctly with the chosen schedule. |
| Timestep modes | `custom_flowmatch_sampler.py:116-200+`, `BaseSDTrainProcess.py:1231-1249` | Discrete 1/2/4/8 steps, next_sample, lognorm_blend, inference-derived shifts. next_sample requires its prediction process, not an enum alias. |
| Alternative losses | `SDTrainer.py:927-940,2183` | Pseudo-Huber c=.01, wavelet, stepped x10, mean-flow; ordinary Huber is not pseudo-Huber. |
| t0 / FFT | `SDTrainer.py:892-925` | Implemented for Z-Image; other architectures remain outside this path. |
| Feature extractor | `SDTrainer.py:396,690-753` | Frozen external extractor, version-specific losses and initialization. |
| Learnable SNR | `BaseSDTrainProcess.py:901,1873`, `SDTrainer.py:1037` | Learned gamma/offset/scale, optimizer state and resumption. |
| Prediction / norm controls | `SDTrainer.py:563-585,970-989,1064-1067` | Prediction/target-noise scales and norm-std implemented for Z-Image. Frozen-base norm correction and negative prior divergence remain. |
| Noise controls | `BaseSDTrainProcess.py:965-1034,1330-1376` | Optimal candidate pairing, deterministic per-image/size noise, blended blur, signal/batch correction, random shifts/scales. |
| Training CFG | `SDTrainer.py:1435,1724+,2075-2095` | Unconditional branch, random CFG/rescale, embedding bypass. Transformer guidance embedding alone does not implement it. |
| Captions / input | `BaseSDTrainProcess.py:1044-1110,1372,1435`, `SDTrainer.py:1599` | Short+long caption duplication, SDXL encoder split, prompt repetition, contrast/standardization, adaptive VAE scale, noisy latent scale. Dynamic controls must survive caching. |
| Adapter lifecycle | `stable_diffusion_model.py:732-771,1116-1131,1221-1234,1817-1830` | Assistant premerge/inverse for Flux; inference-only adapter active only in sampling; broader loaders/model contracts required. |
| Parameter swapping | `BaseSDTrainProcess.py:783,2153,2518` | Rotating trainable subsets and optimizer support; distinct from offloading. |

`TrainConfig.weight_jitter` is not consumed by ordinary SDTrainer execution; active occurrences are slider-specific. Exposing that setting without its special trainer would be misleading.

## Verification boundary

Focused synthetic tests exercise real Z-Image setup orchestration: same teacher/student noisy inputs, alternate caption encoding despite cached original embeddings, detached targets, trainable adapter gradients, blank stabilization, guidance formulas, and hook restoration after teacher failure. Real checkpoint/GPU execution, quantized/offloaded full training, saved adapter reload and downstream render parity remain necessary before claiming 1:1 training support.
