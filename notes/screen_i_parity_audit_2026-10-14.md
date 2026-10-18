# Screen I parity audit

Date: 2026-10-14. Review only; no training code or run configuration changed.

This records the pre-implementation review. Subsequent implementation is tracked
in `docs/AiToolkitParity.md` and the network/training inventories. An additional
source finding corrected the original formula-only comparison: in current
ai-toolkit, Differential Guidance is nested inside Guidance Loss. Matching the
formula alone does not prove it executes for a given recipe.

## Evidence and scope

- OneTrainer branch: `z-image-assistant-lora-differential-guidance`, HEAD `fbef15ba`.
- ai-toolkit HEAD: `b36bb39`. Its current source is newer than the archived run; the exported ai-toolkit LoRA identifies version `0.13.5`.
- Actual experiment files are under `D:/LoRA-Training`, including `CURRENT.md`, `configs/d19latex_screen_i.onetrainer_config.json`, the archived ai-toolkit `AIT_Output/d19latex_screen_i/config.yaml`, run logs, saved OneTrainer run configs, exports and evaluations.
- All seven archived OneTrainer run configs use `FLOAT_W8A8`; the current importable config uses `FLOAT_8`. No archived weight-only rerun was found in this workspace.
- The latest OneTrainer log completes nine epochs of 42 batches and saves the final model. That is 378 updates, not a run ending at step 285. Step 285 is the last scheduled preview; intermediate save filenames use the preceding zero-based step.
- All 11 repository unit tests passed. Config loading through `TrainConfig` and validation passed. This audit did not launch training or inference.

## Confirmed missing or different behavior

### 1. The trained module set is different

Inspection of the actual final safetensors files is decisive:

- ai-toolkit: 300 tensors, 150 A/B module pairs.
- OneTrainer: 240 tensors, 120 A/B module pairs.
- The 30 missing modules are `layers.<0..29>.adaLN_modulation.0`.
- Both exports contain the same 120 attention projection modules. Neither export contains refiner or feed-forward adapters.

Therefore the historical description of Screen H/I as strictly "attention-only" is inaccurate: the actual ai-toolkit experiment also trains adaLN modulation. OneTrainer's `^(?=.*attention)(?!.*refiner).*` filter does not reproduce it. Match the archived 150-module inventory rather than trusting the preset label. A suitable scoped regex for this inventory is `^layers\.\d+\.(attention\.|adaLN_modulation\.)`; verify the resulting module set before training.

### 2. Weighted loss was not ported

ai-toolkit's archived recipe sets `timestep_type: weighted`. In current source, `SDTrainer.calculate_loss` multiplies per-element MSE by the scheduler's `get_weights_for_timesteps`, using `toolkit/timestep_weighing/default_weighing_scheme.py`. OneTrainer's imported recipe uses `loss_weight_fn: CONSTANT`, and its flow loss currently offers only constant or sigma weighting. Neither reproduces this table.

The table has 1000 entries ranging from approximately 0.445 to 1.523. At noise timestep 1 its weight is approximately 0.447; at timestep 900 it is approximately 1.507. On a deterministic uniform grid passed through Screen I's cubic index selection, the mean weight is approximately 0.743. This alters both the reported loss and gradients, not just logging. It does not alone explain a twentyfold historical loss difference.

Missing implementation: an explicit corresponding flow-loss weighting mode, correct descending-index lookup, and tensor/gradient comparison tests. Preserve the source table and its provenance; do not replace it with an arbitrary sigma approximation. Establish the table used by the historical ai-toolkit revision before claiming historical numerical parity.

### 3. Quantization correction exists but has no demonstrated rerun

Archived OneTrainer runs use W8A8 math, including quantized activations and backward propagation. The current recipe already changes transformer and text encoder to `FLOAT_8`, whose `LinearFp8` dequantizes weights for `F.linear` without activation quantization. This is structurally closer to ai-toolkit's current weight-only float8 path.

Missing evidence: a run with this corrected configuration. It is a plausible explanation for progressive damage, not a proven root cause. Quantization implementations and module footprints still differ: OneTrainer includes only `layers`; ai-toolkit also processes extras subject to its sensitive-module exclusions. Match footprints for a strict reproduction, after isolating the weight-only correction if testing causality.

### 4. Caption dropout is not equivalent

ai-toolkit returns an empty caption on its 5% caption-dropout event. The OneTrainer concept uses `FULL` tag dropout with probability 0.05 but `keep_tags_count: 1`. MGDS `DropTags` always preserves those leading tags, including on full dropout.

For whole-caption dropout, use `keep_tags_count: 0` in this concept and verify the actual cached/tokenized outputs. Both caching and variation frequency need inspection in an end-to-end data probe; matching a probability field alone is insufficient.

### 5. Optimizer and budget differ

- ai-toolkit: bitsandbytes AdamW8bit, explicit `eps=1e-6` in its optimizer factory, LR 1e-4, weight decay 1e-4, 380 updates.
- OneTrainer: PyTorch AdamW, `eps=1e-8`, same nominal LR/weight decay, nine epochs with 42 batches = 378 updates.

OneTrainer already supports `ADAMW_8BIT`; an optimizer implementation is not missing. Matching optimizer choice/epsilon is configuration work. Exact 380-update stopping needs an explicit bounded step mechanism or a reproducible runner because this recipe is epoch-driven. The two-update difference is much weaker evidence than the module/loss/quantization differences.

### 6. Timesteps and latent preparation need a numerical parity probe

The cubic low-noise bias is correctly oriented. However, OneTrainer indexes noise sigma as `(index + 1)/1000` while transformer conditioning uses `(1000 - index)/1000`. Its own noise and conditioning therefore differ by one grid step. ai-toolkit uses the same actual scheduler timestep for noise mixing and conditioning. Shared cubic random values also expose endpoint/rounding differences of up to one sigma grid step (0.001).

OneTrainer's Z-Image loader uses VAE posterior mean; ai-toolkit's current `encode_images` uses a posterior sample. Same resolution labels also do not establish identical crop/bucket assignments or image exposure. Verify image path, actual crop size, caption/token IDs, latent, noise, sigma, prediction, guided target, weighted loss and gradients using fixed inputs. Do not attribute major quality failure to the small timestep offset without measurement.

## Implemented and supported by current checks

- Assistant merge before final quantization, inverse sampling context and exception restoration exist.
- Differential Guidance formula and flow prediction sign match the current ai-toolkit source.
- Style-biased timestep sampling and experimental UI exist.
- Assistant is excluded from `model.adapters()` and optimizer parameters in the unit test.
- The inspected final OneTrainer export has only the expected 120 user adapter modules, not extra assistant modules. Shared key names cannot by themselves prove absence of assistant tensor contamination; save/reload behavior still needs a numerical test.

## Additional robustness gaps

- Reproduced: a local assistant file containing only `lora_down` is accepted as merged while the base weights remain unchanged. `strict=False` should permit absent complete modules, not incomplete A/B pairs. Validate required tensor pairs, shapes and finite values before merging.
- Reproduced: `NaN` and positive infinity pass Differential Guidance scale validation. Require a finite positive value.
- Existing tests cover small isolated tensors/layers; none covers the full quantized training-to-sampling-to-export/reload path.

## Required completion sequence

1. Preserve the failed W8A8 baseline and corrected configuration as separate named experiments.
2. Fix adapter validation and finite-scale validation with focused regressions.
3. Add and test the missing weighted flow loss; align the actual 150-module target inventory, full caption dropout and optimizer recipe.
4. Add a fixed-input parity probe for sigma/conditioning, weighted loss and gradients, and a quantized sampling/export/reload check.
5. Run a bounded real-model check with finite loss/gradients and sampling-state restoration, then the controlled training comparison with recorded effective configuration and revisions.
6. Evaluate exported checkpoints in the established ComfyUI target stack: 896x1344, nine steps, CFG 1, AuraFlow shift 3, fixed seeds, trigger-only prompts. Internal 512x512 previews are insufficient.

Screen I itself failed the historical trigger-only material-quality evaluation; Screen H remained the strongest baseline. Reproducing Screen I is an engineering parity goal, not evidence that it is the best material recipe. Do not silently substitute Screen H or Screen J while testing this port.

## Limits

No full model was loaded, no new GPU training/inference was performed, and no numerical framework parity is claimed. Two archived OneTrainer previews (step 0 and step 285, prompt 0) were visually inspected and show clear progressive softening. Historical evaluation reports were read, not independently rerendered. Existing cache-path edits in `GenericTrainer.py` and `SampleWindowController.py` were left untouched. No external publication was performed.
