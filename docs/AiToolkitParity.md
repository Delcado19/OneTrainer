# ai-toolkit LoRA compatibility

This branch ports concrete ai-toolkit behaviors into OneTrainer. It is not a claim
that every ai-toolkit trainer, model architecture or optimizer trajectory is now
reproduced. The reference is the adjacent local ai-toolkit checkout at commit
`b36bb3998ae596a566d85513299696a3a78f0dcb`; archived Screen I used version 0.13.5.

## Available adapter controls

The LoRA tab exposes rank dropout, module dropout, separate convolution rank and
alpha, substring inclusion/exclusion, a minimum module parameter count, full-rank
adapters, LoCON with optional CP convolution, and mixed full-weight deltas.
Existing defaults preserve older OneTrainer recipes.

- Rank dropout omits rank channels per example and shares the mask over tokens or
  pixels. Module dropout omits an entire adapter invocation without rescaling.
  Its zero contribution retains autograd so a frozen base with every adapter
  dropped can still complete backward.
- An unset convolution rank/alpha inherits the main values. Rank zero excludes
  convolution kernels larger than 1x1; 1x1 kernels use the linear rank/alpha.
- Include filters are additional to the native layer filter. Exclusions win.
- Full-rank initializes a dense down projection with identity up, including the
  source's nonzero initial contribution. It is not a zero-initialized low-rank LoRA.
- LoCON starts with a zero trainable scalar and initialized down/up projections.
  Internal backups retain the scalar for optimizer resume; external exports fold
  it into the up weight. CP adds a middle convolution.
- Full-delta modules store `.diff` and optional `.diff_b`. The base parameters are
  restored after each forward, including failures. Quantized full-delta weights
  are rejected explicitly.

Full-rank, CP-LoCON and mixed full deltas require a native adapter output format;
they are not ordinary HF PEFT adapters. External consumer compatibility still
requires checking the chosen model and format. Rank/module dropout is currently
restricted to plain LoRA/LoCON; other PEFT algorithms have different semantics.

## Available timestep and objective controls

The training UI exposes:

- `AI_TOOLKIT_WEIGHTED`: the exact 1000-entry empirical loss table, with source
  license and revision retained in the module.
- `AI_TOOLKIT_BELL` and `AI_TOOLKIT_HALF_BELL`: source mean-normalized tables.
- `CONTENT_BIASED`: high-noise cubic sampling, alongside existing `STYLE_BIASED`.
- `AI_TOOLKIT_ONE/TWO/FOUR/EIGHT_STEP`: fixed sigma supports. As upstream, these
  bypass the noise window and shift and require a 1000-step schedule. OneTrainer
  uses its Torch generator, not ai-toolkit's Python random sequence.
- Maximum optimizer updates, including resumed updates and accumulation. Zero
  keeps epoch-driven stopping. The LR scheduler uses the effective step budget.
  A nonzero budget takes precedence over the epoch estimate and counts only
  successful updates, so short datasets and skipped AMP updates do not stop it
  early. Empty datasets fail explicitly. Scheduler resume uses that same saved
  update counter; legacy backups without it retain the microstep-based fallback
  only when the exact update budget is disabled.
- Pseudo-Huber strength, with source constant `c=0.01`. Set other strengths to zero
  when selecting a pure pseudo-Huber objective.

Z-Image additionally supports contrastive guidance loss (scalar or sampled target
range), CFG-Zero projection, constant/sigma schedules, an unconditional prompt,
class or blank output preservation, blank stabilization, prediction/noise target
scales, spatial target-standard-deviation loss, t0 reconstruction and FFT magnitude
loss, including source inverse-sigma weights.

Output preservation is an additional student/teacher MSE, not a
`PRIOR_PREDICTION` concept replacing the ordinary flow target. It uses the same
noisy input and timestep and freshly encoded alternate captions. The teacher
disables user adapters without removing the merged assistant adapter. A frozen
text encoder is required. Class and blank preservation are mutually exclusive.

Enable **ai-toolkit Guidance Semantics** to reproduce the source's nesting:
Differential Guidance only runs when Guidance Loss is enabled. The default keeps
older OneTrainer standalone Differential Guidance recipes compatible. Enable
**ai-toolkit Timestep Semantics** to condition the transformer on the actual
noise-sigma timestep instead of the older zero-based conditioning index.

The **ai-toolkit no-FF** layer preset selects attention and adaLN modulation in the
main blocks. This is the actual 150-module Screen I layout. The original
attention-only preset selects 120 modules and is intentionally preserved.

## Import supported recipes

From the OneTrainer directory:

```powershell
.\venv\Scripts\python.exe scripts/import_aitoolkit_config.py source.yaml training_configs/imported.json
```

Load the new JSON in OneTrainer. The importer embeds native concepts and samples,
does not load models, does not start training, does not read secrets, and refuses
to overwrite either source or destination. It maps a supported Z-Image image-LoRA
subset, including actual targeting, optimizer epsilon, weight-only FP8 and the
guidance/timestep compatibility flags. Unsupported execution options fail with
their names rather than silently disappearing. Review the printed limitations
and output paths before running.

`process.trigger_word` maps to `aitoolkit_trigger_word` and the class-preservation
trigger. Z-Image training captions replace `[name]`/`[trigger]` and prepend a
missing literal trigger after tag shuffling, before caption dropout. A dropped
caption therefore stays empty. Prior-prediction concepts do not automatically
receive the trigger. Preview prompts replace placeholders without prepending it.
Existing recipes keep caption processing unchanged when the trigger is unset.

For the latest local checks and their limits, see the
[validation record](../notes/aitoolkit_validation_2026-10-18.md).

## Remaining work for comprehensive 1:1 support

The complete source inventories are [network options](../notes/aitoolkit_network_inventory.md)
and [training options](../notes/aitoolkit_training_inventory.md). Open work includes:

- LoRM extraction/replacement/reconstruction training and model-specific hybrid
  full input/output training.
- Per-module heterogeneous rank/alpha inference, exact DoRA/LoKr initialization
  and dropout, and model-specific multistage adapter exports.
- Guidance/preservation and assistant/inference adapters on architectures beyond
  the implemented Z-Image path.
- Mean-flow/JVP, next-sample, wavelet/stepped and feature-extractor objectives;
  learned SNR and the remaining noise/norm/parameter-swapping controls.
- Matching image crops/bucket exposure, latent posterior sampling, caption/cache
  variation frequency, random streams and quantizer implementations.
- Full model GPU training, resume, exported checkpoint reload and downstream
  evaluation on identical prompts, seeds, sampling settings and checkpoints.

Unit tests and layer probes establish the tested contracts. They do not establish
image-quality parity or whole-framework 1:1 training behavior. Historical Screen I
itself failed its trigger-only material gate, even in ai-toolkit; engineering
parity and the best material-training recipe are separate acceptance criteria.
