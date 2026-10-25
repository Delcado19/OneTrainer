# ai-toolkit upstream re-verification - 2026-10-25

Scope: does `ostris/ai-toolkit` moving from `b36bb39` (the pinned reference for
`docs/AiToolkitParity.md` and the two dated audits) to `73ebae1` change any
behavior this branch claims parity with? This is a re-check of an existing
claim, not a new audit; the dated 2026-10-14 and 2026-10-18 records are left
as originally written.

ai-toolkit advanced ~190 commits over six weeks, almost all new model
families (Qwen Image 2.1, Moss Image, Flux3, Fibo2, Iris-3B, Ming-Image,
yue2 audio), an inference-engine server, and UI work. None of that touches
Z-Image Turbo.

Checked every file this branch's `notes/aitoolkit_*_inventory.md` and source
headers cite as a provenance source, by diffing `b36bb39..origin/main` in the
local ai-toolkit checkout:

- `extensions_built_in/diffusion_models/z_image/z_image.py` — unchanged.
- `toolkit/timestep_weighing/default_weighing_scheme.py` — the existing
  1000-entry `default_weighing_scheme` table (our `AI_TOOLKIT_WEIGHTED`
  source) is byte-identical; the diff only adds a separate `x0_weighing_scheme`
  table for x-prediction models (HiDream-style), selected via a new
  `x0_pred` argument. Z-Image is flow-matching (`x0_pred=False`), so
  `get_weights_for_timesteps` still resolves to the same table for it.
- `toolkit/config_modules.py`, `jobs/process/BaseSDTrainProcess.py`,
  `extensions_built_in/sd_trainer/SDTrainer.py`, `toolkit/optimizer.py`,
  `toolkit/samplers/custom_flowmatch_sampler.py` — all have real diffs, but
  every changed line is for audio/video/LLM models, new optimizers
  (automagic4, adamconvrot), a new `band` loss type, EMA feedback rate, a
  `first_timestep_chance` knob, or a gradient-accumulation loss-scaling fix.
  None of it is reachable by the Screen H/I/J recipes (`gradient_accumulation_steps: 1`,
  `batch_size: 1`, `timestep_type: weighted`, `content_or_style` left at
  `balanced`/`style`): the `content_or_style` branch and the Differential
  Guidance / Assistant-LoRA / DOP code paths this branch ported are
  untouched in substance.

Conclusion: no OneTrainer change is needed as a result of this upstream
advance. The local ai-toolkit checkout was fast-forwarded to `73ebae1`
(2026-10-25); `docs/AiToolkitParity.md`'s reference line now notes both
commits.
