# ai-toolkit compatibility validation - 2026-10-18

Scope: `feature/aitoolkit-lora-parity`, local ai-toolkit reference
`b36bb3998ae596a566d85513299696a3a78f0dcb`. This supersedes the implementation
status in the historical 2026-10-14 Screen I audit, not its historical run evidence.

The implementation review identified and fixed:

- Module dropout detached the last trainable path on a frozen base. Dropped
  contributions now keep zero gradients and support backward for LoRA, LoCON and
  full-rank adapters.
- Scheduler resume inferred updates from microsteps and the current accumulation
  setting. It now uses the persisted successful-update count, with a legacy
  fallback for backups that do not contain that count.
- Epoch estimates could terminate imported recipes before their requested update
  budget. Explicit update budgets now drive the training loop and scheduler;
  empty datasets fail instead of retrying indefinitely.
- Class-preservation imports could never supply a trigger. The importer now maps
  the trigger and Z-Image caption injection, retaining full-caption dropout and
  preview placeholder semantics.

Validation command:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Result: 66 tests passed. The regression tests include the real CPU training loop
with one microstep per epoch, accumulation four and three successful updates;
scheduler reconstruction with changed accumulation/skipped-update counts; frozen
base backward after dropout; trigger import/serialization; and actual caption
augmentation ordering/full-caption dropout. Existing tests cover source timestep
tables, loss gradients, 150-module selection, guidance/preservation and synthetic
adapter save/load and optimizer resume.

Ruff 0.16.3 passed for all changed and newly added Python files. `git diff --check`
passed. The existing historical design note is retained as originally written.

No full Z-Image model was loaded. Real GPU AMP overflow, distributed training,
full-model resume/export reload and controlled image-quality parity remain
unverified. The source reference is newer than historical ai-toolkit 0.13.5.
