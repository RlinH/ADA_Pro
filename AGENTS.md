# AGENTS.md

## Project
DGSeg single-source domain-generalization (SSDG) semantic-segmentation benchmark.
Code root = this directory (`DGSEG_BENCHMARK`). Datasets live outside the code dir and are read
directly from their original ZIP archives; set `DGSEG_DATA_ROOT` to the directory containing `downloads/`.

## Logging / update convention (required)
After **every** update (a run, a fix, a data change, etc.):

1. Append a new entry at the **top** of `progress.md` (repository root) containing:
   - a short "Changes" list of what was modified/done;
   - the result tables produced by that update.
2. Commit and push to GitHub (`RlinH/ADA_Pro`, branch `main`).

## Git hygiene
- Never commit raw data or large artifacts. `runs/`, `reports/evaluations/`, `*.zip`, `*.pt`, `*.pth` are gitignored.
- Keep commits focused; write a concise message.