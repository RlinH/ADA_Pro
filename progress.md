# Progress Log

Running log of every update: the changes made and the result tables produced.
Newest entries are added at the top.

---

## 2026-10-27 — A100 single-seed (seed 0) benchmark run

### Changes
- Set up project code at `DGSEG_BENCHMARK` root (repo `RlinH/ADA_Pro`, code commit `d42e183`).
- Assembled the chunked transfer bundle and verified it (`sha256sum -c` OK, `unzip -t` clean); extracted `DGSEG_DATA/downloads/` and re-verified all 5 source archives against the transfer manifest.
- Downloaded ImageNet `ResNet50_Weights.IMAGENET1K_V2` (backbone pretraining) — SHA-256 verified.
- Trained and evaluated **all 9 runs** (3 tracks × {erm, mixstyle, randconv}, seed 0), 80 epochs each, then ran `scripts/summarize.py`.
- Added `reports/A100_RUN_seed0_2026-10-27.md` (detailed run report).
- Updated `reports/benchmark_summary.csv` (filled `ready_unrun` placeholders with seed-0 results).

### Results — seed 0, native-resolution target evaluation
Fractions in `[0, 1]`. `smoke_only=false`, `publishable=true`. Protocol `dgseg-ssdg-v1.0-2026-10-25`.

#### r19 — Cityscapes → Mapillary Vistas v1.2 (19 classes, 2000 images)
| Method | mIoU | mean Dice | pixel acc |
|---|---:|---:|---:|
| erm | 0.5369 | 0.6778 | 0.8888 |
| mixstyle | 0.5484 | 0.6863 | 0.9077 |
| randconv | **0.5727** | 0.7085 | 0.9056 |

#### rs7_u2r — LoveDA Urban → Rural (7 classes, 992 images)
| Method | mIoU | mean Dice | pixel acc |
|---|---:|---:|---:|
| erm | **0.3422** | 0.4877 | 0.6055 |
| mixstyle | 0.3352 | 0.4769 | 0.6174 |
| randconv | 0.3330 | 0.4705 | 0.6043 |

#### rs7_r2u — LoveDA Rural → Urban (7 classes, 677 images)
| Method | mIoU | mean Dice | pixel acc |
|---|---:|---:|---:|
| erm | **0.4812** | 0.6430 | 0.6371 |
| mixstyle | 0.4546 | 0.6191 | 0.6186 |
| randconv | 0.4596 | 0.6233 | 0.6204 |

### Blocked / not run
- `r19 → BDD100K`, `r19 → ACDC`: `blocked_missing_data`.
- `s16 → SYNTHIA`: `blocked_incomplete_synthia`.

### Notes
- Single-seed (seed 0) → summary status `partial`; protocol requires seeds 0/1/2 for a complete comparison.
- Artifacts: `runs/<track>_<method>_seed0/`, `reports/evaluations/`, `reports/benchmark_summary.csv`.