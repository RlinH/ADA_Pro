# Progress Log

Running log of every update: the changes made and the result tables produced.
Newest entries are added at the top.

---

## 2026-10-27 — Verify top-conference comparison scales

### Changes
- Added `reports/LITERATURE_COMPARISON_2026-10-27.md` with checked primary-paper tables, method descriptions, source/target split limitations, and an official-code reproduction order.
- Linked the review from `README.md`. No training, evaluator, data, or frozen-protocol results changed.

### Results — literature and current Mapillary mIoU (%)

| Cityscapes-only labeled source | Backbone / method | Mapillary mIoU | Note |
|---|---|---:|---|
| This repository, seed 0 | R50 DeepLabV3+ ERM | 53.69 | Mapillary v1.2 common evaluator |
| This repository, seed 0 | R50 DeepLabV3+ RandConv adaptation | 57.27 | Same local protocol |
| RobustNet CVPR 2021 | R50 baseline / ISW | 51.68 / 58.64 | Published evaluator; map and inference may differ |
| WildNet CVPR 2022 | R50 WildNet | 58.79 | Uses extra unlabeled ImageNet images |
| SiamDoGe ECCV 2022 | R50 SiamDoGe | 59.00 | Source-only method; published evaluator |
| Rein CVPR 2024 | DINOv2-Large + Rein | 72.3 | Different backbone/pretraining tier |

LoveDA NeurIPS 2021 reports source-only **31.74** (Urban → Rural) and **31.27** (Rural → Urban) on its **Test** split with DeepLabV2-R50; our Val/DeepLabV3+ numbers are 34.22/48.12 and are not matched reproductions. BDD and ACDC remain unavailable for local evaluation.

---

## 2026-10-27 — Audit low baseline scores and add scale diagnostic

### Changes
- Reviewed GitHub's seed-0 summary and the newly uploaded LoveDA ERM run-status/evaluation JSONs against training, label mapping, and global-confusion metric code.
- Added `reports/BASELINE_AUDIT_2026-10-27.md` with literature-scale references, source/target gaps, and the Urban → Rural class confusion pattern.
- Added a non-publishable `--diagnostic` evaluator mode for paired 512/1024 target-input checks with a completed checkpoint and identical deterministic sample IDs. Updated README instructions; kept the v1.0 training/evaluation protocol and existing benchmark scores unchanged.
- Verified Python compilation, four unit tests, and a local native-resolution smoke evaluation.

### Results — ERM seed 0, mIoU (%)

| Track | Best source validation | Target | Gap (target − source) | Target weakest class IoU |
|---|---:|---:|---:|---:|
| LoveDA Urban → Rural | 64.67 | 34.22 | −30.44 | barren 6.54 |
| LoveDA Rural → Urban | 65.34 | 48.12 | −17.22 | barren 34.50 |

Source validation uses resized 512×512 masks; target evaluation uses native 1024×1024 masks, so the gap includes an evaluation-scale difference. The paired diagnostic has not yet been run on A100 and contributes no benchmark result.

---

## 2026-10-27 — Add ERM LoveDA run-status / evaluation JSONs to the repo

### Changes
- Uploaded the small JSON artifacts for the two ERM LoveDA runs (no weights, no raw data):
  - `runs/rs7_u2r_erm_seed0/run_status.json`
  - `runs/rs7_r2u_erm_seed0/run_status.json`
  - `reports/evaluations/rs7_u2r_erm_seed0_rural.json`
  - `reports/evaluations/rs7_r2u_erm_seed0_urban.json`
- Force-added them because `runs/` and `reports/evaluations/` are gitignored; only these specific files are tracked.

### Result snapshots (ERM, seed 0)
| Track | Target | source_val mIoU (best) | Target mIoU | Target pixel acc |
|---|---|---:|---:|---:|
| rs7_u2r | rural | 0.6467 | 0.3422 | 0.6055 |
| rs7_r2u | urban | 0.6534 | 0.4812 | 0.6371 |

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
