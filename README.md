# DGSeg SSDG benchmark

This project implements **single-source domain generalization for semantic segmentation** on data already obtained by the user. It reads original ZIP archives directly and keeps raw datasets outside the code directory. The shared architecture is ResNet-50 + DeepLabV3+ (output stride 16). A training run sees one labeled source domain. Only `scripts/evaluate.py` reads target masks. The source validation set alone selects `best_source_val.pt`.

## Current executable scope

| Track | Source train | Source validation | Offline target evaluation | Classes | Status |
|---|---|---|---|---:|---|
| `r19` | Cityscapes train: 2,975 | Cityscapes val: 500 | Mapillary Vistas Research v1.2 validation: 2,000 | 19 | Local smoke passed |
| `rs7_u2r` | LoveDA Train Urban: 1,040 | Deterministic Urban holdout: 116 | LoveDA Val Rural: 992 | 7 | Local smoke passed |
| `rs7_r2u` | LoveDA Train Rural: 1,229 | Deterministic Rural holdout: 137 | LoveDA Val Urban: 677 | 7 | Local smoke passed |
| `s16` | SYNTHIA-RAND-CITYSCAPES | Source-only holdout | Cityscapes/BDD/Mapillary | 16 | Blocked: official RAR partial, 10/200 chunks |

BDD100K semantic segmentation and ACDC are pending official access; no scores are fabricated. The Mapillary v1.2 decision supersedes the original v1.1 preference. The raw archive's README identifies v1.2 even though its `config.json` contains a `version` value of `1.1`. The [66-row explicit map](configs/mappings/mapillary_v1_2_to_cityscapes19.json) ignores labels without an unambiguous Cityscapes-19 counterpart. Keep this mapping frozen before comparing methods.

The Mapillary mapping and fixed 512×1024 single-scale inference define **this** benchmark protocol. Published results using different class maps, image scales, or sliding-window evaluation are not directly comparable.

## Methods

| Executable method | Source | Implementation used here |
|---|---|---|
| `erm` | Standard supervised baseline | Same architecture, optimizer, crop and checkpoint rule as other methods |
| `mixstyle` | [MixStyle, ICLR 2021](https://github.com/KaiyangZhou/mixstyle-release) | Feature statistics mixed after ResNet layer 1 during training; no target data |
| `randconv` | [RandConv, ICLR 2021](https://github.com/wildphoton/RandConv) | Random convolution of normalized input with moment matching; **segmentation adaptation**, not an exact official paper reproduction |

[RobustNet, CVPR 2021](https://github.com/sunghac/RobustNet) and [SHADE, ECCV 2022](https://github.com/HeliosZhao/SHADE) are established segmentation DG comparisons for a later, separately audited official-code reproduction. They are **not** represented as executable results here. SHADE is specifically relevant to synthetic-to-real `s16` once SYNTHIA is complete. The current shared-backbone comparison covers the three executable methods above.

## Local Windows verification

Local system: Python 3.11, PyTorch 2.0.0+cu118, torchvision 0.15.1+cu118, RTX 3060 6 GiB. In a Command Prompt:

```bat
cd /d E:\ADA2\DGSEG_BENCHMARK
set DGSEG_DATA_ROOT=E:\ADA2\DGSEG_DATA
scripts\smoke_local.cmd
```

The smoke mode uses a single optimization step on real data (road 128×256; LoveDA 128×128), one source-validation batch, and one target sample. It tests the full data/model/evaluator path. Every smoke result has `smoke_only=true` and `publishable=false`; the small random-initialization mIoU values are **not benchmark results**. `--native-smoke` also tests restoring logits to a full-resolution Mapillary label image.

The machine-readable [local smoke report](reports/LOCAL_SMOKE_STATUS.json) verifies the real-data optimization runs, target evaluation paths, CUDA device, manifest hashes, and frozen protocol hash. The [dataset audit](reports/DATASET_AUDIT.md) records archive sizes, acquisition status, split counts, and audit limits. An extra pretrained-backbone smoke passed locally using the official `ResNet50_Weights.IMAGENET1K_V2` file (SHA-256 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`); run `scripts\smoke_pretrained.cmd` separately if you want to repeat the weight download test.

The original data is not extracted. Manifests under `reports/manifests/` contain archive-relative paths and SHA-256 file hashes, so moving both project and data root to Linux does not change sample IDs. The LoveDA source holdout is deterministic with seed 42. `reports/DATASET_AUDIT_SAMPLE.md` indexes 20 real RGB/normalized-mask overlays per available role.

The source ZIP excludes the local overlay images and model checkpoints. After moving the licensed archives, run `python3 -m scripts.audit_data --data-root "$DGSEG_DATA_ROOT"` if you want to regenerate the visual audit on the A100 host.

## A100 handoff

For a single ZIP64 transfer archive on this Windows machine, run `python -m scripts.package_a100`. It copies only the five complete archives needed for the three available tracks plus Git-tracked project files, verifies the data hashes while packing, and writes a SHA-256 sidecar. The default destination is `D:\DGSEG_A100_bundle_2026-10-25.zip`; use `--output` to choose another disk with at least 46 GiB free.

1. Copy this project and the licensed dataset archives to storage accessible from the school A100. Preserve the `downloads/` tree under the new data root. Follow each dataset's license for storage and access; do not push raw data to Git.
2. Create a Python 3.11 environment. Install a matching CUDA build of PyTorch and torchvision using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then `python3 -m pip install -r requirements.txt`.
3. Set `DGSEG_DATA_ROOT` to the directory containing `downloads/`. Check `python3 -c 'import torch; print(torch.cuda.is_available())'` prints `True`.
4. From the project root, run one track first, then the others:

```bash
export DGSEG_DATA_ROOT=/path/to/DGSEG_DATA
TRACKS=r19 METHODS='erm mixstyle randconv' SEEDS='0 1 2' bash scripts/run_a100.sh
TRACKS='rs7_u2r rs7_r2u' METHODS='erm mixstyle randconv' SEEDS='0 1 2' bash scripts/run_a100.sh
```

The launcher uses ImageNet-pretrained ResNet-50, 80 epochs, batch 4, 512×1024 road crops or 512×512 LoveDA crops, SGD, polynomial LR, three seeds, and source-validation checkpoint selection. It resumes from each run's `last.pt` after interruption. The target evaluator processes every image, upsamples logits to the **native** GT size, and writes JSON/CSV. `scripts/summarize.py` reports mean/std across completed seeds; it excludes smoke results. Do not inspect target scores to select a checkpoint or change hyperparameters.

Formal evaluation checks the frozen protocol hash, completed 80-epoch `last.pt`, and source-selected `best_source_val.pt`. Changing epochs, batch size, crop size, learning rate, or backbone pretraining requires a new protocol version; incomplete or smoke checkpoints cannot enter `benchmark_summary.csv`.

Metric JSON/CSV stores IoU, Dice, and accuracy as fractions in `[0, 1]`; multiply by 100 only when presenting percentages. The global confusion matrix is accumulated across the dataset before mIoU is computed.

## Seed-0 result audit and input-scale diagnostic

The A100 host has completed all nine seed-0 runs. See [the run report](reports/A100_RUN_seed0_2026-10-27.md), [the baseline audit](reports/BASELINE_AUDIT_2026-10-27.md), and [the summary](reports/benchmark_summary.csv). All three seeds are still required for a completed comparison. The two ERM LoveDA run-status and per-class target JSONs are now in the repository; full checkpoint histories remain on A100.

The evaluator supports a post-hoc, **non-publishable** input-scale check. Use the same completed ERM checkpoint and the same deterministic 100 target samples at both sizes. Each JSON records the input size and selected sample-ID hash; verify that the hashes match in a pair. Run from the repository root on the A100 with `DGSEG_DATA_ROOT` set:

```bash
python3 -m scripts.evaluate --track rs7_u2r --target rural --checkpoint runs/rs7_u2r_erm_seed0/best_source_val.pt --height 512 --width 512 --diagnostic --max-samples 100 --output reports/diagnostics/u2r_erm_512_100.json
python3 -m scripts.evaluate --track rs7_u2r --target rural --checkpoint runs/rs7_u2r_erm_seed0/best_source_val.pt --height 1024 --width 1024 --diagnostic --max-samples 100 --output reports/diagnostics/u2r_erm_1024_100.json
python3 -m scripts.evaluate --track rs7_r2u --target urban --checkpoint runs/rs7_r2u_erm_seed0/best_source_val.pt --height 512 --width 512 --diagnostic --max-samples 100 --output reports/diagnostics/r2u_erm_512_100.json
python3 -m scripts.evaluate --track rs7_r2u --target urban --checkpoint runs/rs7_r2u_erm_seed0/best_source_val.pt --height 1024 --width 1024 --diagnostic --max-samples 100 --output reports/diagnostics/r2u_erm_1024_100.json
```

These runs write JSON with `diagnostic_only=true` and `publishable=false`; they do not write a benchmark CSV. Keep the v1.0 summary unchanged. A change to the formal input size would need a new frozen protocol and new evaluation, and target scores must not be used to select checkpoints.

## Audit boundaries and next work

- The output [summary](reports/benchmark_summary.csv) contains nine seed-0 rows; seeds 1 and 2 remain outstanding.
- `r19` currently has only one available target. The task document's BDD100K and ACDC three-target aggregate must stay blank until those official data are obtained and adapters are audited.
- `s16` stays blocked until the official SYNTHIA file finishes and its raw label IDs and 16-class protocol are audited. Its preallocated partial RAR is not usable data.
- Full training and native-resolution 2,000-image Mapillary evaluation ran on the A100 host. A 6 GiB RTX 3060 on this computer is used for short engineering smoke only.
