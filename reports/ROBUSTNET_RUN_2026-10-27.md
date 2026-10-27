# RobustNet (CVPR 2021) reproduction on A100 — baseline vs ISW

- **Date:** 2026-10-27 (UTC)
- **Host / GPU:** 1 × NVIDIA A100 80GB PCIe (`cuda:0`)
- **Task:** r19 — Cityscapes (source) → Mapillary Vistas v1.2 (target)
- **Upstream:** [sunghac/RobustNet](https://github.com/sunghac/RobustNet) pinned at `1ff5c047d7abaf9ab1a7cca23aff812b2f09de15`
- **Status:** full run complete for both methods (`FULL_EXIT=0`); results `publishable=true`

## Software
| Item | Value |
|---|---|
| Python | 3.11.6 |
| PyTorch | 2.1.0+cu118 |
| torchvision | 0.16.0+cu118 |
| numpy / scikit-image | 1.26.4 / 0.22.0 |
| tensorboardX / thop / kmeans1d | 2.6.2.2 / 0.1.1 / 0.5.0 |

## Compatibility patches (external checkout only; method math and data protocol unchanged)
1. `RobustNet/transforms/transforms.py`: `gaussian(..., multichannel=True)` and `denoise_bilateral(..., multichannel=True)` → `channel_axis=-1` (scikit-image ≥ 0.19 removed `multichannel`).
2. `RobustNet/train.py`: `vutils.make_grid(..., range=(0,1))` → `value_range=(0,1)` (newer torchvision renamed `range`).

No changes to the network, loss, whitening math, augmentations, or data protocol.

## Protocol / source-only controls
- Official settings: 40,000 iterations, 768 crop, class-uniform sampling (0.5 / tile 1024), `--lr 0.01` poly 0.9, `--scale 0.5 2.0`, `--gblur --color_aug 0.5`.
- **Effective batch size = 4 on one GPU.** The paper used two GPUs (global batch 8); this is a one-GPU reproduction with half the global batch — disclosed here.
- Target domains never used for training, source validation, or checkpoint selection: `--val_dataset` empty (target list), covariance statistics (`--covstat_val_dataset cityscapes`, `--cov_stat_epoch 5` for ISW) use Cityscapes training data only.
- Checkpoint selection: highest Cityscapes-val mIoU (`best_cityscapes_*.pth`).
- ISW config: `--wt_reg_weight 0.6 --relax_denom 0.0 --clusters 2 --cov_stat_epoch 5 --trials 10 --wt_layer 0 0 2 2 2 0 0`.
- Baseline config: `--wt_reg_weight 0 --cov_stat_epoch 0 --wt_layer 0 0 0 0 0 0 0`.

## Source validation (Cityscapes val, 500 images)
| Method | best val mIoU | epoch | val acc | fwavacc |
|---|---:|---:|---:|---:|
| baseline | 0.76313 | 54 | 0.95724 | 0.92120 |
| ISW | 0.75314 | 54 | 0.95578 | 0.91878 |

## Target evaluation — our frozen common evaluator
Mapillary Vistas v1.2 validation, 2000 images, 512×1024 single-scale input, bilinear logits → native GT resolution. Mapping `mapillary_v1.2-b790f910d004`.

| Method | mIoU | mean Dice | pixel acc | checkpoint SHA-256 |
|---|---:|---:|---:|---|
| RobustNet baseline | 0.4668 | 0.5996 | 0.8710 | `0dde76b9781c7ef53f78c8969e6a21c95688b4f49859daa2f764cf1346038212` |
| RobustNet ISW | **0.5193** | 0.6580 | 0.8992 | `5eb12cf5517987a3848292334ed2c4b7a34cadf0ae08903c11784eab78621bda` |

Checks: `num_images=2000`, `smoke_only=false`, `publishable=true`, non-null `checkpoint_sha256`, `training_command` contains `--max_iter 40000` and `--val_dataset --arch`, no `--test_mode`; `--wt_layer` matches the method.

## Comparison on the same common evaluator (r19)
| Method | mIoU (our evaluator) |
|---|---:|
| ERM (v1.0, seed 0) | 0.5369 |
| RandConv (v1.0, seed 0) | 0.5727 |
| RobustNet baseline | 0.4668 |
| RobustNet ISW | 0.5193 |

ISW improves over the ResNet-50 DeepLabV3+ baseline by **+5.25** mIoU points on our evaluator (paper: 51.68 → 58.64, +6.96 under its own protocol). The absolute values are **not** directly comparable to the paper's 51.68 / 58.64 because the paper uses a different Mapillary label mapping and inference preprocessing; only direction/magnitude of the ISW gain should be compared.

## Failed / retried runs
- First smoke attempt failed on the scikit-image `multichannel` API (patched, see above).
- Baseline smoke passed; ISW smoke then failed on the tensorboard `make_grid(range=...)` call (patched), after which the ISW smoke passed.
- No full-run failures.

## Artifacts (not pushed: weights/logs)
- Checkpoints: `runs/robustnet_{baseline,isw}_full/dgseg/robustnet_{method}_full/*/best_cityscapes_epoch_54_*.pth`
- Training logs: `runs/robustnet_{baseline,isw}_full/train.log`
- Evaluation JSON: `reports/evaluations/r19_robustnet_{baseline,isw}_mapillary_v1.2.json`
- Timeline: launched 05:23 UTC, both methods finished evaluating 13:52 UTC (~8.5 h).