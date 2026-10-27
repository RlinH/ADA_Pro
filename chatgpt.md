# A100 agent: reproduce a stronger DG segmentation benchmark

The user wants **measured reproductions**, not a literature-only comparison. Execute the supplied scripts on the school A100 and report actual scores. The existing v1.0 results are frozen; do not rewrite them. This handoff starts with the official [RobustNet CVPR 2021 code](https://github.com/sunghac/RobustNet), comparing its ResNet-50 DeepLabV3+ baseline against Instance Selective Whitening (ISW). Its [paper](https://arxiv.org/pdf/2103.15597) reports Cityscapes → Mapillary 51.68/58.64 mIoU under the paper's own evaluation; our Mapillary v1.2 common evaluator has a different label map and input resize, so report its score separately.

## What has already been checked

- Official RobustNet revision pinned to `1ff5c047d7abaf9ab1a7cca23aff812b2f09de15`.
- On the local RTX 3060 6 GB, both untrained official network variants completed **one real Cityscapes forward/backward/optimizer step** at 96×128 (duplicated sample for ASPP BatchNorm) and **one real Mapillary image inference** at 128×256. They are plumbing smoke tests, **not trained or publishable scores**.
- `scripts/prepare_robustnet.py` pins the checkout, patches only the Cityscapes root path to read `ROBUSTNET_CITYSCAPES_ROOT`, and extracts Cityscapes train/val data from the already obtained official archives outside Git.
- That staging script completed locally: 3,475 train/val images, 13,900 annotation files, and exactly 2,975 training image/label pairs.
- `scripts/run_robustnet_a100.sh` keeps all target domains out of training and source validation (`--val_dataset` empty); ISW covariance statistics use Cityscapes training images only.
- `scripts/evaluate_robustnet.py` requires a source-selected official checkpoint, strict parameter loading, frozen target manifest, and native-resolution Mapillary ground truth. Evaluation outputs are JSON in `reports/evaluations/`.

## Run on A100

Use the repo's existing Python environment with CUDA PyTorch. From the repository root:

```bash
git pull --ff-only origin main
export DGSEG_DATA_ROOT="$(cd ../DGSEG_DATA && pwd)"  # change only if the data folder is elsewhere
export ROBUSTNET_WORK_ROOT="$(cd .. && pwd)/DGSEG_REPRO"
df -h "$ROBUSTNET_WORK_ROOT" "$DGSEG_DATA_ROOT" 2>/dev/null || true
MODE=smoke METHODS="baseline isw" bash scripts/run_robustnet_a100.sh
MODE=full METHODS="baseline isw" bash scripts/run_robustnet_a100.sh
```

The full run uses the official 40,000 iterations, 768 crop, class-uniform sampling, official augmentation, and per-GPU batch 4 on **one** A100. The original script specified two GPUs, so this is a one-GPU reproduction with a smaller global batch; disclose that difference. If 40 GB A100 runs out of memory, rerun with `BS=2` and clearly label the changed batch size. Do not substitute a short smoke checkpoint as a full result.

Run one method at a time if needed: `METHODS=isw MODE=full bash scripts/run_robustnet_a100.sh`. The scripts may take many hours and need enough space to extract Cityscapes plus retain checkpoints. The official ImageNet ResNet-50 weight `resnet50-19c8e357.pth` is downloaded by upstream code during training; network access or a populated torch hub cache is required.

## Verify and report

1. Confirm `MODE=smoke` reaches a training checkpoint for both methods. If official legacy code fails under the A100 Python/PyTorch version, apply the smallest compatibility patch to the **external** checkout; record each change and keep method math and data protocol intact. Do not lower crop/iterations in the final run without recording the changed protocol.
2. Confirm each full result JSON has `num_images=2000`, `smoke_only=false`, `publishable=true`, a non-null `checkpoint_sha256`, and `training_command` containing `--max_iter 40000` and `--val_dataset --arch` (empty target list).
3. Compare `r19_robustnet_baseline_mapillary_v1.2.json` and `r19_robustnet_isw_mapillary_v1.2.json` with the existing ERM 53.69 and RandConv 57.27 on **our common evaluator**. Do not compare the numerical gap directly to the paper's 51.68/58.64 unless label mapping and inference preprocessing are matched.
4. Add a short run report with GPU, software versions, effective batch size, exact upstream commit/patches, source-only controls, checkpoint hashes, source validation, target metrics, and any failed/retried runs. Update `progress.md` at the top with changes and a result table. Commit/push code and small reports to `main` per `AGENTS.md`. Never push datasets, weights, ZIPs, or full training logs.

## Next high-capacity tier

After RobustNet is verified, implement [Rein (CVPR 2024 official code)](https://github.com/w1oves/Rein) as a **separate foundation-model tier** (DINOv2-Large + Mask2Former), using only Cityscapes labels for training and the same frozen Mapillary target list for final evaluation. The paper's 72.3 Mapillary mIoU uses its own mapping/inference protocol. Pin the code and pretrained DINOv2 weights, verify licenses and hashes, and run a local A100 smoke before a full training run. Do not label a downloaded paper checkpoint as a trained reproduction. This phase has **not** been scripted or run yet.
