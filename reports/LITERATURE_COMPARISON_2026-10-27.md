# Literature comparison for the DGSeg SSDG benchmark (2026-10-27)

This is a source-verified comparison of **semantic segmentation domain generalization**. All scores are mIoU percentages. A paper number is a reference scale, not a reproduced result in this repository. Dataset version, class conversion, ignored pixels, checkpoint selection, inference scale, and random seeds can change a score. Do not rank rows from different protocols as if they shared one evaluator.

## R19: Cityscapes train → Mapillary validation

Our frozen v1.0 protocol has Cityscapes as the sole labeled source, ResNet-50 + DeepLabV3+, 19 classes, a manually audited Mapillary Research v1.2 → Cityscapes-19 map, fixed 512×1024 input, and 2,000 target validation images. Only seed 0 is complete: ERM **53.69**, MixStyle adaptation **54.84**, RandConv segmentation adaptation **57.27**.

| Top-conference work | Method / training information | Mapillary mIoU | BDD mIoU | ACDC mIoU | Comparability |
|---|---|---:|---:|---:|---|
| [RobustNet, CVPR 2021, Table 2](https://arxiv.org/pdf/2103.15597) | ResNet-50 DeepLabV3+ source-only baseline; its ISW method selectively whitens style-sensitive feature covariance | **51.68** baseline; **58.64** ISW | 44.96; 50.73 | — | Same source direction and 19-class task, but evaluation conversion and inference recipe are not established as identical to ours. [Official code](https://github.com/sunghac/RobustNet). |
| [WildNet, CVPR 2022, Table 4](https://arxiv.org/pdf/2204.01446) | ResNet-50 DeepLabV3+; diversifies source feature style and content using additional **unlabeled ImageNet images** | **58.79** | 50.94 | — | Same labeled source direction; has extra wild images. Its paper reports 60k iterations, batch 8 and 768×768 crops, versus our 59.5k steps, batch 4 and 512×1024 crops. [Official code](https://github.com/suhyeonlee/WildNet). |
| [SiamDoGe, ECCV 2022, Table 2](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136980590.pdf) | ResNet-50 DeepLabV3+; source-only Siamese views, self-guided feature-statistics randomization and sensitivity-guided consistency | **59.00** | 51.53 | **52.34**, ACDC **val** | The closest published method row to our source and architecture; its Mapillary map/inference settings still need matching. |
| [Rein, CVPR 2024, Tables 3 and 7](https://arxiv.org/pdf/2312.04265) | DINOv2-Large foundation-model backbone + Mask2Former and trainable feature-refinement tokens | **72.3** | 65.0 | **77.6**, ACDC **test** | Different pretraining, architecture and ACDC split; a stronger-model reference, not a ResNet-50 baseline. [Official code](https://github.com/w1oves/Rein). |

The Mapillary result does **not** indicate an abnormally weak ERM by literature scale: 53.69 exceeds RobustNet's reported source-only 51.68. It does show that the current method list is weak as a research comparison. Our 57.27 RandConv result remains below the 58.79–59.00 of published ResNet-50 methods, and MixStyle/RandConv in this repository are segmentation adaptations rather than official reproductions of those papers. BDD and ACDC cannot yet be filled in because their officially accessed data are unavailable here. Rein's large-backbone score must be shown in a separate comparison tier.

Mapillary protocol needs special attention. The [DGInStyle ECCV 2024 evaluation repository](https://github.com/prs-eth/DGInStyle-SegModel) names the same v1.2 archive as ours, but that alone does not establish identical 66→19 mapping, ignore policy, resizing, or checkpoint selection. A credible comparison requires running its evaluator and ours on the same predictions, or documenting each class-map difference.

## RS7: LoveDA Urban ↔ Rural

Our targets are LoveDA **Val** Rural (992 images) and Val Urban (677 images), with a 10% source-only Train holdout for checkpoint selection. The official [LoveDA paper, NeurIPS Datasets and Benchmarks 2021, Table 6](https://arxiv.org/pdf/2110.08733) instead reports domain-transfer results on **Test** with DeepLabV2 + ResNet-50. Its source-only score is **31.74** for Urban → Rural and **31.27** for Rural → Urban. Our DeepLabV3+ ERM seed-0 results are **34.22** and **48.12**, respectively. These numbers use different splits, architectures, training schedules, and evaluation resolution; the 48.12 versus 31.27 difference cannot be claimed as an improvement over the paper.

The same LoveDA Table 6 reports **38.44** (Urban → Rural) and **40.48** (Rural → Urban) for IAST, a self-training **domain adaptation** method. IAST uses unlabeled target-domain images during training, so it is outside this single-source DG protocol. The paper's DeepLabV3+ **47.62** on LoveDA Test (Table 2) is a standard segmentation setting trained with source domains available together, and is likewise not a cross-domain baseline.

The uploaded A100 JSONs show best source-val mIoU **64.67/65.34** and target mIoU **34.22/48.12**. For Urban → Rural, barren has **6.54** IoU and 60.13% of its GT pixels are predicted as background. This is a concrete class failure to inspect. A paired 512/1024-input diagnostic is already implemented but not yet run on the A100; it must remain outside the frozen v1.0 summary.

## What a defensible next result table requires

1. **Keep v1.0 intact.** Its results are an internal, source-only engineering baseline, currently one seed. Do not label MixStyle/RandConv as exact official reproductions or combine their fractions with percentages.
2. **Run a paper-matched road anchor.** Use [RobustNet official code](https://github.com/sunghac/RobustNet) to reproduce its ERM baseline and ISW with Cityscapes train only, then evaluate both on our frozen Mapillary v1.2 manifest and mapping. Record both the paper's original evaluator score and our common-evaluator score, with config, code commit, weights, mapping, and source-only checkpoint rule. SiamDoGe is the next useful no-extra-domain method if its official implementation can be verified.
3. **Run a paper-matched LoveDA anchor.** Reproduce the official LoveDA DeepLabV2-R50 source-only model for both directions, but evaluate on our Val target images under the common seven-class evaluator. Keep the official paper's Test numbers in a distinct literature column. Compare its per-class confusion with our DeepLabV3+ ERM.
4. **Finish uncertainty and coverage.** Run seeds 1 and 2, publish mean ± standard deviation and per-class IoU; run the fixed-checkpoint input-scale diagnostic; add BDD100K and ACDC after official access. Without BDD and ACDC, the requested three-target R19 average and worst-target score are undefined.
5. **Report separate backbone/data tiers.** Show ResNet-50 ImageNet-pretrained source-only methods together. Put WildNet's extra unlabeled wild images, DINOv2/Rein, any synthetic-data generation, and UDA methods in explicitly separated sections.

No external paper score above is entered into `benchmark_summary.csv`, and no frozen training or evaluation protocol was changed by this literature review.
