# Seed-0 baseline audit, 2026-10-27

## Observed result and reference scale

All numbers below are mIoU percentages. The reference results indicate scale, **not a matched reproduction**: architectures, splits, maps, augmentation, input sizes, and evaluation differ.

| Direction | Our ERM, seed 0 | Published source-only reference | Reference |
|---|---:|---:|---|
| Cityscapes → Mapillary v1.2, 19 classes | 53.69 | 51.68 | [RobustNet, CVPR 2021, Table 2](https://arxiv.org/pdf/2103.15597) |
| LoveDA Urban → Rural, 7 classes | 34.22 | 31.74 | [LoveDA, NeurIPS 2021, Table 6](https://arxiv.org/pdf/2110.08733) |
| LoveDA Rural → Urban, 7 classes | 48.12 | 31.27 | [LoveDA, NeurIPS 2021, Table 6](https://arxiv.org/pdf/2110.08733) |

The apparent low values do not, by themselves, show that the ERM implementation is broken. LoveDA's rural/urban appearance shift is a difficult cross-domain test. The published source-only reference for Urban → Rural is about 32%. Our cross-domain scores should not be compared to same-domain segmentation scores. The large Rural → Urban difference from the paper is further evidence that the protocols are not numerically interchangeable.

The uploaded A100 ERM metadata and target JSONs add direct evidence. Source validation uses a deterministic 10% holdout of the training domain and 512×512 masks; target evaluation uses the other domain with predictions restored to native 1024×1024 masks, so the gap includes an evaluation-resolution difference as well as domain shift.

| LoveDA direction | Best source-val mIoU | Target mIoU | Difference | Target images |
|---|---:|---:|---:|---:|
| Urban → Rural | 64.67 | 34.22 | −30.44 points | 992 |
| Rural → Urban | 65.34 | 48.12 | −17.22 points | 677 |

The first target has a class-specific failure rather than a global zero-output failure. For Urban → Rural, target IoU is 6.54% for barren and 17.20% for forest, compared with 49.81% for background and 51.39% for water. The confusion matrix shows 60.13% of true barren pixels, 58.04% of forest pixels, and 39.61% of agriculture pixels predicted as background. Barren recall is only 9.08%. All seven classes are present in both target sets. Rural → Urban target IoU ranges from 34.50% (barren) to 70.52% (water), with no similarly collapsed class. These patterns fit the documented LoveDA class-distribution and complex-background shift, though the input-scale test below is still needed to isolate any resolution penalty.

The published seed-0 differences also show that the added methods are not universally helpful: compared with ERM, MixStyle/RandConv improve Cityscapes → Mapillary by 1.15/3.58 percentage points, while both are below ERM on both LoveDA directions. RandConv here is a segmentation adaptation, not an exact reproduction of the original method.

## Code and data checks

- The A100 run report records complete 80-epoch runs, ImageNet-pretrained ResNet-50, verified source archives, frozen manifest/protocol hashes, and source-validation checkpoint selection. The two ERM LoveDA `run_status.json` and target evaluation JSONs are now available in GitHub and agree on track, method, seed, protocol hash, and target mIoU. Model weights and full training histories remain only on A100.
- The LoveDA map ignores raw 0 and maps raw 1–7 to classes 0–6, consistent with the [official LoveDA class list](https://github.com/Junjue-Wang/LoveDA/blob/master/README.md) and [official loader](https://github.com/Junjue-Wang/LoveDA/blob/master/Unsupervised_Domian_Adaptation/data/loveda.py). The Cityscapes map uses the standard 19 train IDs. Confusion is accumulated globally and unknown labels are ignored.
- The current LoveDA evaluator shrinks a 1024×1024 target RGB image to 512×512 before inference and upsamples logits to the original ground-truth size. This may affect visual detail. In a local geometry-only check on 80 evenly spaced target masks per direction, nearest-neighbor 1024→512→1024 label reconstruction yielded 98.39% and 98.41% mIoU. This check does not measure the effect of shrinking RGB inputs, so it cannot rule out an input-resolution penalty.

## Remaining diagnosis on A100

1. Run the paired 100-image 512/1024 input-scale diagnostic commands in the README, compare `evaluated_sample_ids_sha256`, and compare mIoU. Each output is explicitly non-publishable and excluded from the frozen v1.0 summary.
2. Inspect visual overlays for Urban → Rural barren, forest, and agriculture. Check whether background overprediction is associated with genuinely ambiguous labels or model errors.
3. Complete seeds 1 and 2 before drawing a stable method-ranking conclusion.

No training hyperparameter or formal protocol was changed in this audit.
