"""Global pixel confusion and deterministic segmentation metrics."""

from __future__ import annotations

import numpy as np


class Confusion:
    def __init__(self, num_classes: int, ignore: int = 255):
        self.num_classes = num_classes
        self.ignore = ignore
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.valid_pixels = 0
        self.ignored_pixels = 0

    def update(self, prediction, target) -> None:
        prediction = np.asarray(prediction, dtype=np.int64)
        target = np.asarray(target, dtype=np.int64)
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction/target shapes differ: {prediction.shape} vs {target.shape}")
        keep = target != self.ignore
        self.ignored_pixels += int((~keep).sum())
        self.valid_pixels += int(keep.sum())
        if np.any((target[keep] < 0) | (target[keep] >= self.num_classes)):
            raise ValueError("Target contains an invalid class ID")
        if np.any((prediction[keep] < 0) | (prediction[keep] >= self.num_classes)):
            raise ValueError("Prediction contains an invalid class ID")
        bins = np.bincount(self.num_classes * target[keep] + prediction[keep], minlength=self.num_classes**2)
        self.matrix += bins.reshape(self.num_classes, self.num_classes)

    def summary(self) -> dict:
        matrix = self.matrix.astype(np.float64)
        intersection = np.diag(matrix)
        gt_support = matrix.sum(axis=1)
        pred_support = matrix.sum(axis=0)
        union = gt_support + pred_support - intersection
        present = union > 0
        iou = np.divide(intersection, union, out=np.full_like(union, np.nan), where=present)
        dice_denominator = gt_support + pred_support
        dice = np.divide(2 * intersection, dice_denominator, out=np.full_like(union, np.nan), where=dice_denominator > 0)
        total = self.valid_pixels + self.ignored_pixels
        return {
            "num_classes": self.num_classes,
            "valid_pixels": self.valid_pixels,
            "ignored_pixels": self.ignored_pixels,
            "ignore_ratio": self.ignored_pixels / total if total else None,
            "absent_class_policy": "exclude only classes absent from both GT and prediction",
            "included_class_ids": np.flatnonzero(present).tolist(),
            "miou": float(np.nanmean(iou)) if np.any(present) else None,
            "mean_dice": float(np.nanmean(dice)) if np.any(dice_denominator > 0) else None,
            "pixel_accuracy": float(intersection.sum() / self.valid_pixels) if self.valid_pixels else None,
            "per_class_iou": [None if np.isnan(x) else float(x) for x in iou],
            "per_class_dice": [None if np.isnan(x) else float(x) for x in dice],
            "gt_support": gt_support.astype(np.int64).tolist(),
            "confusion_matrix": self.matrix.tolist(),
        }
