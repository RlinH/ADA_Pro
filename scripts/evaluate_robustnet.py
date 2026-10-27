"""Evaluate a strictly loaded official RobustNet checkpoint on our frozen r19 target.

This common evaluator is deliberately distinct from the paper's Mapillary protocol.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from dgseg.data import track_samples
from dgseg.labels import mapping_version
from dgseg.metrics import Confusion
from scripts.evaluate import native_item


COMMIT = "1ff5c047d7abaf9ab1a7cca23aff812b2f09de15"


def make_model(vendor: Path, method: str) -> torch.nn.Module:
    sys.path.insert(0, str(vendor.resolve()))
    from network import Resnet  # type: ignore
    from network.deepv3 import DeepR50V3PlusD  # type: ignore

    # The checkpoint contains the complete network; skip a redundant ImageNet download.
    original = Resnet.resnet50

    def without_pretrain(*args, **kwargs):
        kwargs["pretrained"] = False
        return original(*args, **kwargs)

    Resnet.resnet50 = without_pretrain
    try:
        wt_layer = [0, 0, 2, 2, 2, 0, 0] if method == "isw" else [0] * 7
        model = DeepR50V3PlusD(
            args=SimpleNamespace(wt_layer=wt_layer, relax_denom=0.0, clusters=2, use_wtloss=method == "isw"),
            num_classes=19,
            criterion=None,
            criterion_aux=None,
        )
    finally:
        Resnet.resnet50 = original
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DGSEG_DATA_ROOT", "_unset_data_root_")))
    parser.add_argument("--method", choices=["baseline", "isw"], required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--smoke", action="store_true", help="Random weights, one real sample; only tests plumbing")
    args = parser.parse_args()
    if not args.vendor.is_dir() or not args.data_root.joinpath("downloads").is_dir():
        parser.error("Vendor checkout and data-root/downloads are required")
    actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.vendor, text=True).strip()
    if actual_commit != COMMIT:
        parser.error(f"Unexpected RobustNet commit {actual_commit}; expected {COMMIT}")
    if not args.smoke and (not args.checkpoint or not args.checkpoint.is_file() or not args.output):
        parser.error("Full evaluation requires --checkpoint and --output")
    if args.height < 32 or args.width < 32:
        parser.error("Input dimensions must be at least 32 pixels")
    root = args.data_root.resolve()
    groups = track_samples(root, "r19", seed=42)
    samples = groups["target_mapillary_v1.2"]
    manifest = Path(__file__).resolve().parents[1] / "reports/manifests/r19_target_mapillary_v1.2.csv"
    with manifest.open(newline="", encoding="utf-8") as source:
        frozen = {row["sample_id"] for row in csv.DictReader(source)}
    if frozen != {row["sample_id"] for row in samples}:
        raise RuntimeError("Mapillary samples differ from frozen manifest")
    model = make_model(args.vendor, args.method)
    checkpoint_sha256 = None
    training_command = None
    if not args.smoke:
        if not args.checkpoint.name.startswith("best_cityscapes_"):
            parser.error("Formal evaluation requires an official source-validation-selected best_cityscapes checkpoint")
        with args.checkpoint.open("rb") as inp:
            digest = hashlib.sha256()
            while chunk := inp.read(1024 * 1024):
                digest.update(chunk)
        checkpoint_sha256 = digest.hexdigest()
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        training_command = checkpoint.get("command", "")
        tokens = shlex.split(training_command)
        if ("--dataset" not in tokens or tokens[tokens.index("--dataset") + 1] != "cityscapes"
                or "--max_iter" not in tokens or tokens[tokens.index("--max_iter") + 1] != "40000"
                or "--val_dataset" not in tokens or tokens[tokens.index("--val_dataset") + 1] != "--arch"
                or "--test_mode" in tokens):
            raise RuntimeError("Checkpoint command does not match full source-only RobustNet training")
        expected = ["0", "0", "2", "2", "2", "0", "0"] if args.method == "isw" else ["0"] * 7
        if "--wt_layer" not in tokens or tokens[tokens.index("--wt_layer") + 1:tokens.index("--wt_layer") + 8] != expected:
            raise RuntimeError("Checkpoint whitening layers do not match requested method")
        state = checkpoint["state_dict"]
        if not state or not all(key.startswith("module.") for key in state):
            raise RuntimeError("Expected official DDP checkpoint with module. keys")
        model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    if args.smoke:
        samples = samples[:1]
        args.height, args.width = 128, 256
    metric = Confusion(19)
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            image, target = native_item(root, sample, (args.height, args.width))
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(image.to(device))
            logits = F.interpolate(logits.float(), size=target.shape, mode="bilinear", align_corners=False)
            metric.update(logits.argmax(1).squeeze(0).cpu().numpy(), target)
            if (index + 1) % 100 == 0:
                print(f"Evaluated {index + 1}/{len(samples)}", flush=True)
    summary = metric.summary()
    result = {
        "method": f"robustnet_{args.method}",
        "track": "r19",
        "target": "mapillary_v1.2",
        "num_images": len(samples),
        "mIoU": summary["miou"],
        "mean_dice": summary["mean_dice"],
        "pixel_acc": summary["pixel_accuracy"],
        "mapping_version": mapping_version("mapillary_v1.2"),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "checkpoint_sha256": checkpoint_sha256,
        "upstream_commit": COMMIT,
        "training_command": training_command,
        "input_size": [args.height, args.width],
        "native_gt_resolution": True,
        "smoke_only": args.smoke,
        "publishable": not args.smoke,
        "metrics": summary,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("method", "num_images", "mIoU", "smoke_only", "publishable")}), flush=True)


if __name__ == "__main__":
    main()
