"""Offline target evaluation. This program never updates weights or selects checkpoints."""

import argparse
import csv
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from dgseg.data import ZipSegDataset, track_samples
from dgseg.labels import map_mask, mapping_version
from dgseg.metrics import Confusion
from dgseg.models import DeepLabV3Plus


def native_item(root: Path, sample: dict, input_size: tuple[int, int]):
    with zipfile.ZipFile(root / sample["image_archive"]) as zf:
        image = Image.open(BytesIO(zf.read(sample["image_member"]))).convert("RGB")
    with zipfile.ZipFile(root / sample["mask_archive"]) as zf:
        target = map_mask(np.asarray(Image.open(BytesIO(zf.read(sample["mask_member"])))), sample["dataset"])
    height, width = input_size
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
    pixels = (pixels - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(pixels.transpose(2, 0, 1)).unsqueeze(0), target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DGSEG_DATA_ROOT", "E:/ADA2/DGSEG_DATA")))
    parser.add_argument("--track", choices=["r19", "rs7_u2r", "rs7_r2u"], required=True)
    parser.add_argument("--target", required=True, help="For example mapillary_v1.2, rural, urban")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--smoke", action="store_true", help="One downsampled real-data sample; metric is never publishable")
    parser.add_argument("--native-smoke", action="store_true", help="One real sample with prediction restored to native GT resolution; never publishable")
    parser.add_argument("--diagnostic", action="store_true", help="Offline input-scale audit with a completed checkpoint; never publishable")
    parser.add_argument("--max-samples", type=int, default=None, help="Evenly spaced target sample count for --diagnostic")
    args = parser.parse_args()
    if args.smoke and args.native_smoke:
        parser.error("Choose only one smoke mode")
    smoke = args.smoke or args.native_smoke
    if args.diagnostic and smoke:
        parser.error("Diagnostic and smoke modes cannot be combined")
    if args.max_samples is not None and (not args.diagnostic or args.max_samples < 1):
        parser.error("--max-samples requires --diagnostic and a positive count")
    if args.height < 1 or args.width < 1:
        parser.error("--height and --width must be positive")
    if smoke:
        args.height, args.width = 128, 256 if args.track == "r19" else 128
    root = args.data_root.resolve()
    groups = track_samples(root, args.track, seed=42)
    target_key = f"target_{args.target}"
    if target_key not in groups:
        parser.error(f"Unavailable target {target_key}; available: {[k for k in groups if k.startswith('target_')]}")
    manifest_path = Path(__file__).resolve().parents[1] / "reports/manifests" / f"{args.track}_{target_key}.csv"
    if not manifest_path.exists():
        parser.error(f"Missing frozen target manifest {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as source:
        frozen_ids = {row["sample_id"] for row in csv.DictReader(source)}
    if frozen_ids != {sample["sample_id"] for sample in groups[target_key]}:
        parser.error("Current target samples differ from frozen manifest")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    protocol_bytes = (Path(__file__).resolve().parents[1] / "configs/protocol.json").read_bytes()
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = json.loads(protocol_bytes)
    if metadata["track"] != args.track:
        parser.error("Checkpoint track does not match evaluation track")
    if metadata.get("protocol_sha256") != protocol_sha256:
        parser.error("Checkpoint protocol SHA-256 does not match current protocol")
    if metadata["smoke_only"] and not smoke:
        parser.error("A smoke checkpoint cannot produce a formal benchmark result")
    if not smoke:
        frozen = protocol["training"]
        last_path = args.checkpoint.parent / "last.pt"
        if not last_path.is_file():
            parser.error("Formal evaluation requires the completed run's last.pt beside best_source_val.pt")
        last_metadata = torch.load(last_path, map_location="cpu", weights_only=False)["metadata"]
        valid_checkpoint = (
            args.checkpoint.name == "best_source_val.pt"
            and metadata["source_only"]
            and metadata["pretrained_backbone"]
            and 1 <= metadata["epoch"] <= frozen["epochs"]
            and last_metadata["epoch"] == frozen["epochs"]
            and metadata["planned_epochs"] == frozen["epochs"]
            and metadata["batch_size"] == frozen["batch_size"]
            and metadata["image_size"] == frozen["crop_by_track"][args.track]
            and metadata["seed"] in frozen["seeds"]
            and all(last_metadata[key] == metadata[key] for key in ("track", "method", "seed", "manifest_sha256", "protocol_sha256"))
        )
        valid_input = args.diagnostic or [args.height, args.width] == protocol["evaluation"]["input_size_by_track"][args.track]
        if not valid_checkpoint or not valid_input:
            parser.error("Formal evaluation requires the complete, source-selected frozen-protocol checkpoint and input size")
    num_classes = 19 if args.track == "r19" else 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3Plus(num_classes, metadata["method"], pretrained_backbone=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    samples = groups[target_key][:1] if smoke else groups[target_key]
    if args.max_samples is not None and args.max_samples < len(samples):
        indices = np.linspace(0, len(samples) - 1, args.max_samples, dtype=int)
        samples = [samples[int(index)] for index in indices]
    metric = Confusion(num_classes)
    with torch.no_grad():
        if args.smoke:
            dataset = ZipSegDataset(root, samples, (args.height, args.width), train=False)
            for index in range(len(dataset)):
                image, target, _ = dataset[index]
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    logits = model(image.unsqueeze(0).to(device))
                metric.update(logits.argmax(1).squeeze(0).cpu().numpy(), target.numpy())
        else:
            for index, sample in enumerate(samples):
                image, target = native_item(root, sample, (args.height, args.width))
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    logits = model(image.to(device))
                logits = F.interpolate(logits.float(), size=target.shape, mode="bilinear", align_corners=False)
                prediction = logits.argmax(1).squeeze(0).cpu().numpy()
                metric.update(prediction, target)
                if (index + 1) % 100 == 0:
                    print(f"Evaluated {index + 1}/{len(samples)}", flush=True)
    summary = metric.summary()
    result = {
        "track": args.track,
        "source": "cityscapes" if args.track == "r19" else "loveda_urban" if args.track == "rs7_u2r" else "loveda_rural",
        "target": args.target,
        "condition": "all",
        "method": metadata["method"],
        "num_classes": num_classes,
        "num_images": len(samples),
        "mIoU": summary["miou"],
        "mean_dice": summary["mean_dice"],
        "pixel_acc": summary["pixel_accuracy"],
        "seed": metadata["seed"],
        "mapping_version": mapping_version(samples[0]["dataset"]),
        "manifest_hash": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "protocol_sha256": protocol_sha256,
        "native_gt_resolution": not args.smoke,
        "smoke_only": smoke,
        "diagnostic_only": args.diagnostic,
        "evaluation_input_size": [args.height, args.width],
        "evaluated_sample_ids_sha256": hashlib.sha256("\n".join(sample["sample_id"] for sample in samples).encode("utf-8")).hexdigest(),
        "publishable": not smoke and not args.diagnostic and not metadata["smoke_only"],
        "metrics": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not smoke and not args.diagnostic:
        csv_path = args.output.with_suffix(".csv")
        fields = ["track", "source", "target", "condition", "method", "num_classes", "num_images", "mIoU", "mean_dice", "pixel_acc", "seed", "mapping_version", "manifest_hash", "protocol_sha256", "native_gt_resolution", "smoke_only", "publishable"]
        with csv_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerow({key: result[key] for key in fields})
    print(json.dumps({"target": args.target, "num_images": len(samples), "smoke_only": smoke, "diagnostic_only": args.diagnostic, "native_gt_resolution": result["native_gt_resolution"], "publishable": result["publishable"]}), flush=True)


if __name__ == "__main__":
    main()
