import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dgseg.data import ZipSegDataset, source_samples
from dgseg.metrics import Confusion
from dgseg.models import DeepLabV3Plus


def make_loader(root, samples, size, train, batch_size, workers):
    return DataLoader(ZipSegDataset(root, samples, size, train=train), batch_size=batch_size, shuffle=train, num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=train)


def save_checkpoint(payload: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


@torch.no_grad()
def source_validate(model, loader, num_classes, device, max_batches=None):
    model.eval()
    metric = Confusion(num_classes)
    for index, (images, masks, _) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        logits = model(images.to(device))
        metric.update(logits.argmax(1).cpu().numpy(), masks.numpy())
    return metric.summary()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DGSEG_DATA_ROOT", "E:/ADA2/DGSEG_DATA")))
    parser.add_argument("--track", choices=["r19", "rs7_u2r", "rs7_r2u"], required=True)
    parser.add_argument("--method", choices=["erm", "mixstyle", "randconv"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--resume", type=Path, help="Resume an interrupted run from last.pt")
    parser.add_argument("--smoke", action="store_true", help="One short real-data optimization and source-validation pass; never a benchmark score")
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error("Batch size must be at least 2 because the ASPP pooled branch uses BatchNorm")
    if args.smoke:
        args.epochs, args.height, args.width, args.workers = 1, 128, 256 if args.track == "r19" else 128, 0
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    project = Path(__file__).resolve().parents[1]
    protocol_bytes = (project / "configs/protocol.json").read_bytes()
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = json.loads(protocol_bytes)
    if not args.smoke:
        frozen = protocol["training"]
        if (args.epochs, args.batch_size, [args.height, args.width], args.lr, args.pretrained_backbone, args.seed in frozen["seeds"]) != (frozen["epochs"], frozen["batch_size"], frozen["crop_by_track"][args.track], frozen["base_lr"], True, True):
            parser.error("Full run differs from frozen protocol; create a new protocol version before training")
    manifest_dir = project / "reports/manifests"
    manifest_hashes = {}
    for role in ("source_train", "source_val"):
        path = manifest_dir / f"{args.track}_{role}.csv"
        if not path.exists():
            parser.error(f"Missing frozen manifest {path}; run build_manifests.py first")
        manifest_hashes[role] = hashlib.sha256(path.read_bytes()).hexdigest()
    groups = source_samples(args.data_root, args.track, seed=42)
    if set(groups) != {"source_train", "source_val"}:
        raise RuntimeError("Training catalog exposed a non-source role")
    for role, samples in groups.items():
        with (manifest_dir / f"{args.track}_{role}.csv").open(newline="", encoding="utf-8") as source:
            frozen_ids = {row["sample_id"] for row in csv.DictReader(source)}
        if frozen_ids != {sample["sample_id"] for sample in samples}:
            parser.error(f"Current {role} samples differ from frozen manifest")
    num_classes = 19 if args.track == "r19" else 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3Plus(num_classes, args.method, args.pretrained_backbone and not args.resume).to(device)
    if args.smoke and args.method == "mixstyle":
        model.mixstyle.p = 1.0
    if args.smoke and args.method == "randconv":
        model.randconv.p = 1.0
    size = (args.height, args.width)
    train_loader = make_loader(args.data_root, groups["source_train"], size, True, args.batch_size, args.workers)
    val_loader = make_loader(args.data_root, groups["source_val"], size, False, 1, args.workers)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    steps = 0
    start_epoch = 0
    if args.resume:
        previous = torch.load(args.resume, map_location="cpu", weights_only=False)
        previous_meta = previous["metadata"]
        if (previous_meta["track"], previous_meta["method"], previous_meta["seed"], previous_meta["manifest_sha256"], previous_meta["pretrained_backbone"], previous_meta["image_size"], previous_meta.get("protocol_sha256"), previous_meta.get("planned_epochs"), previous_meta.get("batch_size"), previous_meta.get("base_lr")) != (args.track, args.method, args.seed, manifest_hashes, args.pretrained_backbone, [args.height, args.width], protocol_sha256, args.epochs, args.batch_size, args.lr):
            parser.error("Resume checkpoint does not match this track/method/seed/source manifest")
        model.load_state_dict(previous["state_dict"], strict=True)
        optimizer.load_state_dict(previous["optimizer"])
        if "scaler" in previous:
            scaler.load_state_dict(previous["scaler"])
        start_epoch = int(previous_meta["epoch"])
        steps = int(previous_meta["steps"])
        best = float(previous_meta.get("best_source_val_miou", previous_meta["source_val_miou"]))
        random.setstate(previous["python_rng"])
        np.random.set_state(previous["numpy_rng"])
        torch.set_rng_state(previous["torch_rng"])
        if device.type == "cuda" and "cuda_rng" in previous:
            torch.cuda.set_rng_state_all(previous["cuda_rng"])
    max_steps = 1 if args.smoke else args.epochs * len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        for images, masks, _ in train_loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, masks, ignore_index=255)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            losses.append(float(loss.detach().cpu()))
            if args.smoke:
                break
            progress = steps / max_steps
            for group in optimizer.param_groups:
                group["lr"] = args.lr * (1 - progress) ** 0.9
        report = source_validate(model, val_loader, num_classes, device, max_batches=1 if args.smoke else None)
        val_miou = report["miou"] if report["miou"] is not None else -1.0
        improved = val_miou > best
        best = max(best, val_miou)
        metadata = {
            "track": args.track, "method": args.method, "seed": args.seed, "epoch": epoch + 1,
            "steps": steps, "num_classes": num_classes, "source_only": True,
            "planned_epochs": args.epochs, "batch_size": args.batch_size, "base_lr": args.lr,
            "pretrained_backbone": args.pretrained_backbone,
            "backbone_weights": "ResNet50_Weights.IMAGENET1K_V2" if args.pretrained_backbone else "random_initialization",
            "image_size": [args.height, args.width],
            "manifest_sha256": manifest_hashes,
            "protocol_sha256": protocol_sha256,
            "smoke_only": args.smoke,
            "source_val_miou": val_miou,
            "best_source_val_miou": best,
            "training_loss": float(np.mean(losses)),
            "torch_version": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        }
        checkpoint = {"state_dict": model.state_dict(), "metadata": metadata, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()}
        if device.type == "cuda":
            checkpoint["cuda_rng"] = torch.cuda.get_rng_state_all()
        save_checkpoint(checkpoint, args.run_dir / "last.pt")
        if improved:
            save_checkpoint(checkpoint, args.run_dir / "best_source_val.pt")
        (args.run_dir / "run_status.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps({"epoch": epoch + 1, "steps": steps, "loss": metadata["training_loss"], "source_val_miou": val_miou, "smoke_only": args.smoke}), flush=True)


if __name__ == "__main__":
    main()
