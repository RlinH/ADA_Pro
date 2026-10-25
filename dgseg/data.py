"""Read the legally obtained original ZIP archives without extracting them."""

from __future__ import annotations

import csv
import hashlib
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
import random
import zipfile

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

from .labels import IGNORE, map_mask, mapping_version


ARCHIVES = {
    "city_images": "downloads/cityscapes/leftImg8bit_trainvaltest.zip",
    "city_masks": "downloads/cityscapes/gtFine_trainvaltest.zip",
    "loveda_train": "downloads/loveda/Train.zip",
    "loveda_val": "downloads/loveda/Val.zip",
    "mapillary": "downloads/mapillary_v1.2/mapillary-vistas-dataset_public_v1.2.zip",
}


@lru_cache(maxsize=12)
def _archive(path: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(path)


def _sample(dataset: str, split: str, domain: str, image_archive: str, image_member: str, mask_archive: str, mask_member: str, version: str) -> dict:
    return {
        "dataset": dataset, "split": split, "domain": domain,
        "sample_id": f"{dataset}/{split}/{domain}/{Path(image_member).stem}",
        "image_archive": image_archive, "image_member": image_member,
        "mask_archive": mask_archive, "mask_member": mask_member,
        "source_version": version,
        "mapping_version": mapping_version(dataset),
    }


def _cityscapes(root: Path) -> dict[str, list[dict]]:
    images_path = root / ARCHIVES["city_images"]
    masks_path = root / ARCHIVES["city_masks"]
    with zipfile.ZipFile(images_path) as images, zipfile.ZipFile(masks_path) as masks:
        mask_names = set(masks.namelist())
        groups = {"train": [], "val": []}
        for name in images.namelist():
            pieces = name.split("/")
            if len(pieces) != 4 or pieces[1] not in groups or not name.endswith("_leftImg8bit.png"):
                continue
            split, city, filename = pieces[1], pieces[2], pieces[3]
            base = filename.removesuffix("_leftImg8bit.png")
            mask = f"gtFine/{split}/{city}/{base}_gtFine_labelIds.png"
            if mask not in mask_names:
                raise ValueError(f"Missing Cityscapes label for {name}")
            groups[split].append(_sample("cityscapes", split, city, ARCHIVES["city_images"], name, ARCHIVES["city_masks"], mask, "Cityscapes gtFine original labelIds"))
    for rows in groups.values():
        rows.sort(key=lambda x: x["sample_id"])
    return groups


def _loveda(root: Path) -> dict[str, list[dict]]:
    groups = {}
    for split, archive_key in [("Train", "loveda_train"), ("Val", "loveda_val")]:
        path = root / ARCHIVES[archive_key]
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            for domain in ("Urban", "Rural"):
                rows = []
                prefix = f"{split}/{domain}/images_png/"
                for image in sorted(n for n in names if n.startswith(prefix) and n.lower().endswith(".png")):
                    mask = f"{split}/{domain}/masks_png/{Path(image).name}"
                    if mask not in names:
                        raise ValueError(f"Missing LoveDA label for {image}")
                    rows.append(_sample("loveda", split.lower(), domain.lower(), ARCHIVES[archive_key], image, ARCHIVES[archive_key], mask, "LoveDA Zenodo record 5706578"))
                groups[f"{split.lower()}_{domain.lower()}"] = rows
    return groups


def _loveda_train_only(root: Path, domain: str) -> list[dict]:
    """Source-only indexer: never opens LoveDA Val or any target archive."""
    path = root / ARCHIVES["loveda_train"]
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        prefix = f"Train/{domain}/images_png/"
        rows = []
        for image in sorted(n for n in names if n.startswith(prefix) and n.lower().endswith(".png")):
            mask = f"Train/{domain}/masks_png/{Path(image).name}"
            if mask not in names:
                raise ValueError(f"Missing LoveDA source label for {image}")
            rows.append(_sample("loveda", "train", domain.lower(), ARCHIVES["loveda_train"], image, ARCHIVES["loveda_train"], mask, "LoveDA Zenodo record 5706578"))
    return rows


def _mapillary(root: Path) -> list[dict]:
    path = root / ARCHIVES["mapillary"]
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        rows = []
        for image in sorted(n for n in names if n.startswith("validation/images/") and n.lower().endswith((".jpg", ".jpeg", ".png"))):
            mask = f"validation/labels/{Path(image).stem}.png"
            if mask not in names:
                raise ValueError(f"Missing Mapillary semantic label for {image}")
            rows.append(_sample("mapillary_v1.2", "validation", "global", ARCHIVES["mapillary"], image, ARCHIVES["mapillary"], mask, "Mapillary Vistas Research Edition v1.2"))
    return rows


def _holdout(rows: list[dict], seed: int, fraction: float = 0.1) -> tuple[list[dict], list[dict]]:
    ranked = sorted(rows, key=lambda x: hashlib.sha256(f"{seed}:{x['sample_id']}".encode()).hexdigest())
    n_val = max(1, round(len(ranked) * fraction))
    val_ids = {x["sample_id"] for x in ranked[:n_val]}
    train = sorted((x for x in rows if x["sample_id"] not in val_ids), key=lambda x: x["sample_id"])
    val = sorted((x for x in rows if x["sample_id"] in val_ids), key=lambda x: x["sample_id"])
    return train, val


def track_samples(root: Path, track: str, seed: int = 42) -> dict[str, list[dict]]:
    root = root.resolve()
    if track == "r19":
        city = _cityscapes(root)
        targets = {"mapillary_v1.2": _mapillary(root)} if (root / ARCHIVES["mapillary"]).exists() else {}
        return {"source_train": city["train"], "source_val": city["val"], **{f"target_{k}": v for k, v in targets.items()}}
    if track in ("rs7_u2r", "rs7_r2u"):
        data = _loveda(root)
        source = "urban" if track == "rs7_u2r" else "rural"
        target = "rural" if source == "urban" else "urban"
        train, source_val = _holdout(data[f"train_{source}"], seed)
        return {"source_train": train, "source_val": source_val, f"target_{target}": data[f"val_{target}"]}
    raise ValueError(f"Unknown or unavailable track: {track}")


def source_samples(root: Path, track: str, seed: int = 42) -> dict[str, list[dict]]:
    """Training-facing catalog: indexes only the selected source domain."""
    root = root.resolve()
    if track == "r19":
        city = _cityscapes(root)
        return {"source_train": city["train"], "source_val": city["val"]}
    if track in ("rs7_u2r", "rs7_r2u"):
        domain = "Urban" if track == "rs7_u2r" else "Rural"
        train, val = _holdout(_loveda_train_only(root, domain), seed)
        return {"source_train": train, "source_val": val}
    raise ValueError(f"Unknown or unavailable source track: {track}")


def _read_member(root: Path, archive_rel: str, member: str) -> bytes:
    return _archive(str(root / archive_rel)).read(member)


class ZipSegDataset(Dataset):
    def __init__(self, root: Path, samples: list[dict], size: tuple[int, int], train: bool = False):
        self.root = Path(root)
        self.samples = samples
        self.size = size  # (height, width)
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(BytesIO(_read_member(self.root, sample["image_archive"], sample["image_member"]))).convert("RGB")
        mask_image = Image.open(BytesIO(_read_member(self.root, sample["mask_archive"], sample["mask_member"])))
        raw = np.asarray(mask_image)
        mask = Image.fromarray(map_mask(raw, sample["dataset"]), mode="L")
        height, width = self.size
        if self.train:
            scale = random.uniform(0.75, 1.5)
            resized = (max(width, round(width * scale)), max(height, round(height * scale)))
            image = image.resize(resized, Image.Resampling.BILINEAR)
            mask = mask.resize(resized, Image.Resampling.NEAREST)
            pad_right = max(0, width - image.width)
            pad_bottom = max(0, height - image.height)
            if pad_right or pad_bottom:
                image = ImageOps.expand(image, (0, 0, pad_right, pad_bottom), fill=(123, 116, 103))
                mask = ImageOps.expand(mask, (0, 0, pad_right, pad_bottom), fill=IGNORE)
            left = random.randint(0, image.width - width)
            top = random.randint(0, image.height - height)
            box = (left, top, left + width, top + height)
            image, mask = image.crop(box), mask.crop(box)
            if random.random() < 0.5:
                image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
        else:
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            mask = mask.resize((width, height), Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype=np.float32).copy() / 255.0
        image_array = (image_array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1))
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())
        return image_tensor, mask_tensor, sample["sample_id"]


MANIFEST_FIELDS = ["dataset", "split", "domain", "role", "sample_id", "image_path", "mask_path", "width", "height", "source_version", "mapping_version", "sha256_or_file_id"]


def write_manifests(root: Path, track: str, destination: Path, seed: int = 42) -> dict[str, dict]:
    groups = track_samples(root, track, seed)
    destination.mkdir(parents=True, exist_ok=True)
    result = {}
    seen = set()
    for role, samples in groups.items():
        records = []
        for sample in samples:
            if sample["sample_id"] in seen:
                raise ValueError(f"Sample leaks across roles: {sample['sample_id']}")
            seen.add(sample["sample_id"])
            if sample["dataset"] == "cityscapes":
                width, height = 2048, 1024
            elif sample["dataset"] == "loveda":
                width, height = 1024, 1024
            else:
                image = Image.open(BytesIO(_read_member(root, sample["image_archive"], sample["image_member"])))
                width, height = image.size
            records.append({
                "dataset": sample["dataset"], "split": sample["split"], "domain": sample["domain"], "role": role,
                "sample_id": sample["sample_id"],
                "image_path": f"zip://{sample['image_archive']}!{sample['image_member']}",
                "mask_path": f"zip://{sample['mask_archive']}!{sample['mask_member']}",
                "width": width, "height": height,
                "source_version": sample["source_version"], "mapping_version": sample["mapping_version"],
                "sha256_or_file_id": sample["sample_id"],
            })
        path = destination / f"{track}_{role}.csv"
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(records)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result[role] = {"count": len(records), "path": str(path), "sha256": digest}
    (destination / f"{track}_manifest_hashes.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
