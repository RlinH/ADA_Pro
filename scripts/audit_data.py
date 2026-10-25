"""Sampled real-data label and RGB/mask overlay audit."""

import argparse
from collections import Counter
from io import BytesIO
import json
import os
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from dgseg.data import track_samples
from dgseg.labels import CITYSCAPES19, LOVEDA7, map_mask


CITY_COLORS = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]
LOVEDA_COLORS = [(255, 255, 255), (255, 0, 0), (255, 255, 0), (0, 0, 255), (159, 129, 183), (0, 255, 0), (255, 195, 128)]


def read(root, archive, member):
    with zipfile.ZipFile(root / archive) as zf:
        return zf.read(member)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DGSEG_DATA_ROOT", "E:/ADA2/DGSEG_DATA")))
    parser.add_argument("--samples-per-role", type=int, default=20)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output_dir = project / "reports/audit_overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    markdown = ["# Sampled label audit", "", "Each row is a real ZIP member. Counts below cover sampled masks only.", ""]
    for track in ("r19", "rs7_u2r", "rs7_r2u"):
        groups = track_samples(args.data_root, track)
        for role, rows in groups.items():
            key = f"{track}_{role}"
            indices = sorted(set(np.linspace(0, len(rows) - 1, min(args.samples_per_role, len(rows)), dtype=int).tolist()))
            counts = Counter()
            raw_values = set()
            entries = []
            colors = CITY_COLORS if track == "r19" else LOVEDA_COLORS
            for position, index in enumerate(indices):
                sample = rows[index]
                image = Image.open(BytesIO(read(args.data_root, sample["image_archive"], sample["image_member"]))).convert("RGB")
                raw = np.asarray(Image.open(BytesIO(read(args.data_root, sample["mask_archive"], sample["mask_member"]))))
                mapped = map_mask(raw, sample["dataset"])
                raw_values.update(np.unique(raw).tolist())
                ids, pixel_counts = np.unique(mapped, return_counts=True)
                counts.update({int(i): int(n) for i, n in zip(ids, pixel_counts)})
                image.thumbnail((512, 512), Image.Resampling.BILINEAR)
                small_mask = Image.fromarray(mapped).resize(image.size, Image.Resampling.NEAREST)
                mask_array = np.asarray(small_mask)
                overlay = np.asarray(image).copy()
                for class_id, color in enumerate(colors):
                    where = mask_array == class_id
                    overlay[where] = (0.55 * overlay[where] + 0.45 * np.asarray(color)).astype(np.uint8)
                filename = f"{key}_{position:02d}.jpg"
                Image.fromarray(overlay).save(output_dir / filename, quality=85)
                entries.append({"sample_id": sample["sample_id"], "overlay": filename})
            valid = sum(n for i, n in counts.items() if i != 255)
            ignored = counts.get(255, 0)
            report[key] = {"sampled_images": len(entries), "raw_unique": sorted(raw_values), "mapped_pixel_counts": dict(sorted(counts.items())), "ignore_ratio": ignored / (valid + ignored), "entries": entries}
            markdown.extend([f"## {key}", "", f"Sampled images: {len(entries)}; ignore ratio: {report[key]['ignore_ratio']:.4f}; raw IDs: {sorted(raw_values)}", ""])
            markdown.extend(f"- `{entry['sample_id']}` → [overlay](audit_overlays/{entry['overlay']})" for entry in entries)
            markdown.append("")
    (project / "reports/DATASET_AUDIT_SAMPLE.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (project / "reports/DATASET_AUDIT_SAMPLE.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps({key: value["sampled_images"] for key, value in report.items()}))


if __name__ == "__main__":
    main()
