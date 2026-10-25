"""Aggregate only formal, native-resolution, source-selected results."""

import csv
import json
from pathlib import Path
import statistics


PROJECT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT / "reports/evaluations"
OUTPUT = PROJECT / "reports/benchmark_summary.csv"
METHODS = ("erm", "mixstyle", "randconv")
TARGETS = [
    ("r19", "mapillary_v1.2", "ready"),
    ("r19", "bdd100k", "blocked_missing_data"),
    ("r19", "acdc_all", "blocked_missing_data"),
    ("rs7_u2r", "rural", "ready"),
    ("rs7_r2u", "urban", "ready"),
    ("s16", "cityscapes", "blocked_incomplete_synthia"),
]
FIELDS = ["track", "source", "target", "method", "status", "num_seeds", "seeds", "num_images", "num_classes", "mIoU_mean", "mIoU_std", "mean_dice_mean", "pixel_acc_mean", "mapping_version", "manifest_hash", "protocol_sha256"]


def main() -> None:
    by_key = {}
    for path in EVAL_DIR.glob("*.json") if EVAL_DIR.exists() else []:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("publishable") or payload.get("smoke_only") or not payload.get("native_gt_resolution"):
            continue
        key = (payload["track"], payload["target"], payload["method"])
        by_key.setdefault(key, []).append(payload)
    rows = []
    for track, target, availability in TARGETS:
        for method in METHODS:
            results = sorted(by_key.get((track, target, method), []), key=lambda x: x["seed"])
            seeds = [x["seed"] for x in results]
            if len(seeds) != len(set(seeds)):
                raise ValueError(f"Duplicate seed for {track}/{target}/{method}")
            if results and (len({x["mapping_version"] for x in results}) != 1 or len({x["manifest_hash"] for x in results}) != 1 or len({x["num_images"] for x in results}) != 1 or len({x["protocol_sha256"] for x in results}) != 1):
                raise ValueError(f"Inconsistent evaluation protocol for {track}/{target}/{method}")
            status = ("ready_unrun" if availability == "ready" else availability) if not results else "complete_3seed" if len(results) == 3 else "partial"
            scores = [x["mIoU"] for x in results]
            source = results[0]["source"] if results else "cityscapes" if track == "r19" else "loveda_urban" if track == "rs7_u2r" else "loveda_rural" if track == "rs7_r2u" else "synthia"
            rows.append({
                "track": track, "source": source, "target": target, "method": method,
                "status": status, "num_seeds": len(results), "seeds": ",".join(map(str, seeds)),
                "num_images": results[0]["num_images"] if results else "",
                "num_classes": results[0]["num_classes"] if results else 19 if track == "r19" else 7 if track.startswith("rs7") else 16,
                "mIoU_mean": statistics.mean(scores) if scores else "",
                "mIoU_std": statistics.stdev(scores) if len(scores) > 1 else "",
                "mean_dice_mean": statistics.mean(x["mean_dice"] for x in results) if results else "",
                "pixel_acc_mean": statistics.mean(x["pixel_acc"] for x in results) if results else "",
                "mapping_version": results[0]["mapping_version"] if results else "",
                "manifest_hash": results[0]["manifest_hash"] if results else "",
                "protocol_sha256": results[0]["protocol_sha256"] if results else "",
            })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} protocol rows to {OUTPUT}")


if __name__ == "__main__":
    main()
