import argparse
import json
import os
from pathlib import Path
import zipfile

from dgseg.data import ARCHIVES, write_manifests
from dgseg.labels import export_mapillary_audit

EXPECTED_COUNTS = {
    "r19": {"source_train": 2975, "source_val": 500, "target_mapillary_v1.2": 2000},
    "rs7_u2r": {"source_train": 1040, "source_val": 116, "target_rural": 992},
    "rs7_r2u": {"source_train": 1229, "source_val": 137, "target_urban": 677},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DGSEG_DATA_ROOT", "E:/ADA2/DGSEG_DATA")))
    parser.add_argument("--track", choices=["r19", "rs7_u2r", "rs7_r2u", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    tracks = ["r19", "rs7_u2r", "rs7_r2u"] if args.track == "all" else [args.track]
    for track in tracks:
        result = write_manifests(args.data_root, track, project / "reports/manifests", args.seed)
        counts = {role: value["count"] for role, value in result.items()}
        if args.seed == 42 and counts != EXPECTED_COUNTS[track]:
            raise SystemExit(f"Unexpected {track} coverage: {counts}; expected {EXPECTED_COUNTS[track]}")
        print(track, json.dumps(counts))
    if "r19" in tracks:
        with zipfile.ZipFile(args.data_root / ARCHIVES["mapillary"]) as zf:
            config = json.loads(zf.read("config.json"))
        version = export_mapillary_audit(config, project / "configs/mappings/mapillary_v1_2_to_cityscapes19.json")
        print("Mapillary mapping audit:", version)


if __name__ == "__main__":
    main()
