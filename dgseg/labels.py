"""Versioned, explicit label mappings. Unknown IDs fail loudly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


IGNORE = 255
CITYSCAPES19 = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle",
]
LOVEDA7 = ["background", "building", "road", "water", "barren", "forest", "agriculture"]

# Official Cityscapes labelIds -> trainIds. All other documented IDs are ignored.
CITYSCAPES_LABELIDS_TO_19 = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}

# Mapillary Vistas Research Edition v1.2: conservative label audit.
# Each key is the raw semantic-label PNG index. None means ignore.
MAPILLARY_V12_AUDIT = {
    0: (None, "bird has no Cityscapes-19 class"),
    1: (None, "ground animal has no Cityscapes-19 class"),
    2: (None, "curb is ignored in Cityscapes-19"),
    3: (4, "fence"),
    4: (None, "guard rail is ignored in Cityscapes-19"),
    5: (None, "other barrier is semantically mixed"),
    6: (3, "wall"),
    7: (None, "bike lane does not have an unambiguous Cityscapes-19 match"),
    8: (0, "plain crosswalk surface is part of the road"),
    9: (None, "curb cut may be road or sidewalk"),
    10: (None, "parking is ignored in Cityscapes-19"),
    11: (None, "pedestrian area is not necessarily sidewalk"),
    12: (None, "rail track is ignored in Cityscapes-19"),
    13: (0, "road"),
    14: (0, "service lane is a road surface"),
    15: (1, "sidewalk"),
    16: (None, "bridge is ignored in Cityscapes-19"),
    17: (2, "building"),
    18: (None, "tunnel is ignored in Cityscapes-19"),
    19: (11, "person"),
    20: (12, "bicyclist is a rider"),
    21: (12, "motorcyclist is a rider"),
    22: (None, "other rider may not match the Cityscapes rider definition"),
    23: (0, "crosswalk marking lies on road"),
    24: (0, "general lane marking lies on road"),
    25: (None, "mountain extent may mix vegetation and terrain"),
    26: (None, "sand may be terrain or road shoulder"),
    27: (10, "sky"),
    28: (None, "snow may cover several underlying classes"),
    29: (9, "terrain"),
    30: (8, "vegetation"),
    31: (None, "water has no Cityscapes-19 class"),
    32: (None, "banner is ignored in Cityscapes-19"),
    33: (None, "bench is ignored in Cityscapes-19"),
    34: (None, "bike rack is ignored in Cityscapes-19"),
    35: (None, "billboard is not a traffic sign"),
    36: (None, "catch basin is ignored in Cityscapes-19"),
    37: (None, "CCTV camera is ignored in Cityscapes-19"),
    38: (None, "fire hydrant is ignored in Cityscapes-19"),
    39: (None, "junction box is ignored in Cityscapes-19"),
    40: (None, "mailbox is ignored in Cityscapes-19"),
    41: (None, "manhole is ignored in Cityscapes-19"),
    42: (None, "phone booth is ignored in Cityscapes-19"),
    43: (None, "pothole is not a separate Cityscapes-19 class"),
    44: (None, "street light includes lamp and support"),
    45: (5, "pole"),
    46: (None, "traffic-sign frame may include sign and support"),
    47: (5, "utility pole"),
    48: (6, "traffic light"),
    49: (7, "back of a traffic sign"),
    50: (7, "front of a traffic sign"),
    51: (None, "trash can is ignored in Cityscapes-19"),
    52: (18, "bicycle"),
    53: (None, "boat has no Cityscapes-19 class"),
    54: (15, "bus"),
    55: (13, "car"),
    56: (None, "caravan is ignored in Cityscapes-19"),
    57: (17, "motorcycle"),
    58: (16, "rail vehicle corresponds to train"),
    59: (None, "other vehicle is semantically mixed"),
    60: (None, "trailer is ignored in Cityscapes-19"),
    61: (14, "truck"),
    62: (None, "wheeled slow has no unambiguous class"),
    63: (None, "car mount is void"),
    64: (None, "ego vehicle is void"),
    65: (None, "unlabeled is void"),
}


def map_mask(raw: np.ndarray, dataset: str) -> np.ndarray:
    if raw.ndim != 2:
        raise ValueError(f"Expected indexed 2D mask, got shape {raw.shape}")
    raw = np.asarray(raw)
    if dataset == "cityscapes":
        allowed = (raw <= 33) | (raw == IGNORE)
        mapping = CITYSCAPES_LABELIDS_TO_19
    elif dataset == "mapillary_v1.2":
        allowed = raw <= 65
        mapping = {k: v for k, (v, _) in MAPILLARY_V12_AUDIT.items() if v is not None}
    elif dataset == "loveda":
        allowed = raw <= 7
        mapping = {i: i - 1 for i in range(1, 8)}
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if not bool(np.all(allowed)):
        unknown = np.unique(raw[~allowed]).tolist()
        raise ValueError(f"Unexpected {dataset} raw label IDs: {unknown[:20]}")
    out = np.full(raw.shape, IGNORE, dtype=np.uint8)
    for source, target in mapping.items():
        out[raw == source] = target
    return out


def mapping_version(dataset: str) -> str:
    if dataset == "cityscapes":
        obj = CITYSCAPES_LABELIDS_TO_19
    elif dataset == "mapillary_v1.2":
        obj = MAPILLARY_V12_AUDIT
    elif dataset == "loveda":
        obj = {i: i - 1 for i in range(1, 8)}
    else:
        raise ValueError(dataset)
    digest = hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]
    return f"{dataset}-{digest}"


def export_mapillary_audit(config: dict, destination: Path) -> str:
    labels = config["labels"]
    if len(labels) != 66 or set(MAPILLARY_V12_AUDIT) != set(range(66)):
        raise ValueError("Mapillary v1.2 label schema is not the audited 66-class schema")
    rows = []
    for raw_id, label in enumerate(labels):
        target, reason = MAPILLARY_V12_AUDIT[raw_id]
        rows.append({
            "raw_id": raw_id,
            "official_name": label["name"],
            "readable": label["readable"],
            "cityscapes19_id": target,
            "cityscapes19_name": CITYSCAPES19[target] if target is not None else None,
            "reason": reason,
        })
    payload = {"source_version": "Mapillary Vistas Research Edition v1.2", "config_version_field": config.get("version"), "mapping_version": mapping_version("mapillary_v1.2"), "rows": rows}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload["mapping_version"]
