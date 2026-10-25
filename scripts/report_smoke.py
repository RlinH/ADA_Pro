import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
RUNS = [
    "r19_erm_smoke", "r19_mixstyle_smoke", "r19_randconv_smoke",
    "rs7_u2r_erm_smoke", "rs7_r2u_erm_smoke",
]
PRETRAINED_RUN = "r19_erm_pretrained_smoke"
EVALUATIONS = [
    "r19_erm_mapillary_native_smoke.json",
    "rs7_u2r_erm_smoke.json",
    "rs7_r2u_erm_smoke.json",
]


def main() -> None:
    protocol_hash = hashlib.sha256((PROJECT / "configs/protocol.json").read_bytes()).hexdigest()
    runs = {}
    for name in RUNS:
        path = PROJECT / "runs" / name / "run_status.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report["smoke_only"] or not report["source_only"] or report["steps"] != 1:
            raise ValueError(f"Invalid local smoke result: {path}")
        if report["protocol_sha256"] != protocol_hash:
            raise ValueError(f"Stale protocol in {path}")
        if not (PROJECT / "runs" / name / "best_source_val.pt").is_file():
            raise ValueError(f"Checkpoint missing: {name}")
        runs[name] = {key: report[key] for key in ("track", "method", "steps", "device", "gpu", "manifest_sha256", "protocol_sha256")}
    pretrained_path = PROJECT / "runs" / PRETRAINED_RUN / "run_status.json"
    pretrained_hash = None
    if pretrained_path.exists():
        report = json.loads(pretrained_path.read_text(encoding="utf-8"))
        if not report["smoke_only"] or not report["pretrained_backbone"] or report["backbone_weights"] != "ResNet50_Weights.IMAGENET1K_V2" or report["steps"] != 1:
            raise ValueError("Pretrained backbone smoke is invalid")
        if report["protocol_sha256"] != protocol_hash:
            raise ValueError("Stale protocol in pretrained backbone smoke")
        runs[PRETRAINED_RUN] = {key: report[key] for key in ("track", "method", "steps", "device", "gpu", "manifest_sha256", "protocol_sha256", "backbone_weights")}
        weights_path = PROJECT / "torch_cache/hub/checkpoints/resnet50-11ad3fa6.pth"
        if not weights_path.is_file():
            raise ValueError("Pretrained checkpoint file is missing")
        digest = hashlib.sha256()
        with weights_path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        pretrained_hash = digest.hexdigest()
        if pretrained_hash != "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca":
            raise ValueError("Pretrained checkpoint SHA-256 mismatch")
    evaluations = {}
    for name in EVALUATIONS:
        path = PROJECT / "reports/evaluations" / name
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report["smoke_only"] or report["publishable"] or report["num_images"] != 1:
            raise ValueError(f"Invalid target smoke result: {path}")
        if report["protocol_sha256"] != protocol_hash:
            raise ValueError(f"Stale protocol in {path}")
        evaluations[name] = {key: report[key] for key in ("track", "target", "num_images", "native_gt_resolution", "smoke_only", "publishable", "mapping_version", "manifest_hash", "protocol_sha256")}
    if not evaluations["r19_erm_mapillary_native_smoke.json"]["native_gt_resolution"]:
        raise ValueError("Mapillary native-resolution smoke missing")
    result = {"status": "passed", "model_runs": runs, "target_evaluations": evaluations, "pretrained_backbone_sha256": pretrained_hash, "note": "Engineering smoke only; no full-training benchmark scores."}
    destination = PROJECT / "reports/LOCAL_SMOKE_STATUS.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Verified {len(runs)} real-data optimization runs and {len(evaluations)} target evaluations")


if __name__ == "__main__":
    main()
