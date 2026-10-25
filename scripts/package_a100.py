"""Build a self-contained ZIP64 transfer bundle from audited archives and Git-tracked code."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DGSEG_DATA_ROOT", r"E:\ADA2\DGSEG_DATA"))
DEFAULT_OUTPUT = Path(r"D:\DGSEG_A100_bundle_2026-10-25.zip")
CHUNK = 8 * 1024 * 1024
ARCHIVES = (
    ("cityscapes/leftImg8bit_trainvaltest.zip", 11_592_327_197, "md5", "0a6e97e94b616a514066c9e2adb0c97f"),
    ("cityscapes/gtFine_trainvaltest.zip", 252_567_705, "md5", "4237c19de34c8a376e9ba46b495d6f66"),
    ("loveda/Train.zip", 4_021_669_263, "md5", "de2b196043ed9b4af1690b3f9a7d558f"),
    ("loveda/Val.zip", 2_425_958_254, "md5", "84cae2577468ff0b5386758bb386d31d"),
    ("mapillary_v1.2/mapillary-vistas-dataset_public_v1.2.zip", 25_624_067_065, "sha256", "61518267961941a440e3d561da2ed84ef1aa4980bb1feeeafbc5c546d120f24f"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    partial = output.with_suffix(output.suffix + ".part")
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or partial.exists() or sidecar.exists():
        raise FileExistsError(f"Refusing to overwrite an existing transfer artifact: {output}")

    total = 0
    for relative, expected_size, _, _ in ARCHIVES:
        source = DATA_ROOT / "downloads" / relative
        actual_size = source.stat().st_size
        if actual_size != expected_size:
            raise ValueError(f"Wrong size for {source}: {actual_size} != {expected_size}")
        total += actual_size
    output.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output.parent).free
    if free < total + 2 * 1024**3:
        raise OSError(f"Output disk needs {total + 2 * 1024**3:,} free bytes; has {free:,}")

    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=PROJECT).decode("utf-8").split("\0")
    tracked = [p for p in tracked if p]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    instruction = (
        "# A100 transfer bundle\n\n"
        "This ZIP contains the five complete, audited archives used by the current benchmark and the Git-tracked code. "
        "It does not contain incomplete SYNTHIA, BDD100K, ACDC, checkpoints, or smoke images.\n\n"
        "Extract with a ZIP64-capable tool. The resulting folders are `DGSEG_BENCHMARK/` and "
        "`DGSEG_DATA/downloads/`. Keep the dataset archives in their original ZIP form.\n\n"
        "```bash\n"
        "export DGSEG_DATA_ROOT=/path/to/extracted/DGSEG_DATA\n"
        "cd /path/to/extracted/DGSEG_BENCHMARK\n"
        "python3 -m unittest discover -s tests -v\n"
        "TRACKS=r19 METHODS='erm mixstyle randconv' SEEDS='0 1 2' bash scripts/run_a100.sh\n"
        "TRACKS='rs7_u2r rs7_r2u' METHODS='erm mixstyle randconv' SEEDS='0 1 2' bash scripts/run_a100.sh\n"
        "```\n\n"
        "Install matching PyTorch/torchvision CUDA wheels and `requirements.txt` before running. "
        "Follow the datasets' licenses when transferring and storing the archives.\n"
    )
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "code_commit": commit, "data_archives": []}

    print(f"Output: {output}", flush=True)
    print(f"Dataset bytes: {total:,}; free bytes: {free:,}", flush=True)
    try:
        with zipfile.ZipFile(partial, "w", allowZip64=True) as bundle:
            bundle.writestr("TRANSFER_README.md", instruction, compress_type=zipfile.ZIP_DEFLATED)
            for relative in tracked:
                bundle.write(PROJECT / relative, f"DGSEG_BENCHMARK/{relative}", compress_type=zipfile.ZIP_DEFLATED)
            print(f"Added {len(tracked)} Git-tracked code/report files", flush=True)

            for relative, expected_size, hash_kind, expected_hash in ARCHIVES:
                source = DATA_ROOT / "downloads" / relative
                destination = f"DGSEG_DATA/downloads/{relative}"
                verified = hashlib.new(hash_kind)
                sha256 = hashlib.sha256()
                info = zipfile.ZipInfo(destination)
                info.compress_type = zipfile.ZIP_STORED
                info.file_size = expected_size
                print(f"Adding {relative} ({expected_size:,} bytes)...", flush=True)
                copied = 0
                with source.open("rb") as reader, bundle.open(info, "w", force_zip64=True) as writer:
                    while block := reader.read(CHUNK):
                        writer.write(block)
                        verified.update(block)
                        sha256.update(block)
                        copied += len(block)
                        if copied // (4 * 1024**3) != (copied - len(block)) // (4 * 1024**3):
                            print(f"  {copied:,} / {expected_size:,} bytes", flush=True)
                if copied != expected_size or verified.hexdigest() != expected_hash:
                    raise ValueError(f"Hash/size mismatch while packing {relative}")
                manifest["data_archives"].append({"path": destination, "size": copied, "sha256": sha256.hexdigest(), "source_hash_kind": hash_kind, "source_hash": expected_hash})
                print(f"Verified {relative}: SHA-256 {sha256.hexdigest()}", flush=True)

            bundle.writestr("TRANSFER_MANIFEST.json", json.dumps(manifest, indent=2) + "\n", compress_type=zipfile.ZIP_DEFLATED)

        with zipfile.ZipFile(partial) as bundle:
            bad = bundle.infolist()
            expected_entries = len(tracked) + len(ARCHIVES) + 2
            if len(bad) != expected_entries:
                raise ValueError(f"ZIP central directory has {len(bad)} entries, expected {expected_entries}")
            for relative, expected_size, _, _ in ARCHIVES:
                info = bundle.getinfo(f"DGSEG_DATA/downloads/{relative}")
                if info.file_size != expected_size:
                    raise ValueError(f"ZIP entry has wrong size: {relative}")

        outer_hash = sha256_file(partial)
        partial.replace(output)
        sidecar.write_text(f"{outer_hash}  {output.name}\n", encoding="ascii")
        print(f"Complete: {output} ({output.stat().st_size:,} bytes)", flush=True)
        print(f"SHA-256: {outer_hash}", flush=True)
    except Exception:
        print(f"Packaging failed; incomplete file retained at {partial}", flush=True)
        raise


if __name__ == "__main__":
    main()
