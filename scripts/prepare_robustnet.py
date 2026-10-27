"""Stage the official RobustNet Cityscapes loader from verified local archives.

No target-domain archive is opened here. Files are written outside this Git repo.
"""

import argparse
import os
from pathlib import Path
import subprocess
import zipfile


UPSTREAM = "https://github.com/sunghac/RobustNet.git"
COMMIT = "1ff5c047d7abaf9ab1a7cca23aff812b2f09de15"
ARCHIVES = {
    "leftImg8bit_trainvaltest.zip": "leftImg8bit_trainvaltest",
    "gtFine_trainvaltest.zip": "gtFine_trainvaltest",
}


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def safe_extract(archive: Path, destination: Path, small: bool) -> tuple[int, int]:
    count = 0
    bytes_written = 0
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if member.is_dir():
                continue
            if member.filename in {"README", "license.txt"}:
                continue
            parts = Path(member.filename).parts
            if not parts or parts[0] not in {"leftImg8bit", "gtFine"}:
                raise ValueError(f"Unexpected archive member: {member.filename}")
            if ".." in parts or Path(member.filename).is_absolute():
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if small and (len(parts) < 3 or parts[1] not in {"train", "val"}):
                continue
            target = destination.joinpath(*parts)
            if not target.exists() or target.stat().st_size != member.file_size:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as inp, target.open("wb") as out:
                    while chunk := inp.read(1024 * 1024):
                        out.write(chunk)
                if target.stat().st_size != member.file_size:
                    raise IOError(f"Incomplete extraction: {target}")
            count += 1
            bytes_written += member.file_size
    return count, bytes_written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DGSEG_DATA_ROOT", "_unset_data_root_")))
    parser.add_argument("--work-root", type=Path, required=True, help="Outside repository: official source and extracted Cityscapes")
    parser.add_argument("--extract", action="store_true", help="Extract source images and labels for official loader")
    args = parser.parse_args()
    if not args.data_root.joinpath("downloads").is_dir():
        parser.error("Set --data-root to the directory containing downloads/")
    work = args.work_root.resolve()
    work.mkdir(parents=True, exist_ok=True)
    vendor = work / "RobustNet"
    if not vendor.exists():
        git("clone", UPSTREAM, str(vendor))
    if git("rev-parse", "HEAD", cwd=vendor) != COMMIT:
        git("fetch", "origin", COMMIT, cwd=vendor)
        git("checkout", "--detach", COMMIT, cwd=vendor)
    actual = git("rev-parse", "HEAD", cwd=vendor)
    if actual != COMMIT:
        raise RuntimeError(f"Unexpected RobustNet revision: {actual}")
    config = vendor / "config.py"
    source = config.read_text(encoding="utf-8")
    old = "__C.DATASET.CITYSCAPES_DIR = '/data/davian/segmentation/cityscapes'"
    new = "__C.DATASET.CITYSCAPES_DIR = os.environ.get('ROBUSTNET_CITYSCAPES_ROOT', '/data/davian/segmentation/cityscapes')"
    if old in source:
        source = source.replace("import torch\n", "import torch\nimport os\n", 1).replace(old, new)
        config.write_text(source, encoding="utf-8")
    elif new not in source:
        raise RuntimeError("Unrecognized upstream config.py; inspect before patching")
    city = work / "cityscapes"
    if args.extract:
        for filename, subdir in ARCHIVES.items():
            archive = args.data_root / "downloads" / "cityscapes" / filename
            if not archive.is_file():
                raise FileNotFoundError(archive)
            count, size = safe_extract(archive, city / subdir, small=True)
            print(f"{filename}: {count} files, {size} uncompressed bytes", flush=True)
        image_count = len(list((city / "leftImg8bit_trainvaltest" / "leftImg8bit" / "train").glob("*/*_leftImg8bit.png")))
        mask_count = len(list((city / "gtFine_trainvaltest" / "gtFine" / "train").glob("*/*_gtFine_labelIds.png")))
        if image_count != 2975 or mask_count != 2975:
            raise RuntimeError(f"Expected 2975 Cityscapes train pairs, got {image_count} images and {mask_count} masks")
        print(f"Cityscapes train pairs: {image_count}; validation path present: {(city / 'leftImg8bit_trainvaltest' / 'leftImg8bit' / 'val').is_dir()}")
    print(f"RobustNet commit: {actual}")
    print(f"ROBUSTNET_CITYSCAPES_ROOT={city}")


if __name__ == "__main__":
    main()
