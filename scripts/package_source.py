"""Create a portable code/protocol bundle without raw images or checkpoints."""

import hashlib
from pathlib import Path
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT.parent / "DGSEG_BENCHMARK_source_2026-10-25.zip"
PREFIXES = ("dgseg/", "scripts/", "configs/", "tests/", "reports/manifests/")
SINGLES = {"README.md", "requirements.txt", ".gitignore", "reports/benchmark_summary.csv", "reports/DATASET_AUDIT.md", "reports/LOCAL_SMOKE_STATUS.json"}


def main() -> None:
    paths = []
    for path in PROJECT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(PROJECT).as_posix()
        if "__pycache__" in path.parts or path.suffix in (".pt", ".pth", ".zip", ".rar"):
            continue
        if relative in SINGLES or relative.startswith(PREFIXES):
            paths.append((relative, path))
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for relative, path in sorted(paths):
            zf.write(path, relative)
    digest = hashlib.sha256()
    with OUTPUT.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    OUTPUT.with_suffix(OUTPUT.suffix + ".sha256").write_text(f"{digest.hexdigest()}  {OUTPUT.name}\n", encoding="ascii")
    print(f"{OUTPUT}\nfiles={len(paths)} bytes={OUTPUT.stat().st_size} sha256={digest.hexdigest()}")


if __name__ == "__main__":
    main()
