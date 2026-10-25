"""Split the verified A100 transfer ZIP into individually checksummed upload parts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil


SOURCE = Path(r"D:\DGSEG_A100_bundle_2026-10-25.zip")
OUTPUT = Path(r"E:\ADA2\DGSEG_A100_parts_2026-10-25")
PART_BYTES = 1_900_000_000  # Below common 2 GB per-file transfer limits.
BUFFER_BYTES = 8 * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--part-bytes", type=int, default=PART_BYTES)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if args.part_bytes < BUFFER_BYTES or args.part_bytes >= 2_000_000_000:
        raise ValueError("part-bytes must be at least 8 MiB and below 2,000,000,000")
    sidecar = source.with_suffix(source.suffix + ".sha256")
    expected = sidecar.read_text(encoding="ascii").split()[0].lower()
    if len(expected) != 64:
        raise ValueError("Invalid SHA-256 sidecar")
    source_bytes = source.stat().st_size
    needed = source_bytes + 1 * 1024**3
    free = shutil.disk_usage(output.parent).free
    if free < needed:
        raise OSError(f"Output needs {needed:,} free bytes, has {free:,}")
    output.mkdir(parents=False, exist_ok=False)

    parts = []
    source_hash = hashlib.sha256()
    print(f"Splitting {source} ({source_bytes:,} bytes) into {math.ceil(source_bytes / args.part_bytes)} parts", flush=True)
    with source.open("rb") as stream:
        index = 0
        while stream.tell() < source_bytes:
            name = f"{source.name}.part-{index:03d}"
            target = output / name
            pending = output / (name + ".partial")
            remaining = min(args.part_bytes, source_bytes - stream.tell())
            part_hash = hashlib.sha256()
            with pending.open("xb") as sink:
                while remaining:
                    block = stream.read(min(BUFFER_BYTES, remaining))
                    if not block:
                        raise EOFError("Source ZIP ended before its recorded size")
                    sink.write(block)
                    part_hash.update(block)
                    source_hash.update(block)
                    remaining -= len(block)
            pending.rename(target)
            parts.append({"file": name, "bytes": target.stat().st_size, "sha256": part_hash.hexdigest()})
            print(f"[{index + 1:02d}] {name}: {part_hash.hexdigest()}", flush=True)
            index += 1

    actual = source_hash.hexdigest()
    if actual != expected:
        raise ValueError(f"Source ZIP changed or sidecar is wrong: {actual} != {expected}")

    # Read the newly written parts back from disk and verify both every part and their concatenation.
    assembled = hashlib.sha256()
    for part in parts:
        path = output / part["file"]
        check = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(BUFFER_BYTES), b""):
                check.update(block)
                assembled.update(block)
        if check.hexdigest() != part["sha256"]:
            raise ValueError(f"On-disk part failed verification: {path}")
    if assembled.hexdigest() != expected:
        raise ValueError("Reassembled part stream does not match original ZIP")

    checksums = "".join(f"{part['sha256']}  {part['file']}\n" for part in parts)
    (output / "SHA256SUMS").write_text(checksums, encoding="ascii")
    (output / sidecar.name).write_text(sidecar.read_text(encoding="ascii"), encoding="ascii")
    manifest = {"source_file": source.name, "source_bytes": source_bytes, "source_sha256": expected,
                "part_bytes": args.part_bytes, "parts": parts}
    (output / "PARTS_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    readme = (
        "# A100 chunked transfer\n\n"
        "Copy every file in this directory to a single directory on A100. Each part is under 2 GB.\n"
        "From that directory, first verify the individual uploads:\n\n"
        "```bash\nsha256sum -c SHA256SUMS\n"
        f"cat {source.name}.part-* > {source.name}\n"
        f"sha256sum -c {sidecar.name}\n"
        f"unzip -t {source.name}\n```\n"
        "\nThe second check must print `OK` before extracting the bundle. "
        "If a part fails, re-upload only that part. The `cat` command overwrites any existing "
        "bundle of the same name in this directory, so run it only after all part checks pass. "
        "Allow enough space for both the parts and assembled ZIP, or assemble on a separate volume.\n"
    )
    (output / "README_A100.md").write_text(readme, encoding="utf-8")
    print(f"Verified all {len(parts)} parts and reassembled SHA-256: {expected}", flush=True)
    print(f"Transfer directory: {output}", flush=True)


if __name__ == "__main__":
    main()
