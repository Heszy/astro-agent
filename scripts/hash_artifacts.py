"""Write SHA256SUMS for distribution artifacts in a dist directory."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", nargs="?", type=Path, default=Path("dist"))
    args = parser.parse_args(argv)

    if not args.dist.is_dir():
        raise SystemExit(f"Distribution directory does not exist: {args.dist}")

    artifacts = sorted(
        path
        for path in args.dist.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    if not artifacts:
        raise SystemExit(f"No distribution artifacts found in {args.dist}")

    manifest = args.dist / "SHA256SUMS"
    manifest.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in artifacts),
        encoding="utf-8",
    )
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
