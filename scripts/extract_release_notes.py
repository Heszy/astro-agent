"""Extract one version section from CHANGELOG.md for a GitHub Release."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def extract(changelog: str, version: str) -> str:
    escaped = re.escape(version.removeprefix("v"))
    pattern = re.compile(
        rf"^##\s+\[?v?{escaped}\]?[^\n]*\n(?P<body>.*?)(?=^##\s+|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(changelog)
    if not match:
        raise ValueError(f"No changelog section found for {version}")
    body = match.group("body").strip()
    if not re.search(r"^###\s+Highlights\s*$", body, re.MULTILINE):
        # Commitizen generates category headings from Conventional Commits. If a
        # release section has no manually curated Highlights heading, promote the
        # first few generated bullets so the GitHub Release remains user-facing.
        bullets = re.findall(r"^\s*-\s+.+$", body, re.MULTILINE)
        if not bullets:
            raise ValueError(f"Changelog section for {version} has no release entries")
        highlights = "### Highlights\n\n" + "\n".join(bullets[:3])
        body = highlights + "\n\n" + body
    return body + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Release version, for example v0.2.0")
    parser.add_argument("changelog", nargs="?", type=Path, default=Path("CHANGELOG.md"))
    parser.add_argument("output", nargs="?", type=Path, default=Path("release-notes.md"))
    args = parser.parse_args()

    args.output.write_text(
        extract(args.changelog.read_text(encoding="utf-8"), args.version),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
