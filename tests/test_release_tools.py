from pathlib import Path

import pytest

from scripts.extract_release_notes import extract
from scripts.hash_artifacts import main as hash_main


def test_extract_release_notes_requires_highlights() -> None:
    changelog = """# Changelog

## [0.2.0] - 2026-09-29

### Highlights

- A user-facing improvement.

### Fixed

- A bug.
"""

    assert "A user-facing improvement." in extract(changelog, "v0.2.0")


def test_extract_release_notes_rejects_missing_version() -> None:
    with pytest.raises(ValueError, match="No changelog section"):
        extract("## [0.1.0]\n\n### Highlights\n- Old\n", "v0.2.0")


def test_hash_artifacts_writes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    artifact = dist / "example.whl"
    artifact.write_bytes(b"astro-agent")
    monkeypatch.chdir(tmp_path)

    hash_main([])

    manifest = (dist / "SHA256SUMS").read_text(encoding="utf-8")
    assert "example.whl" in manifest
    assert len(manifest.split()[0]) == 64
