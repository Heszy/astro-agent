"""Import a normalized catalog snapshot into DuckDB."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.storage import DuckDBCatalogStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--version-label")
    parser.add_argument("--db", type=Path, default=Path("data/astro_catalog.duckdb"))
    args = parser.parse_args()

    store = DuckDBCatalogStore(args.db)
    metadata = store.ingest(args.source, args.dataset, args.version_label)
    print(f"Imported {metadata['row_count']} rows")
    print(f"dataset={metadata['dataset_id']}")
    print(f"version={metadata['version_id']} ({metadata['version_label']})")
    print(f"source_sha256={metadata['source_sha256']}")


if __name__ == "__main__":
    main()
