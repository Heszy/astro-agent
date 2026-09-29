from pathlib import Path

import pandas as pd

from app.catalog import CatalogAnalyzer
from app.storage import DuckDBCatalogStore


def _write_catalog(path: Path, rows: int = 8) -> None:
    pd.DataFrame(
        {
            "structure_id": list(range(rows)),
            "cloud_id": ["cloud-a"] * rows,
            "radius_pc": [float(index + 1) for index in range(rows)],
            "velocity_dispersion_kms": [float(index + 2) for index in range(rows)],
            "mass_msun": [100.0] * rows,
        }
    ).to_csv(path, index=False)


def test_ingest_is_idempotent_and_preserves_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "catalog.csv"
    database = tmp_path / "catalog.duckdb"
    _write_catalog(source)
    store = DuckDBCatalogStore(database)

    first = store.ingest(source, "default", "2026.09.29.1")
    second = store.ingest(source, "default", "2026.09.29.1")

    assert first["version_id"] == second["version_id"]
    assert first["source_sha256"]
    assert first["row_count"] == 8
    assert len(store.list_versions("default")) == 1


def test_new_source_content_creates_new_default_version(tmp_path: Path) -> None:
    source = tmp_path / "catalog.csv"
    database = tmp_path / "catalog.duckdb"
    _write_catalog(source)
    store = DuckDBCatalogStore(database)
    first = store.ingest(source, "default")

    frame = pd.read_csv(source)
    frame.loc[0, "mass_msun"] = 999.0
    frame.to_csv(source, index=False)
    second = store.ingest(source, "default")

    assert first["version_id"] != second["version_id"]
    assert store.get_version("latest", "default")["version_id"] == second["version_id"]
    assert len(store.list_versions("default")) == 2


def test_analyzer_can_load_a_duckdb_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "catalog.csv"
    database = tmp_path / "catalog.duckdb"
    _write_catalog(source)
    store = DuckDBCatalogStore(database)
    metadata = store.ingest(source, "default")

    analyzer = CatalogAnalyzer(
        None,
        tmp_path / "results",
        store=store,
        dataset_id="default",
        version_id=metadata["version_id"],
    )

    schema = analyzer.schema()
    assert schema["rows"] == 8
    assert schema["data_version"]["version_id"] == metadata["version_id"]
    assert analyzer.query(limit=2)["returned_rows"] == 2
