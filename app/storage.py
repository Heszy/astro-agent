"""DuckDB-backed catalog snapshots and provenance metadata."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from app.data_profiles import normalize_catalog


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NORMALIZER_VERSION = "1"


class CatalogStoreError(ValueError):
    """Raised when a catalog cannot be stored or resolved."""


def _quote_identifier(identifier: str) -> str:
    if not _IDENTIFIER.fullmatch(identifier):
        raise CatalogStoreError(f"Unsafe SQL identifier: {identifier}")
    return f'"{identifier}"'


class DuckDBCatalogStore:
    """Store immutable normalized catalog snapshots in an embedded DuckDB file."""

    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(self.database_path))
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT current_timestamp
            );
            CREATE TABLE IF NOT EXISTS dataset_versions (
                version_id VARCHAR PRIMARY KEY,
                dataset_id VARCHAR NOT NULL,
                version_label VARCHAR NOT NULL,
                table_name VARCHAR NOT NULL UNIQUE,
                source_path VARCHAR NOT NULL,
                source_sha256 VARCHAR NOT NULL,
                data_profile VARCHAR NOT NULL,
                provenance_json VARCHAR NOT NULL,
                row_count BIGINT NOT NULL,
                normalizer_version VARCHAR NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
                is_default BOOLEAN NOT NULL DEFAULT FALSE
            );
            """
        )

    @staticmethod
    def _source_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def ingest(
        self,
        source_path: Path,
        dataset_id: str,
        version_label: str | None = None,
        set_default: bool = True,
    ) -> dict[str, Any]:
        source_path = Path(source_path)
        if not source_path.is_file():
            raise CatalogStoreError(f"Catalog source not found: {source_path}")
        if not _IDENTIFIER.fullmatch(dataset_id):
            raise CatalogStoreError(
                "dataset_id must contain only letters, numbers and underscores"
            )

        source_sha256 = self._source_sha256(source_path)
        raw = pd.read_csv(source_path)
        try:
            normalized, profile, provenance = normalize_catalog(raw)
        except ValueError as exc:
            raise CatalogStoreError(str(exc)) from exc

        fingerprint = hashlib.sha256(
            f"{dataset_id}:{source_sha256}:{profile}:{_NORMALIZER_VERSION}".encode()
        ).hexdigest()
        version_id = fingerprint[:24]
        existing = self.connection.execute(
            "SELECT * FROM dataset_versions WHERE version_id = ?", [version_id]
        ).fetchone()
        if existing:
            return self._version_row_to_dict(existing)

        table_name = f"catalog_v_{version_id}"
        label = version_label or f"{datetime.now(UTC):%Y.%m.%d}-{version_id[:8]}"
        provenance_payload = {
            **provenance,
            "source_sha256": source_sha256,
            "normalizer_version": _NORMALIZER_VERSION,
        }
        self.connection.execute("BEGIN TRANSACTION")
        try:
            self.connection.execute(
                "INSERT OR IGNORE INTO datasets (dataset_id, name) VALUES (?, ?)",
                [dataset_id, dataset_id],
            )
            self.connection.register("normalized_catalog", normalized)
            self.connection.execute(
                f"CREATE TABLE {_quote_identifier(table_name)} AS SELECT * FROM normalized_catalog"
            )
            self.connection.unregister("normalized_catalog")
            if set_default:
                self.connection.execute(
                    "UPDATE dataset_versions SET is_default = FALSE WHERE dataset_id = ?",
                    [dataset_id],
                )
            self.connection.execute(
                """
                INSERT INTO dataset_versions (
                    version_id, dataset_id, version_label, table_name, source_path,
                    source_sha256, data_profile, provenance_json, row_count,
                    normalizer_version, is_default
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    version_id,
                    dataset_id,
                    label,
                    table_name,
                    str(source_path.resolve()),
                    source_sha256,
                    profile,
                    json.dumps(provenance_payload, ensure_ascii=False),
                    len(normalized),
                    _NORMALIZER_VERSION,
                    set_default,
                ],
            )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        return self.get_version(version_id)

    @staticmethod
    def _version_row_to_dict(row: tuple[Any, ...]) -> dict[str, Any]:
        keys = [
            "version_id",
            "dataset_id",
            "version_label",
            "table_name",
            "source_path",
            "source_sha256",
            "data_profile",
            "provenance_json",
            "row_count",
            "normalizer_version",
            "created_at",
            "is_default",
        ]
        result = dict(zip(keys, row, strict=True))
        result["provenance"] = json.loads(result.pop("provenance_json"))
        result["created_at"] = str(result["created_at"])
        return result

    def get_version(
        self, version_id: str | None = None, dataset_id: str | None = None
    ) -> dict[str, Any]:
        if version_id and version_id != "latest":
            row = self.connection.execute(
                """
                SELECT * FROM dataset_versions
                WHERE (version_id = ? OR version_label = ?)
                  AND (? IS NULL OR dataset_id = ?)
                """,
                [version_id, version_id, dataset_id, dataset_id],
            ).fetchone()
        else:
            row = self.connection.execute(
                """
                SELECT * FROM dataset_versions
                WHERE is_default = TRUE AND (? IS NULL OR dataset_id = ?)
                ORDER BY created_at DESC LIMIT 1
                """,
                [dataset_id, dataset_id],
            ).fetchone()
        if row is None:
            raise CatalogStoreError("No matching catalog version found")
        return self._version_row_to_dict(row)

    def list_versions(self, dataset_id: str | None = None) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT * FROM dataset_versions
            WHERE (? IS NULL OR dataset_id = ?)
            ORDER BY created_at DESC
            """,
            [dataset_id, dataset_id],
        ).fetchall()
        return [self._version_row_to_dict(row) for row in rows]

    def load_frame(
        self, dataset_id: str | None = None, version_id: str | None = None
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        metadata = self.get_version(version_id, dataset_id)
        table = _quote_identifier(metadata["table_name"])
        frame = self.connection.execute(f"SELECT * FROM {table}").df()
        return frame, metadata

    def schema(
        self, dataset_id: str | None = None, version_id: str | None = None
    ) -> dict[str, Any]:
        metadata = self.get_version(version_id, dataset_id)
        columns = self.connection.execute(
            f"PRAGMA table_info({_quote_identifier(metadata['table_name'])})"
        ).fetchall()
        return {
            **metadata,
            "columns": [
                {"name": row[1], "dtype": row[2], "nullable": not bool(row[3])}
                for row in columns
            ],
        }
