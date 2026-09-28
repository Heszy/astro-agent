from pathlib import Path

import pytest

from app.catalog import CatalogAnalyzer, CatalogError
from scripts.generate_sample_data import main as generate_sample_data


@pytest.fixture(scope="module")
def analyzer(tmp_path_factory) -> CatalogAnalyzer:
    # The generator writes relative to cwd, so normal project execution creates the sample first.
    if not Path("data/sample_catalog.csv").exists():
        generate_sample_data()
    return CatalogAnalyzer(Path("data/sample_catalog.csv"), tmp_path_factory.mktemp("results"))


def test_schema(analyzer: CatalogAnalyzer) -> None:
    schema = analyzer.schema()
    assert schema["rows"] == 720
    assert any(item["name"] == "radius_pc" for item in schema["columns"])


def test_query_filters(analyzer: CatalogAnalyzer) -> None:
    result = analyzer.query({"hierarchy_level": {"eq": 2}}, limit=5)
    assert result["matched_rows"] > 0
    assert len(result["rows"]) <= 5
    assert all(row["hierarchy_level"] == 2 for row in result["rows"])


def test_fit_relation(analyzer: CatalogAnalyzer) -> None:
    result = analyzer.fit_scaling_relation(bootstrap=30)
    assert 0.2 < result["slope"] < 0.9
    assert result["sample_size"] == 720
    assert len(result["slope_ci95"]) == 2


def test_reject_unknown_column(analyzer: CatalogAnalyzer) -> None:
    with pytest.raises(CatalogError):
        analyzer.query({"does_not_exist": {"gt": 1}})

