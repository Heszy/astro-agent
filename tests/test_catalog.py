from pathlib import Path

import pandas as pd
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


def test_newtrunks_profile_maps_units_and_hierarchy(tmp_path: Path) -> None:
    source = tmp_path / "newtrunks.csv"
    pd.DataFrame(
        {
            "_idx": [0, 1],
            "cloudidx": [10, 10],
            "parent": [None, 0],
            "radius": [1.5, 0.5],
            "v_rms": [250.0, 125.0],
            "mass": [100.0, 20.0],
            "vp": [1.2, 0.8],
            "level": [0, 1],
            "Nstru": [1, 0],
            "Dist": [2.5, 2.5],
            "arms": ["loc", "loc"],
            "touch": [0, 1],
        }
    ).to_csv(source, index=False)

    mapped = CatalogAnalyzer(source, tmp_path / "results")

    assert mapped.data_profile == "newtrunks"
    assert mapped.df["structure_id"].tolist() == ["10:0", "10:1"]
    assert mapped.df["parent_id"].tolist() == [None, "10:0"]
    assert mapped.df["velocity_dispersion_kms"].tolist() == [0.25, 0.125]
    assert mapped.df["radius_pc"].tolist() == [1.5, 0.5]
    assert mapped.df["child_structure_count"].tolist() == [1, 0]
    assert mapped.df["distance_kpc"].tolist() == [2.5, 2.5]
    assert mapped.df["spiral_arm"].tolist() == ["loc", "loc"]
    assert mapped.df["touches_datacube_edge"].tolist() == [False, True]
    assert mapped.data_provenance["quality"] == {
        "source_rows": 2,
        "non_null_parent_references": 1,
        "parent_references_missing_from_table": 0,
    }
