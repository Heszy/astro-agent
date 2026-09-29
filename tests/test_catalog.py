from pathlib import Path

import numpy as np
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


def test_grouped_scaling_plot_returns_each_group(tmp_path: Path) -> None:
    source = tmp_path / "grouped.csv"
    rows = []
    structure_id = 1
    for group, slope in (("loc", 0.4), ("out", 0.6), ("per", 0.8)):
        for radius in range(1, 11):
            radius_value = float(radius) / 10
            rows.append(
                {
                    "structure_id": structure_id,
                    "cloud_id": group,
                    "radius_pc": radius_value,
                    "velocity_dispersion_kms": radius_value**slope,
                    "spiral_arm": group,
                    "touches_datacube_edge": False,
                }
            )
            structure_id += 1
    pd.DataFrame(rows).to_csv(source, index=False)

    analyzer = CatalogAnalyzer(source, tmp_path / "results")
    result = analyzer.plot_grouped_scaling_relation(
        filters={"touches_datacube_edge": {"eq": False}},
        bootstrap=30,
    )

    assert Path(result["plot_path"]).exists()
    assert result["plotted_groups"] == ["loc", "out", "per"]
    assert set(result["groups"]) == {"loc", "out", "per"}
    assert all(result["groups"][group]["sample_size"] == 10 for group in result["groups"])


def test_odr_fit_supports_optional_measurement_uncertainties(tmp_path: Path) -> None:
    source = tmp_path / "odr.csv"
    radius = np.geomspace(0.1, 10.0, 40)
    pd.DataFrame(
        {
            "structure_id": range(len(radius)),
            "cloud_id": "cloud",
            "radius_pc": radius,
            "velocity_dispersion_kms": 0.7 * radius**0.55,
            "radius_error_pc": radius * 0.03,
            "velocity_error_kms": radius**0.55 * 0.02,
        }
    ).to_csv(source, index=False)

    analyzer = CatalogAnalyzer(source, tmp_path / "results")
    result = analyzer.fit_scaling_relation(
        fit_method="odr",
        x_error_column="radius_error_pc",
        y_error_column="velocity_error_kms",
        bootstrap=20,
    )

    assert result["fit_method"] == "odr"
    assert result["sample_size"] == 40
    assert result["uncertainty_columns"] == {
        "x": "radius_error_pc",
        "y": "velocity_error_kms",
    }
    assert result["diagnostics"]["weighted"] is True
    assert result["diagnostics"]["converged"] is True
    assert len(result["slope_ci95"]) == 2


def test_invalid_fit_method_is_rejected(analyzer: CatalogAnalyzer) -> None:
    with pytest.raises(CatalogError, match="fit_method"):
        analyzer.fit_scaling_relation(fit_method="robust")
