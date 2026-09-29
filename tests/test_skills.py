from pathlib import Path

from app.catalog import CatalogAnalyzer
from app.skills import GroupedScalingSkill, ScalingRelationSkill, SkillRegistry
from app.tools import TOOL_SCHEMAS, build_tool_registry


def test_scaling_skill_has_fixed_auditable_steps(tmp_path: Path) -> None:
    analyzer = CatalogAnalyzer(Path("data/sample_catalog.csv"), tmp_path / "results")
    result = ScalingRelationSkill(analyzer).run(bootstrap=20, make_plot=True)

    assert result["skill"] == {
        "name": "scaling_relation_analysis",
        "version": "1.0.0",
    }
    assert result["status"] == "completed"
    assert [step["name"] for step in result["steps"]] == [
        "get_catalog_schema",
        "fit_scaling_relation",
        "make_plot",
    ]
    assert result["result"]["fit_method"] == "ols"
    assert result["result"]["sample_size"] == 720
    assert result["artifacts"][0]["type"] == "image/png"
    assert Path(result["artifacts"][0]["path"]).exists()


def test_grouped_skill_returns_one_skill_result(tmp_path: Path) -> None:
    analyzer = CatalogAnalyzer(Path("data/sample_catalog.csv"), tmp_path / "results")
    result = GroupedScalingSkill(analyzer).run(
        group_column="cloud_id",
        bootstrap=10,
    )

    assert result["skill"]["name"] == "grouped_scaling_analysis"
    assert result["status"] == "completed"
    assert result["steps"][1]["name"] == "plot_grouped_scaling_relation"
    assert result["artifacts"][0]["type"] == "image/png"


def test_skill_registry_is_exposed_as_high_level_tools(tmp_path: Path) -> None:
    analyzer = CatalogAnalyzer(Path("data/sample_catalog.csv"), tmp_path / "results")
    registry = build_tool_registry(analyzer)
    result = registry["run_scaling_relation_skill"](bootstrap=10)

    schema_names = {
        item["function"]["name"] for item in TOOL_SCHEMAS
    }
    assert "run_scaling_relation_skill" in schema_names
    assert "run_grouped_scaling_skill" in schema_names
    assert result["skill"]["name"] == "scaling_relation_analysis"
