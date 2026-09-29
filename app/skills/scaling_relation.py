from __future__ import annotations

from pathlib import Path
from typing import Any

from app.catalog import CatalogAnalyzer
from app.skills.base import SkillExecution


class ScalingRelationSkill:
    name = "scaling_relation_analysis"
    version = "1.0.0"

    def __init__(self, analyzer: CatalogAnalyzer):
        self.analyzer = analyzer

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": (
                "Inspect the catalog, fit one log-log scaling relation, "
                "and optionally create a plot."
            ),
        }

    def run(
        self,
        x: str = "radius_pc",
        y: str = "velocity_dispersion_kms",
        filters: dict[str, dict[str, Any]] | None = None,
        fit_method: str = "ols",
        x_error_column: str | None = None,
        y_error_column: str | None = None,
        bootstrap: int = 300,
        make_plot: bool = False,
    ) -> dict[str, Any]:
        steps: list[dict[str, Any]] = []
        schema = self.analyzer.schema()
        steps.append(
            {
                "name": "get_catalog_schema",
                "status": "ok",
                "rows": schema["rows"],
                "data_profile": schema["data_profile"],
            }
        )

        fit = self.analyzer.fit_scaling_relation(
            x=x,
            y=y,
            filters=filters,
            fit_method=fit_method,
            x_error_column=x_error_column,
            y_error_column=y_error_column,
            bootstrap=bootstrap,
        )
        steps.append(
            {
                "name": "fit_scaling_relation",
                "status": "ok",
                "fit_method": fit["fit_method"],
                "sample_size": fit["sample_size"],
            }
        )

        artifacts: list[dict[str, Any]] = []
        if make_plot:
            plot = self.analyzer.make_plot(
                x=x,
                y=y,
                filters=filters,
                fit_method=fit_method,
                x_error_column=x_error_column,
                y_error_column=y_error_column,
            )
            plot_path = Path(plot["plot_path"])
            artifacts.append(
                {
                    "type": "image/png",
                    "path": str(plot_path),
                    "name": plot_path.name,
                }
            )
            steps.append(
                {
                    "name": "make_plot",
                    "status": "ok",
                    "path": str(plot_path),
                }
            )

        warnings = list(fit.get("warnings", []))
        return SkillExecution(
            skill=self.name,
            version=self.version,
            status="completed",
            steps=steps,
            result=fit,
            artifacts=artifacts,
            warnings=warnings,
        ).as_dict()


class GroupedScalingSkill:
    name = "grouped_scaling_analysis"
    version = "1.0.0"

    def __init__(self, analyzer: CatalogAnalyzer):
        self.analyzer = analyzer

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": (
                "Inspect the catalog, fit one scaling relation per categorical "
                "group, and draw all groups in one plot."
            ),
        }

    def run(
        self,
        x: str = "radius_pc",
        y: str = "velocity_dispersion_kms",
        group_column: str = "spiral_arm",
        filters: dict[str, dict[str, Any]] | None = None,
        groups: list[str] | None = None,
        fit_method: str = "ols",
        x_error_column: str | None = None,
        y_error_column: str | None = None,
        bootstrap: int = 300,
    ) -> dict[str, Any]:
        schema = self.analyzer.schema()
        plot = self.analyzer.plot_grouped_scaling_relation(
            x=x,
            y=y,
            group_column=group_column,
            filters=filters,
            groups=groups,
            fit_method=fit_method,
            x_error_column=x_error_column,
            y_error_column=y_error_column,
            bootstrap=bootstrap,
        )
        plot_path = Path(plot["plot_path"])
        steps = [
            {
                "name": "get_catalog_schema",
                "status": "ok",
                "rows": schema["rows"],
                "data_profile": schema["data_profile"],
            },
            {
                "name": "plot_grouped_scaling_relation",
                "status": "ok",
                "group_column": group_column,
                "fit_method": fit_method,
                "plotted_groups": plot["plotted_groups"],
            },
        ]
        return SkillExecution(
            skill=self.name,
            version=self.version,
            status="completed",
            steps=steps,
            result=plot,
            artifacts=[
                {
                    "type": "image/png",
                    "path": str(plot_path),
                    "name": plot_path.name,
                }
            ],
            warnings=list(plot.get("warnings", [])),
        ).as_dict()
