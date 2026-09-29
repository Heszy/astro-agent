from __future__ import annotations

from typing import Any, Callable

from app.catalog import CatalogAnalyzer
from app.skills import GroupedScalingSkill, ScalingRelationSkill, SkillRegistry


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_catalog_schema",
            "description": "Return catalog fields, data types, physical meanings, units, and row count.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_structures",
            "description": "Filter molecular-cloud structures and return a small preview. Supported operators: eq, ne, gt, ge, lt, le, in.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filters": {
                        "type": "object",
                        "description": "Example: {\"hierarchy_level\": {\"eq\": 2}, \"mass_msun\": {\"gt\": 1000}}",
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fit_scaling_relation",
            "description": "Fit y versus x in log10 space using OLS or orthogonal distance regression (ODR), returning slope, uncertainty, correlation, and sample size.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "string"},
                    "y": {"type": "string"},
                    "filters": {"type": "object"},
                    "fit_method": {"type": "string", "enum": ["ols", "odr"]},
                    "x_error_column": {"type": "string"},
                    "y_error_column": {"type": "string"},
                    "bootstrap": {"type": "integer", "minimum": 0, "maximum": 2000},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_scaling_relations",
            "description": "Split the catalog by a numeric threshold and compare two log-log scaling relation slopes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "string"},
                    "y": {"type": "string"},
                    "group_column": {"type": "string"},
                    "split_value": {"type": "number"},
                    "fit_method": {"type": "string", "enum": ["ols", "odr"]},
                    "x_error_column": {"type": "string"},
                    "y_error_column": {"type": "string"},
                    "bootstrap": {"type": "integer", "minimum": 0, "maximum": 2000},
                },
                "required": ["x", "y", "group_column", "split_value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_plot",
            "description": "Create a log-log scatter plot and fitted scaling relation, saving it as a PNG.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "string"},
                    "y": {"type": "string"},
                    "filters": {"type": "object"},
                    "fit_method": {"type": "string", "enum": ["ols", "odr"]},
                    "x_error_column": {"type": "string"},
                    "y_error_column": {"type": "string"},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_grouped_scaling_relation",
            "description": "Group a catalog by a categorical field, fit one log-log scaling relation per group, and draw all groups in one PNG. Use for requests such as fitting by spiral arm.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "string"},
                    "y": {"type": "string"},
                    "group_column": {"type": "string"},
                    "filters": {"type": "object"},
                    "groups": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "fit_method": {"type": "string", "enum": ["ols", "odr"]},
                    "x_error_column": {"type": "string"},
                    "y_error_column": {"type": "string"},
                    "bootstrap": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2000,
                    },
                },
                "required": ["x", "y", "group_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_scaling_relation_skill",
            "description": "Run the deterministic scaling-relation workflow: inspect schema, fit one log-log relation, and optionally create a plot. Prefer this skill for standard scaling questions instead of planning low-level tool calls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "string"},
                    "y": {"type": "string"},
                    "filters": {"type": "object"},
                    "fit_method": {"type": "string", "enum": ["ols", "odr"]},
                    "x_error_column": {"type": "string"},
                    "y_error_column": {"type": "string"},
                    "bootstrap": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2000,
                    },
                    "make_plot": {"type": "boolean"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_grouped_scaling_skill",
            "description": "Run the deterministic grouped scaling workflow: inspect schema, fit one relation per categorical group, and create one combined plot. Prefer this skill for grouped or spiral-arm questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "string"},
                    "y": {"type": "string"},
                    "group_column": {"type": "string"},
                    "filters": {"type": "object"},
                    "groups": {"type": "array", "items": {"type": "string"}},
                    "fit_method": {"type": "string", "enum": ["ols", "odr"]},
                    "x_error_column": {"type": "string"},
                    "y_error_column": {"type": "string"},
                    "bootstrap": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2000,
                    },
                },
                "required": ["group_column"],
            },
        },
    },
]


def build_tool_registry(analyzer: CatalogAnalyzer) -> dict[str, Callable[..., dict[str, Any]]]:
    skill_registry = SkillRegistry(
        [ScalingRelationSkill(analyzer), GroupedScalingSkill(analyzer)]
    )
    return {
        "get_catalog_schema": analyzer.schema,
        "query_structures": analyzer.query,
        "fit_scaling_relation": analyzer.fit_scaling_relation,
        "compare_scaling_relations": analyzer.compare_scaling_relations,
        "make_plot": analyzer.make_plot,
        "plot_grouped_scaling_relation": analyzer.plot_grouped_scaling_relation,
        "run_scaling_relation_skill": lambda **arguments: skill_registry.execute(
            "scaling_relation_analysis", **arguments
        ),
        "run_grouped_scaling_skill": lambda **arguments: skill_registry.execute(
            "grouped_scaling_analysis", **arguments
        ),
    }
