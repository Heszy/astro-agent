from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from app.data_profiles import STANDARD_REQUIRED_COLUMNS, normalize_catalog

REQUIRED_COLUMNS = STANDARD_REQUIRED_COLUMNS

DATA_DICTIONARY = {
    "structure_id": "结构唯一编号",
    "parent_id": "父结构编号；根结构为空",
    "cloud_id": "所属分子云编号",
    "radius_pc": "等效半径，单位pc",
    "velocity_dispersion_kms": "速度弥散，单位km/s",
    "mass_msun": "质量，单位太阳质量",
    "column_density_cm2": "H2柱密度，单位cm^-2",
    "virial_parameter": "维里参数，无量纲",
    "hierarchy_level": "层级深度，根节点为0",
    "_idx": "该结构在所属 cloud 内的编号",
    "radius": "原始半径，单位 pc",
    "v_rms": "原始速度弥散，单位 m/s",
    "mass": "原始质量，单位太阳质量",
    "cloudidx": "该结构所属 cloud 的编号",
    "Nstru": "该结构包含的子结构数量",
    "Dist": "该结构的距离，单位 kpc",
    "arms": "该结构所属的旋臂",
    "touch": "该结构是否与 datacube 边缘相接；1 是，0 否",
    "child_structure_count": "子结构数量，由 Nstru 映射",
    "distance_kpc": "结构距离，单位 kpc，由 Dist 映射",
    "spiral_arm": "所属旋臂，由 arms 映射",
    "touches_datacube_edge": "是否与 datacube 边缘相接，由 touch 转为布尔值",
    "parent": "原目录中的 cloud 内父结构编号",
    "level": "原目录层级深度",
}

ALLOWED_OPERATORS = {"eq", "ne", "gt", "ge", "lt", "le", "in"}


class CatalogError(ValueError):
    pass


class CatalogAnalyzer:
    def __init__(self, data_path: Path, results_dir: Path):
        if not data_path.exists():
            raise CatalogError(
                f"Catalog not found: {data_path}. Run: python scripts/generate_sample_data.py"
            )
        self.data_path = data_path
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        raw = pd.read_csv(data_path)
        try:
            self.df, self.data_profile, self.data_provenance = normalize_catalog(raw)
        except ValueError as exc:
            raise CatalogError(str(exc)) from exc
        missing = REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise CatalogError(f"Missing required columns: {sorted(missing)}")

    def schema(self) -> dict[str, Any]:
        return {
            "rows": int(len(self.df)),
            "data_profile": self.data_profile,
            "data_provenance": self.data_provenance,
            "columns": [
                {
                    "name": column,
                    "dtype": str(self.df[column].dtype),
                    "description": DATA_DICTIONARY.get(column, "用户扩展字段"),
                }
                for column in self.df.columns
            ],
        }

    def _apply_filters(self, filters: dict[str, dict[str, Any]] | None) -> pd.DataFrame:
        frame = self.df.copy()
        for column, condition in (filters or {}).items():
            if column not in frame.columns:
                raise CatalogError(f"Unknown column: {column}")
            if not isinstance(condition, dict) or len(condition) != 1:
                raise CatalogError(
                    f"Filter for {column} must contain exactly one operator, e.g. {{'gt': 1.0}}"
                )
            operator, value = next(iter(condition.items()))
            if operator not in ALLOWED_OPERATORS:
                raise CatalogError(f"Unsupported operator: {operator}")
            series = frame[column]
            if operator == "eq":
                mask = series == value
            elif operator == "ne":
                mask = series != value
            elif operator == "gt":
                mask = series > value
            elif operator == "ge":
                mask = series >= value
            elif operator == "lt":
                mask = series < value
            elif operator == "le":
                mask = series <= value
            else:
                if not isinstance(value, list):
                    raise CatalogError("The 'in' operator requires a list")
                mask = series.isin(value)
            frame = frame.loc[mask]
        return frame

    def query(
        self, filters: dict[str, dict[str, Any]] | None = None, limit: int = 10
    ) -> dict[str, Any]:
        limit = max(1, min(int(limit), 50))
        frame = self._apply_filters(filters)
        preview_frame = frame.head(limit).astype(object)
        preview = preview_frame.where(pd.notnull(preview_frame), None).to_dict(
            orient="records"
        )
        return {
            "matched_rows": int(len(frame)),
            "returned_rows": int(len(preview)),
            "rows": preview,
        }

    def _fit_frame(
        self,
        x: str,
        y: str,
        filters: dict[str, dict[str, Any]] | None,
        bootstrap: int,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        for column in (x, y):
            if column not in self.df.columns:
                raise CatalogError(f"Unknown column: {column}")
        frame = self._apply_filters(filters)[[x, y]].dropna()
        frame = frame[(frame[x] > 0) & (frame[y] > 0)]
        if len(frame) < 8:
            raise CatalogError("At least 8 positive data points are required")

        log_x = np.log10(frame[x].to_numpy(dtype=float))
        log_y = np.log10(frame[y].to_numpy(dtype=float))
        result = stats.linregress(log_x, log_y)

        bootstrap = max(0, min(int(bootstrap), 2000))
        slopes: list[float] = []
        if bootstrap:
            rng = np.random.default_rng(20260927)
            n = len(frame)
            for _ in range(bootstrap):
                idx = rng.integers(0, n, n)
                if np.std(log_x[idx]) == 0:
                    continue
                slopes.append(float(stats.linregress(log_x[idx], log_y[idx]).slope))

        payload: dict[str, Any] = {
            "x": x,
            "y": y,
            "sample_size": int(len(frame)),
            "slope": float(result.slope),
            "intercept": float(result.intercept),
            "r_value": float(result.rvalue),
            "p_value": float(result.pvalue),
            "stderr": float(result.stderr),
            "bootstrap_iterations": int(len(slopes)),
        }
        if slopes:
            payload["slope_ci95"] = [
                float(np.percentile(slopes, 2.5)),
                float(np.percentile(slopes, 97.5)),
            ]
        return frame, payload

    def fit_scaling_relation(
        self,
        x: str = "radius_pc",
        y: str = "velocity_dispersion_kms",
        filters: dict[str, dict[str, Any]] | None = None,
        bootstrap: int = 300,
    ) -> dict[str, Any]:
        _, payload = self._fit_frame(x, y, filters, bootstrap)
        return payload

    def compare_scaling_relations(
        self,
        x: str,
        y: str,
        group_column: str,
        split_value: float,
        bootstrap: int = 300,
    ) -> dict[str, Any]:
        if group_column not in self.df.columns:
            raise CatalogError(f"Unknown group column: {group_column}")
        low_filters = {group_column: {"lt": split_value}}
        high_filters = {group_column: {"ge": split_value}}
        low = self.fit_scaling_relation(x, y, low_filters, bootstrap)
        high = self.fit_scaling_relation(x, y, high_filters, bootstrap)
        return {
            "group_column": group_column,
            "split_value": split_value,
            "low_group": low,
            "high_group": high,
            "slope_difference_high_minus_low": float(high["slope"] - low["slope"]),
            "note": "A slope difference alone is not a significance test; inspect both confidence intervals.",
        }

    def make_plot(
        self,
        x: str = "radius_pc",
        y: str = "velocity_dispersion_kms",
        filters: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        frame, fit = self._fit_frame(x, y, filters, bootstrap=0)
        x_values = frame[x].to_numpy(dtype=float)
        y_values = frame[y].to_numpy(dtype=float)
        grid = np.geomspace(x_values.min(), x_values.max(), 200)
        prediction = 10 ** fit["intercept"] * grid ** fit["slope"]

        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
        ax.scatter(x_values, y_values, s=13, alpha=0.55, edgecolors="none")
        ax.plot(grid, prediction, color="#b1123f", linewidth=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"log-log slope = {fit['slope']:.3f}, N = {fit['sample_size']}")
        ax.grid(alpha=0.2, which="both")
        fig.tight_layout()

        filename = f"scaling_{uuid.uuid4().hex[:8]}.png"
        output_path = self.results_dir / filename
        fig.savefig(output_path)
        plt.close(fig)
        return {"plot_path": str(output_path.resolve()), "fit": fit}
