from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.data_profiles import STANDARD_REQUIRED_COLUMNS, normalize_catalog
from app.fitting import FitError, fit_with_bootstrap, validate_fit_method

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
        fit_method: str = "ols",
        x_error_column: str | None = None,
        y_error_column: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        try:
            method = validate_fit_method(fit_method)
        except FitError as exc:
            raise CatalogError(str(exc)) from exc
        for column in (x, y):
            if column not in self.df.columns:
                raise CatalogError(f"Unknown column: {column}")
        if method == "ols" and (x_error_column or y_error_column):
            raise CatalogError("Measurement uncertainties can only be used with ODR")
        error_columns = [
            column for column in (x_error_column, y_error_column) if column is not None
        ]
        for column in error_columns:
            if column not in self.df.columns:
                raise CatalogError(f"Unknown uncertainty column: {column}")

        selected_columns = [x, y, *error_columns]
        frame = self._apply_filters(filters)[selected_columns].copy()
        for column in selected_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        rows_before = len(frame)
        valid = frame[x].notna() & frame[y].notna() & (frame[x] > 0) & (frame[y] > 0)
        for column in error_columns:
            valid &= frame[column].notna() & (frame[column] > 0)
        frame = frame.loc[valid]
        if len(frame) < 8:
            raise CatalogError("At least 8 positive data points are required")

        log_x = np.log10(frame[x].to_numpy(dtype=float))
        log_y = np.log10(frame[y].to_numpy(dtype=float))
        sigma_x = None
        sigma_y = None
        if x_error_column:
            sigma_x = frame[x_error_column].to_numpy(dtype=float) / (
                frame[x].to_numpy(dtype=float) * np.log(10.0)
            )
        if y_error_column:
            sigma_y = frame[y_error_column].to_numpy(dtype=float) / (
                frame[y].to_numpy(dtype=float) * np.log(10.0)
            )
        try:
            fit, slopes = fit_with_bootstrap(
                log_x,
                log_y,
                fit_method=method,
                sigma_x=sigma_x,
                sigma_y=sigma_y,
                bootstrap=bootstrap,
            )
        except FitError as exc:
            raise CatalogError(str(exc)) from exc

        payload: dict[str, Any] = {
            "x": x,
            "y": y,
            "fit_method": method,
            "fit_space": "log10",
            "sample_size": int(len(frame)),
            "rows_dropped_during_preparation": int(rows_before - len(frame)),
            "uncertainty_columns": {"x": x_error_column, "y": y_error_column},
            **fit,
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
        fit_method: str = "ols",
        x_error_column: str | None = None,
        y_error_column: str | None = None,
    ) -> dict[str, Any]:
        _, payload = self._fit_frame(
            x,
            y,
            filters,
            bootstrap,
            fit_method,
            x_error_column,
            y_error_column,
        )
        if fit_method == "odr" and not (x_error_column or y_error_column):
            payload.setdefault("warnings", []).append(
                "ODR was run without measurement uncertainties."
            )
        return payload

    def compare_scaling_relations(
        self,
        x: str,
        y: str,
        group_column: str,
        split_value: float,
        bootstrap: int = 300,
        fit_method: str = "ols",
        x_error_column: str | None = None,
        y_error_column: str | None = None,
    ) -> dict[str, Any]:
        if group_column not in self.df.columns:
            raise CatalogError(f"Unknown group column: {group_column}")
        low_filters = {group_column: {"lt": split_value}}
        high_filters = {group_column: {"ge": split_value}}
        low = self.fit_scaling_relation(
            x,
            y,
            low_filters,
            bootstrap,
            fit_method,
            x_error_column,
            y_error_column,
        )
        high = self.fit_scaling_relation(
            x,
            y,
            high_filters,
            bootstrap,
            fit_method,
            x_error_column,
            y_error_column,
        )
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
        fit_method: str = "ols",
        x_error_column: str | None = None,
        y_error_column: str | None = None,
    ) -> dict[str, Any]:
        frame, fit = self._fit_frame(
            x,
            y,
            filters,
            bootstrap=0,
            fit_method=fit_method,
            x_error_column=x_error_column,
            y_error_column=y_error_column,
        )
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
        ax.set_title(
            f"{fit['fit_method'].upper()} log-log slope = "
            f"{fit['slope']:.3f}, N = {fit['sample_size']}"
        )
        ax.grid(alpha=0.2, which="both")
        fig.tight_layout()

        filename = f"scaling_{fit['fit_method']}_{uuid.uuid4().hex[:8]}.png"
        output_path = self.results_dir / filename
        fig.savefig(output_path)
        plt.close(fig)
        return {"plot_path": str(output_path.resolve()), "fit": fit}

    def plot_grouped_scaling_relation(
        self,
        x: str = "radius_pc",
        y: str = "velocity_dispersion_kms",
        group_column: str = "spiral_arm",
        filters: dict[str, dict[str, Any]] | None = None,
        groups: list[str] | None = None,
        bootstrap: int = 300,
        fit_method: str = "ols",
        x_error_column: str | None = None,
        y_error_column: str | None = None,
    ) -> dict[str, Any]:
        """Fit and plot one log-log relation per categorical group."""

        for column in (x, y, group_column):
            if column not in self.df.columns:
                raise CatalogError(f"Unknown column: {column}")
        if filters and group_column in filters:
            raise CatalogError(
                f"Do not filter {group_column} directly; use the 'groups' argument."
            )

        base_frame = self._apply_filters(filters)
        available_groups = (
            base_frame[group_column].dropna().astype(str).sort_values().unique().tolist()
        )
        group_values = available_groups if groups is None else [str(group) for group in groups]
        if not group_values:
            raise CatalogError("No non-null groups remain after filtering")

        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        colors = plt.get_cmap("tab10")
        group_results: dict[str, dict[str, Any]] = {}
        plotted_groups: list[str] = []

        for index, group in enumerate(group_values):
            group_filters = dict(filters or {})
            group_filters[group_column] = {"eq": group}
            try:
                group_frame, fit = self._fit_frame(
                    x,
                    y,
                    group_filters,
                    bootstrap,
                    fit_method,
                    x_error_column,
                    y_error_column,
                )
            except CatalogError as exc:
                group_results[group] = {
                    "status": "skipped",
                    "sample_size": int(
                        len(base_frame[base_frame[group_column].astype(str) == group])
                    ),
                    "reason": str(exc),
                }
                continue

            color = colors(index % 10)
            x_values = group_frame[x].to_numpy(dtype=float)
            y_values = group_frame[y].to_numpy(dtype=float)
            grid = np.geomspace(x_values.min(), x_values.max(), 200)
            prediction = 10 ** fit["intercept"] * grid ** fit["slope"]
            ax.scatter(
                x_values,
                y_values,
                s=16,
                alpha=0.45,
                color=color,
                edgecolors="none",
            )
            ci = fit.get("slope_ci95")
            ci_text = "" if ci is None else f", 95% CI=[{ci[0]:.3f}, {ci[1]:.3f}]"
            ax.plot(
                grid,
                prediction,
                color=color,
                linewidth=2,
                label=(
                    f"{group}: N={fit['sample_size']}, "
                    f"slope={fit['slope']:.3f}{ci_text}"
                ),
            )
            group_results[group] = {"status": "ok", **fit}
            plotted_groups.append(group)

        if not plotted_groups:
            plt.close(fig)
            raise CatalogError("No group has at least 8 positive data points")

        ax.set_xscale("log")
        ax.set_yscale("log")
        axis_labels = {
            "radius_pc": "Radius (pc)",
            "velocity_dispersion_kms": "Velocity dispersion (km/s)",
        }
        ax.set_xlabel(axis_labels.get(x, x))
        ax.set_ylabel(axis_labels.get(y, y))
        ax.set_title(
            f"{fit_method.upper()} log-log scaling relation by {group_column}"
        )
        ax.legend()
        ax.grid(alpha=0.2, which="both")
        fig.tight_layout()

        filename = (
            f"grouped_{fit_method}_{x}_vs_{y}_by_{group_column}_"
            f"{uuid.uuid4().hex[:8]}.png"
        )
        output_path = self.results_dir / filename
        fig.savefig(output_path)
        plt.close(fig)
        return {
            "plot_path": str(output_path.resolve()),
            "x": x,
            "y": y,
            "group_column": group_column,
            "groups": group_results,
            "plotted_groups": plotted_groups,
            "filters": filters or {},
            "fit_method": fit_method,
            "uncertainty_columns": {"x": x_error_column, "y": y_error_column},
            "warnings": (
                ["ODR was run without measurement uncertainties."]
                if fit_method == "odr" and not (x_error_column or y_error_column)
                else []
            ),
        }
