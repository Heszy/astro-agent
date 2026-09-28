from __future__ import annotations

from typing import Any

import pandas as pd


STANDARD_REQUIRED_COLUMNS = {
    "structure_id",
    "cloud_id",
    "radius_pc",
    "velocity_dispersion_kms",
}

NEWTRUNKS_REQUIRED_COLUMNS = {
    "_idx",
    "cloudidx",
    "parent",
    "radius",
    "v_rms",
    "mass",
    "vp",
    "level",
    "Nstru",
    "Dist",
    "arms",
    "touch",
}


def _qualified_id(cloud_id: pd.Series, local_id: pd.Series) -> pd.Series:
    cloud = cloud_id.astype("Int64").astype(str)
    local = local_id.astype("Int64").astype(str)
    return cloud + ":" + local


def _normalize_newtrunks(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the Agent's canonical fields without discarding source columns.

    newtrunks stores ``v_rms`` in m/s.  The Agent's canonical velocity field
    is km/s, so this adapter applies the explicit factor 1e-3.
    ``_idx`` is only unique within a cloud; IDs are therefore qualified by
    ``cloudidx``. Parent IDs use the same namespace, even when the referenced
    parent was excluded from the supplied table.
    """

    normalized = frame.drop(
        columns=[column for column in frame if column.startswith("Unnamed:")]
    ).copy()
    normalized["structure_id"] = _qualified_id(frame["cloudidx"], frame["_idx"])
    normalized["cloud_id"] = frame["cloudidx"].astype("Int64").astype(str)
    normalized["parent_id"] = None
    has_parent = frame["parent"].notna()
    normalized.loc[has_parent, "parent_id"] = _qualified_id(
        frame.loc[has_parent, "cloudidx"], frame.loc[has_parent, "parent"]
    )
    normalized["radius_pc"] = pd.to_numeric(frame["radius"], errors="coerce")
    normalized["velocity_dispersion_kms"] = (
        pd.to_numeric(frame["v_rms"], errors="coerce") / 1000.0
    )
    normalized["mass_msun"] = pd.to_numeric(frame["mass"], errors="coerce")
    normalized["virial_parameter"] = pd.to_numeric(frame["vp"], errors="coerce")
    normalized["hierarchy_level"] = pd.to_numeric(
        frame["level"], errors="coerce"
    ).astype("Int64")
    normalized["child_structure_count"] = pd.to_numeric(
        frame["Nstru"], errors="coerce"
    ).astype("Int64")
    normalized["distance_kpc"] = pd.to_numeric(frame["Dist"], errors="coerce")
    normalized["spiral_arm"] = frame["arms"].astype("string")
    touch = pd.to_numeric(frame["touch"], errors="coerce")
    invalid_touch = touch.dropna()[~touch.dropna().isin([0, 1])]
    if not invalid_touch.empty:
        raise ValueError("newtrunks column 'touch' must contain only 0, 1 or blanks")
    normalized["touches_datacube_edge"] = touch.map({0: False, 1: True}).astype(
        "boolean"
    )
    return normalized


def normalize_catalog(frame: pd.DataFrame) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """Detect a supported catalog layout and return canonical analysis fields."""

    columns = set(frame.columns)
    if STANDARD_REQUIRED_COLUMNS <= columns:
        return frame.copy(), "standard", {"transformations": []}
    if NEWTRUNKS_REQUIRED_COLUMNS <= columns:
        normalized = _normalize_newtrunks(frame)
        present_ids = set(normalized["structure_id"])
        parent_ids = normalized["parent_id"].dropna()
        missing_parent_references = int((~parent_ids.isin(present_ids)).sum())
        return (
            normalized,
            "newtrunks",
            {
                "transformations": [
                    "structure_id = cloudidx:_idx",
                    "parent_id = cloudidx:parent",
                    "radius_pc = radius",
                    "velocity_dispersion_kms = v_rms / 1000",
                    "mass_msun = mass",
                    "virial_parameter = vp",
                    "hierarchy_level = level",
                    "child_structure_count = Nstru",
                    "distance_kpc = Dist",
                    "spiral_arm = arms",
                    "touches_datacube_edge = bool(touch)",
                ],
                "source_definitions": {
                    "_idx": "structure number within its cloud",
                    "radius": "radius in pc",
                    "v_rms": "velocity dispersion in m/s",
                    "mass": "mass in solar masses",
                    "cloudidx": "cloud number",
                    "Nstru": "number of child structures",
                    "Dist": "structure distance in kpc",
                    "arms": "spiral-arm membership",
                    "touch": "whether the structure touches the datacube edge",
                },
                "quality": {
                    "source_rows": int(len(frame)),
                    "non_null_parent_references": int(len(parent_ids)),
                    "parent_references_missing_from_table": missing_parent_references,
                },
            },
        )

    standard_missing = sorted(STANDARD_REQUIRED_COLUMNS - columns)
    newtrunks_missing = sorted(NEWTRUNKS_REQUIRED_COLUMNS - columns)
    raise ValueError(
        "Unsupported catalog columns. "
        f"Standard profile is missing {standard_missing}; "
        f"newtrunks profile is missing {newtrunks_missing}."
    )
