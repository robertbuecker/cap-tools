from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


SETTINGS_VERSION = 1

DEFAULT_OVERALL_COLUMNS = [
    "name",
    "N used",
    "complete",
    "redundancy",
    "F2/sig(F2)",
    "Rurim",
    "Rpim",
    "CC1/2",
    "R1_gt",
    "wR_ref",
    "GOOF",
]
DEFAULT_SHELL_COLUMNS = [
    "dmin",
    "dmax",
    "complete",
    "redundancy",
    "F2/sig(F2)",
    "Rurim",
    "Rpim",
    "CC1/2",
    "deltaCC",
]
DEFAULT_SHELL_PLOTS = ["CC1/2", "complete"]
DEFAULT_RADAR_METRICS = [
    "complete",
    "F2/sig(F2)",
    "Rurim",
    "Rpim",
    "CC1/2",
    "redundancy",
    "R1_gt",
    "wR_ref",
]

LOWER_IS_BETTER = {
    "Rint",
    "Rurim",
    "Rpim",
    "RsigmaB",
    "R1_gt",
    "wR_ref",
}

DISPLAY_LABELS = {
    "name": "Finalization",
    "N inputs": "N total",
    "N used": "N used",
    "complete": "Comp",
    "redundancy": "Red",
    "F2/sig(F2)": "I/sig",
    "R1_gt": "R1",
    "wR_ref": "wR2",
}


def default_directions(fields: Iterable[str]) -> Dict[str, str]:
    return {field: ("lower" if field in LOWER_IS_BETTER else "higher") for field in fields}


@dataclass
class ViewSettings:
    overall_columns: List[str] = field(default_factory=lambda: DEFAULT_OVERALL_COLUMNS.copy())
    shell_columns: List[str] = field(default_factory=lambda: DEFAULT_SHELL_COLUMNS.copy())
    shell_plots: List[str] = field(default_factory=lambda: DEFAULT_SHELL_PLOTS.copy())
    radar_metrics: List[str] = field(default_factory=lambda: DEFAULT_RADAR_METRICS.copy())
    radar_directions: Dict[str, str] = field(default_factory=lambda: default_directions(DEFAULT_RADAR_METRICS))

    def to_dict(self) -> dict:
        return {
            "version": SETTINGS_VERSION,
            "overall_columns": list(self.overall_columns),
            "shell_columns": list(self.shell_columns),
            "shell_plots": list(self.shell_plots),
            "radar_metrics": list(self.radar_metrics),
            "radar_directions": dict(self.radar_directions),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ViewSettings":
        if raw.get("version") != SETTINGS_VERSION:
            raise ValueError("Unsupported finalization-viewer settings version")

        def string_list(name: str, default: Sequence[str]) -> List[str]:
            value = raw.get(name, default)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"Invalid {name} setting")
            return list(dict.fromkeys(value))

        directions = raw.get("radar_directions", {})
        if not isinstance(directions, dict):
            raise ValueError("Invalid radar_directions setting")
        clean_directions = {
            str(key): str(value)
            for key, value in directions.items()
            if value in {"higher", "lower"}
        }
        result = cls(
            overall_columns=string_list("overall_columns", DEFAULT_OVERALL_COLUMNS),
            shell_columns=string_list("shell_columns", DEFAULT_SHELL_COLUMNS),
            shell_plots=string_list("shell_plots", DEFAULT_SHELL_PLOTS),
            radar_metrics=string_list("radar_metrics", DEFAULT_RADAR_METRICS),
            radar_directions=clean_directions,
        )
        for metric, direction in default_directions(result.radar_metrics).items():
            result.radar_directions.setdefault(metric, direction)
        return result

    def reconcile(
        self,
        *,
        overall_fields: Sequence[str],
        shell_fields: Sequence[str],
        radar_fields: Sequence[str],
    ) -> "ViewSettings":
        """Retain saved preferences and add newly introduced application defaults."""
        for value in DEFAULT_OVERALL_COLUMNS:
            if value in overall_fields and value not in self.overall_columns:
                self.overall_columns.append(value)
        for value in DEFAULT_SHELL_COLUMNS:
            if value in shell_fields and value not in self.shell_columns:
                self.shell_columns.append(value)
        for value in DEFAULT_SHELL_PLOTS:
            if value in shell_fields and value not in self.shell_plots:
                self.shell_plots.append(value)
        for value in DEFAULT_RADAR_METRICS:
            if value in radar_fields and value not in self.radar_metrics:
                self.radar_metrics.append(value)
        for value in radar_fields:
            self.radar_directions.setdefault(value, "lower" if value in LOWER_IS_BETTER else "higher")
        return self

    def active_overall(self, available: Sequence[str]) -> List[str]:
        active = [value for value in self.overall_columns if value in available]
        return ["name", *[value for value in active if value != "name"]]

    def active_shell_columns(self, available: Sequence[str]) -> List[str]:
        selected = [value for value in self.shell_columns if value in available]
        prefix = [value for value in ("dmin", "dmax") if value in available]
        return [*prefix, *[value for value in selected if value not in prefix]]

    def active_shell_plots(self, available: Sequence[str]) -> List[str]:
        return [value for value in self.shell_plots if value in available and value not in {"name", "dmin", "dmax", "1/d"}]

    def active_radar(self, available: Sequence[str]) -> List[str]:
        return [value for value in self.radar_metrics if value in available]


def settings_path() -> Path:
    appdata = os.environ.get("APPDATA")
    root = Path(appdata) if appdata else Path.home() / ".cap-tools"
    return root / "cap-tools" / "finalization_viewer.json" if appdata else root / "finalization_viewer.json"


def load_view_settings(path: Optional[Path] = None) -> tuple[ViewSettings, Optional[str]]:
    target = path or settings_path()
    if not target.is_file():
        return ViewSettings(), None
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        return ViewSettings.from_dict(raw), None
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return ViewSettings(), f"Could not load view settings from {target}: {exc}"


def save_view_settings(settings: ViewSettings, path: Optional[Path] = None) -> None:
    target = path or settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(settings.to_dict(), indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def stable_fields(frames: Iterable[pd.DataFrame]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for frame in frames:
        for column in frame.columns:
            if column not in seen:
                seen.add(column)
                result.append(column)
    return result


def view_option_fields(
    overall_fields: Sequence[str],
    shell_fields: Sequence[str],
    radar_fields: Sequence[str],
) -> List[str]:
    """Return the stable row order for the view-options checkbox matrix."""
    seen: set[str] = set()
    result: List[str] = []
    for fields in (overall_fields, shell_fields, radar_fields):
        for field in fields:
            if field not in seen:
                seen.add(field)
                result.append(field)
    return result


def numeric_fields(frame: pd.DataFrame, *, exclude: Iterable[str] = ()) -> List[str]:
    excluded = set(exclude)
    fields: List[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().any():
            fields.append(column)
    return fields


def normalize_radar_frame(
    frame: pd.DataFrame,
    metrics: Sequence[str],
    directions: Mapping[str, str],
) -> pd.DataFrame:
    """Normalize radar metrics within Cluster, or across the whole frame."""
    if frame.empty:
        return pd.DataFrame(columns=["name", *metrics])
    result = frame.copy()
    if "Cluster" not in result or result["Cluster"].isna().all():
        result["__radar_group"] = "all"
    else:
        result["__radar_group"] = result["Cluster"].fillna("unassigned").astype(str)

    for metric in metrics:
        values = pd.to_numeric(result.get(metric), errors="coerce")
        result[metric] = np.nan
        for _, indices in result.groupby("__radar_group", sort=False).groups.items():
            group = values.loc[indices]
            finite = group[np.isfinite(group)]
            if finite.empty:
                continue
            if directions.get(metric, "higher") == "lower":
                positive = finite[finite > 0]
                if positive.empty:
                    continue
                scale = positive.min()
                normalized = scale / group.where(group > 0)
            else:
                scale = finite.max()
                if not math.isfinite(scale) or scale == 0:
                    continue
                normalized = group / scale
            result.loc[indices, metric] = normalized.replace([np.inf, -np.inf], np.nan).clip(0, 1)
    return result[[column for column in ["name", *metrics] if column in result]]


def normalized_radar_selection(
    frame: pd.DataFrame,
    selected_names: Sequence[str],
    metrics: Sequence[str],
    directions: Mapping[str, str],
) -> pd.DataFrame:
    """Normalize against every loaded row, then retain the selected rows."""
    normalized = normalize_radar_frame(frame, metrics, directions)
    if "name" not in normalized:
        return normalized
    return normalized[normalized["name"].isin(selected_names)].copy()
