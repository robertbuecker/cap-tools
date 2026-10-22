import json

import numpy as np
import pandas as pd


def test_view_settings_round_trip_and_reconcile(tmp_path):
    from cap_tools.finalization_view_model import ViewSettings, load_view_settings, save_view_settings

    path = tmp_path / "settings.json"
    settings = ViewSettings(overall_columns=["name", "CC1/2"], radar_metrics=["R1_gt"])
    save_view_settings(settings, path)
    loaded, warning = load_view_settings(path)
    assert warning is None
    assert loaded.overall_columns == ["name", "CC1/2"]
    assert loaded.radar_metrics == ["R1_gt"]

    loaded.reconcile(
        overall_fields=["name", "CC1/2", "R1_gt", "wR_ref", "GOOF"],
        shell_fields=["dmin", "dmax", "CC1/2", "complete"],
        radar_fields=["CC1/2", "R1_gt", "wR_ref"],
    )
    assert "R1_gt" not in loaded.overall_columns
    assert "wR_ref" not in loaded.radar_metrics
    assert loaded.radar_directions["R1_gt"] == "lower"


def test_reconcile_adds_new_defaults_once_but_preserves_deselection():
    from cap_tools.finalization_view_model import ViewSettings

    settings = ViewSettings(
        overall_columns=["name"],
        radar_metrics=["CC1/2"],
        known_overall_fields=["name"],
        known_radar_fields=["CC1/2"],
    )
    available = ["name", "CC1/2", "R1_gt", "wR_ref", "GOOF"]
    settings.reconcile(overall_fields=available, shell_fields=[], radar_fields=available[1:])
    assert {"R1_gt", "wR_ref", "GOOF"} <= set(settings.overall_columns)
    assert {"R1_gt", "wR_ref"} <= set(settings.radar_metrics)

    settings.overall_columns.remove("R1_gt")
    settings.radar_metrics.remove("R1_gt")
    settings.reconcile(overall_fields=available, shell_fields=[], radar_fields=available[1:])
    assert "R1_gt" not in settings.overall_columns
    assert "R1_gt" not in settings.radar_metrics


def test_corrupt_view_settings_fall_back_to_defaults(tmp_path):
    from cap_tools.finalization_view_model import load_view_settings

    path = tmp_path / "settings.json"
    path.write_text("{bad json", encoding="utf-8")
    settings, warning = load_view_settings(path)
    assert settings.overall_columns[0] == "name"
    assert warning and "Could not load" in warning


def test_radar_normalization_respects_direction_and_clusters():
    from cap_tools.finalization_view_model import normalize_radar_frame

    data = pd.DataFrame(
        {
            "name": ["a", "b", "c"],
            "Cluster": [1, 1, 2],
            "complete": [50.0, 100.0, 25.0],
            "R1_gt": [0.1, 0.2, 0.0],
        }
    )
    normalized = normalize_radar_frame(
        data, ["complete", "R1_gt"], {"complete": "higher", "R1_gt": "lower"}
    ).set_index("name")

    assert normalized.loc["a", "complete"] == 0.5
    assert normalized.loc["b", "complete"] == 1.0
    assert normalized.loc["a", "R1_gt"] == 1.0
    assert normalized.loc["b", "R1_gt"] == 0.5
    assert normalized.loc["c", "complete"] == 1.0
    assert np.isnan(normalized.loc["c", "R1_gt"])


def test_radar_selection_is_normalized_against_all_loaded_rows():
    from cap_tools.finalization_view_model import normalized_radar_selection

    data = pd.DataFrame(
        {
            "name": ["selected_a", "selected_b", "not_selected"],
            "complete": [25.0, 50.0, 100.0],
            "R1_gt": [0.2, 0.4, 0.1],
        }
    )
    normalized = normalized_radar_selection(
        data,
        ["selected_a", "selected_b"],
        ["complete", "R1_gt"],
        {"complete": "higher", "R1_gt": "lower"},
    ).set_index("name")

    assert list(normalized.index) == ["selected_a", "selected_b"]
    assert normalized.loc["selected_a", "complete"] == 0.25
    assert normalized.loc["selected_b", "complete"] == 0.5
    assert normalized.loc["selected_a", "R1_gt"] == 0.5
    assert normalized.loc["selected_b", "R1_gt"] == 0.25


def test_radar_selection_order_controls_legend_and_series_colors():
    from matplotlib.figure import Figure

    from cap_tools.finalization_view_model import normalized_radar_selection
    from cap_tools.interact_figures import fom_radar_plot

    frame = pd.DataFrame(
        {
            "name": ["a", "b"],
            "complete": [50.0, 100.0],
            "CC1/2": [0.8, 0.9],
            "R1_gt": [0.2, 0.1],
        }
    )
    selected = ["b", "a"]
    metrics = ["complete", "CC1/2", "R1_gt"]
    overall = normalized_radar_selection(frame, selected, metrics, {"R1_gt": "lower"})
    axes = fom_radar_plot(
        overall,
        overall.copy(),
        fig_handle=Figure(),
        foms=metrics,
        colors=["C0", "C1"],
    )

    assert list(overall["name"]) == selected
    assert [text.get_text() for text in axes[0].get_legend().get_texts()] == selected
    series = [line for line in axes[0].lines if line.get_color() in {"C0", "C1"}]
    assert [line.get_color() for line in series] == ["C0", "C1"]


def test_radar_axis_labels_mark_only_inverted_metrics():
    from cap_tools.finalization_view_model import radar_axis_labels

    assert radar_axis_labels(
        ["complete", "R1_gt", "wR_ref", "custom"],
        {"complete": "higher", "R1_gt": "lower", "wR_ref": "lower"},
    ) == ["Comp", "1/R1", "1/wR2", "custom"]


def test_view_option_matrix_rows_use_stable_union_order():
    from cap_tools.finalization_view_model import view_option_fields

    assert view_option_fields(
        ["name", "complete", "R1_gt"],
        ["dmin", "dmax", "complete", "CC1/2"],
        ["complete", "CC1/2", "R1_gt", "wR_ref"],
    ) == ["name", "complete", "R1_gt", "dmin", "dmax", "CC1/2", "wR_ref"]


def test_highest_shell_radar_omits_overall_only_axes_and_breaks_fill():
    from matplotlib.figure import Figure

    from cap_tools.interact_figures import fom_radar_plot

    metrics = ["complete", "CC1/2", "redundancy", "R1_gt", "wR_ref"]
    overall = pd.DataFrame(
        [{"name": "a", "complete": 0.9, "CC1/2": 0.8, "redundancy": 0.7,
          "R1_gt": 0.6, "wR_ref": 0.5}]
    )
    highest = pd.DataFrame(
        [{"name": "a", "complete": 0.8, "CC1/2": 0.7, "redundancy": 0.6,
          "R1_gt": np.nan, "wR_ref": np.nan}]
    )

    axes = fom_radar_plot(
        overall, highest, fig_handle=Figure(), foms=metrics,
        fom_lbl=["Comp", "CC1/2", "Red", "1/R1", "1/wR2"], colors=["C0"],
    )

    assert [label.get_text() for label in axes[1].get_xticklabels()] == [
        "Comp", "CC1/2", "Red", "", ""
    ]
    data_lines = [line for line in axes[1].lines if line.get_color() == "C0"]
    assert len(data_lines) == 1
    assert len(data_lines[0].get_xdata()) == 3
    assert data_lines[0].get_xdata()[0] != data_lines[0].get_xdata()[-1]
    assert len(axes[1].patches) == 1
    fill_vertices = axes[1].patches[0].get_xy()
    assert fill_vertices[0, 1] == 0
    assert fill_vertices[-2, 1] == 0
