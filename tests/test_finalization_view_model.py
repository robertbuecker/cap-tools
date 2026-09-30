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
    assert "R1_gt" in loaded.overall_columns
    assert "wR_ref" in loaded.radar_metrics
    assert loaded.radar_directions["R1_gt"] == "lower"


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


def test_view_option_matrix_rows_use_stable_union_order():
    from cap_tools.finalization_view_model import view_option_fields

    assert view_option_fields(
        ["name", "complete", "R1_gt"],
        ["dmin", "dmax", "complete", "CC1/2"],
        ["complete", "CC1/2", "R1_gt", "wR_ref"],
    ) == ["name", "complete", "R1_gt", "dmin", "dmax", "CC1/2", "wR_ref"]
