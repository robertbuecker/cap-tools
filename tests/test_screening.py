import numpy as np
import pandas as pd


def test_load_experiment_metadata_uses_cap_auto_parser(monkeypatch):
    from cap_tools import screening

    calls = []

    def fake_parse_cap_meta(experiments, include=None, log_fun=None):
        calls.append((experiments, include, log_fun))
        return [
            {"name": "exp1", "path": "C:/data/exp1", "r_int": 0.1},
            {"name": "exp2", "path": "C:/data/exp2", "r_int": 0.2},
        ]

    monkeypatch.setattr(screening, "parse_cap_meta", fake_parse_cap_meta)
    frame = screening.load_experiment_metadata(["C:/data"], pre_only=True, log=lambda msg: None)

    assert calls[0][0] == ["C:/data"]
    assert calls[0][1] == ["pre_"]
    assert list(frame.index) == ["exp1", "exp2"]
    assert list(frame["path"]) == ["C:/data/exp1", "C:/data/exp2"]


def test_find_peaks_raw_frame_normalizes_peakfinder_output(monkeypatch):
    from cap_tools import screening

    def fake_peakfinder8(*args):
        return [
            [10.0, 20.0],
            [11.0, 21.0],
            [100.0, 200.0],
            [0, 1],
            [3, 4],
            [60.0, 70.0],
            [5.0, 6.0],
            [12.0, 13.0],
        ]

    monkeypatch.setattr(screening, "peakfinder8", fake_peakfinder8)
    peaks = screening.find_peaks_raw_frame(np.ones((32, 32), dtype=np.float32))

    expected = pd.DataFrame(
        {
            "x": [10.0, 20.0],
            "y": [11.0, 21.0],
            "I": [100.0, 200.0],
            "peak_index": [0, 1],
            "npix": [3, 4],
            "max_intensity": [60.0, 70.0],
            "sigma": [5.0, 6.0],
            "snr": [12.0, 13.0],
        }
    )
    pd.testing.assert_frame_equal(peaks, expected)
