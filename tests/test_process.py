import pandas as pd


def test_get_diff_info_adapts_cap_auto_results(monkeypatch):
    from cap_tools import process

    def fake_get_diff_info(path, **kwargs):
        assert path == "C:/data/exp1"
        assert kwargs["keep_peak_file"] is True
        assert kwargs["keep_powder_file"] is False
        assert kwargs["redo_peak_hunt"] is False
        assert kwargs["wavelength"] == 0.0251
        assert kwargs["cap"] is None
        return (
            [{"d_max": 10.0, "d_min": 1.0, "I_tot": 100.0, "I_peak": 25.0, "N_peaks": 2, "peak_ratio": 0.25}],
            [{"x": 1.0, "y": 2.0, "R": 0.5, "I": 25.0, "inv_d": 20.0, "d": 0.05}],
            [{"d_value": 2.0, "intensity": 100.0, "inv_d": 0.5}],
            "C:/data/exp1_diff_screen.png",
        )

    monkeypatch.setattr(process, "_cap_auto_get_diff_info", fake_get_diff_info)

    shelldata, peak_table, powder, diff_img = process.get_diff_info(
        "C:/data/exp1",
        keep_peak_file=True,
        keep_powder_file=False,
        redo_peak_hunt=False,
    )

    assert diff_img == "C:/data/exp1_diff_screen.png"
    pd.testing.assert_frame_equal(
        shelldata,
        pd.DataFrame(
            [{"d_max": 10.0, "d_min": 1.0, "I_tot": 100.0, "I_peak": 25.0, "N_peaks": 2, "peak_ratio": 0.25}]
        ),
    )
    assert list(peak_table["1/d"]) == [20.0]
    assert list(powder["d-value"]) == [2.0]
    assert list(powder["intx"]) == [100.0]


def test_get_diff_info_loads_supplied_cap_before_delegating(monkeypatch):
    from cap_tools import process

    class FakeCap:
        def __init__(self):
            self.loaded = []

        def load_experiment(self, filename):
            self.loaded.append(filename)

    fake_cap = FakeCap()

    def fake_get_diff_info(path, **kwargs):
        assert kwargs["cap"] is fake_cap
        return ([], [], [], "C:/data/exp1_diff_screen.png")

    monkeypatch.setattr(process, "_cap_auto_get_diff_info", fake_get_diff_info)

    process.get_diff_info("C:/data/exp1", cap=fake_cap)

    assert fake_cap.loaded == ["C:/data/exp1.par"]
