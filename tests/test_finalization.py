import os
from pathlib import Path

import pandas as pd
import pytest


def _fake_result_parser(self, check_current=False, timeout=0):
    if "bad" in self.name:
        raise RuntimeError("malformed test result")
    self.shells = pd.DataFrame(
        [{"dmax": 2.0, "dmin": 1.0, "complete": 90.0, "CC1/2": 0.9, "1/d": 0.75}]
    )
    self.overall = pd.DataFrame(
        [{"dmax": 20.0, "dmin": 1.0, "complete": 95.0, "CC1/2": 0.95, "1/d": 0.525}]
    )


def _summary(folder: Path, name: str, timestamp: float) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    result = folder / f"{name}_red.sum"
    result.write_text("test\n", encoding="utf-8")
    os.utime(result, (timestamp, timestamp))
    return result


def test_olex2_parser_uses_exact_root_file(tmp_path):
    from cap_tools.finalization import parse_olex2_refinement

    base = tmp_path / "exp_1" / "result"
    olex = base.parent / "struct" / "olex2_result"
    olex.mkdir(parents=True)
    (olex / "ov_result.res").write_text(
        "REM R1_gt = 0.999\nREM wR_ref = 0.999\nREM GOOF = 9.99\n", encoding="utf-8"
    )
    (olex / "result.res").write_text(
        "REM R1_gt = 0.1234\nREM wR_ref = 0.2345\nREM GOOF = 1.111\n", encoding="utf-8"
    )
    temp = olex / "olex2" / "temp"
    temp.mkdir(parents=True)
    (temp / "result.res").write_text("REM R1_gt = 0.888\n", encoding="utf-8")

    parsed = parse_olex2_refinement(str(base))

    assert parsed.status == "complete"
    assert parsed.r1_gt == pytest.approx(0.1234)
    assert parsed.wr_ref == pytest.approx(0.2345)
    assert parsed.goof == pytest.approx(1.111)
    assert parsed.source == str(olex / "result.res")


def test_olex2_parser_allows_partial_and_missing_results(tmp_path):
    from cap_tools.finalization import parse_olex2_refinement

    base = tmp_path / "exp_1" / "result"
    assert parse_olex2_refinement(str(base)).status == "not found"

    olex = base.parent / "struct" / "olex2_result"
    olex.mkdir(parents=True)
    (olex / "result.res").write_text("REM R1_gt = 0.1\nREM wR_ref = n/a\n", encoding="utf-8")
    parsed = parse_olex2_refinement(str(base))
    assert parsed.status == "partial"
    assert parsed.r1_gt == pytest.approx(0.1)
    assert parsed.wr_ref is None
    assert parsed.goof is None


def test_merge_membership_maps_clustering_indices(tmp_path):
    from cap_tools.finalization import parse_merge_membership

    base = tmp_path / "merged" / "result"
    expinfo = base.parent / "expinfo"
    expinfo.mkdir(parents=True)
    (expinfo / "merged.ini").write_text(
        """[Number of merged experiments]
number of merged experiments=3
[Merged experiment 1]
experiment folder path=C:\\data\\exp_1
experiment name=exp_1
rrpprof name=reproc_exp_1
[Merged experiment 2]
experiment folder path=C:\\data\\exp_2
experiment name=exp_2
rrpprof name=reproc_exp_2
[Merged experiment 3]
experiment folder path=C:\\data\\exp_3
experiment name=exp_3
rrpprof name=reproc_exp_3
""",
        encoding="utf-8",
    )
    (expinfo / "clustering.ini").write_text(
        """[Selected batches]
is 1 batch selected=1
is 2 batch selected=0
is 3 batch selected=1
""",
        encoding="utf-8",
    )

    membership = parse_merge_membership(str(base))

    assert membership.total_count == 3
    assert membership.used_count == 2
    assert membership.used_names == ("exp_1", "exp_3")
    assert membership.excluded_names == ("exp_2",)
    assert not membership.warnings

    (expinfo / "clustering.ini").write_text(
        "[Selected batches]\nis 1 batch selected=1\nis 2 batch selected=1\n",
        encoding="utf-8",
    )
    inconsistent = parse_merge_membership(str(base))
    assert inconsistent.used_count is None
    assert any("do not match" in message for message in inconsistent.warnings)

    (expinfo / "clustering.ini").unlink()
    without_selection = parse_merge_membership(str(base))
    assert without_selection.total_count == 3
    assert without_selection.used_count is None


def test_folder_load_modes_and_newest_valid_fallback(tmp_path, monkeypatch):
    from cap_tools import finalization

    monkeypatch.setattr(finalization.Finalization, "parse_finalization_results", _fake_result_parser)
    exp1 = tmp_path / "exp_1"
    _summary(exp1, "old", 100)
    _summary(exp1, "bad_new", 300)
    _summary(exp1, "C2_auto", 200)
    exp2 = tmp_path / "exp_2"
    _summary(exp2, "other", 100)

    with pytest.warns(RuntimeWarning, match="newer malformed"):
        current = finalization.FinalizationCollection.from_folder(
            str(tmp_path), include_subfolders=True
        )
    assert set(current) == {"C2_auto", "other"}
    assert current.skipped_count == 2
    assert current.malformed_count == 1

    with pytest.warns(RuntimeWarning, match="bad_new"):
        all_results = finalization.FinalizationCollection.from_folder(
            str(tmp_path),
            include_subfolders=True,
            ignore_parse_errors=True,
            load_options=finalization.FinalizationLoadOptions(finalization.FinalizationLoadMode.ALL),
        )
    assert set(all_results) == {"C2_auto", "old", "other"}
    assert all_results.skipped_count == 1
    assert all_results.malformed_count == 1

    patterns = finalization.FinalizationLoadOptions(
        finalization.FinalizationLoadMode.PATTERNS,
        include_patterns=("*C2*",),
        exclude_patterns=("*auto*",),
    )
    filtered = finalization.FinalizationCollection.from_folder(
        str(tmp_path), include_subfolders=True, ignore_parse_errors=True, load_options=patterns
    )
    assert set(filtered) == {"C2_auto"}
    assert filtered.skipped_count == 3
    assert filtered.malformed_count == 0


def test_duplicate_basenames_get_parent_qualifiers(tmp_path, monkeypatch):
    from cap_tools import finalization

    monkeypatch.setattr(finalization.Finalization, "parse_finalization_results", _fake_result_parser)
    _summary(tmp_path / "exp_1", "result", 100)
    _summary(tmp_path / "exp_2", "result", 100)
    collection = finalization.FinalizationCollection.from_folder(
        str(tmp_path),
        include_subfolders=True,
        load_options=finalization.FinalizationLoadOptions(finalization.FinalizationLoadMode.ALL),
    )
    assert len(collection) == 2
    assert "result" in collection
    assert any(key.startswith("result [exp_") for key in collection if key != "result")


def test_csv_current_and_pattern_modes(tmp_path, monkeypatch):
    from cap_tools import finalization

    monkeypatch.setattr(finalization.Finalization, "parse_finalization_results", _fake_result_parser)
    experiment = tmp_path / "exp_1"
    _summary(experiment, "selected", 100)
    _summary(experiment, "C2_extra", 200)
    csv_path = tmp_path / "all_sets.csv"
    preamble = "\n".join(
        ["VERSION 1", "HEADER INFO:", "Number of experiments: 1", "Number of columns 3", "", "", "", ""]
    )
    csv_path.write_text(
        preamble
        + "Experiment_name,Experiment_path,Finalization_output_file\n"
        + f'exp_1,"{experiment}",selected\n',
        encoding="utf-8",
    )

    current = finalization.FinalizationCollection.from_csv(str(csv_path))
    assert set(current) == {"selected"}
    assert current["selected"].meta["Experiment"] == "exp_1"

    options = finalization.FinalizationLoadOptions(
        finalization.FinalizationLoadMode.PATTERNS, include_patterns=("*c2*",)
    )
    filtered = finalization.FinalizationCollection.from_csv(str(csv_path), load_options=options)
    assert set(filtered) == {"C2_extra"}
