import hashlib
import os
from pathlib import Path

import pandas as pd
import pytest


DD_LAUNCH_DATASET = Path(r"C:\XcaliburData\DD\RESE\Tue-Jul-02-13-19-14-2024")


def _fixture_root(env_var: str) -> Path:
    value = os.environ.get(env_var)
    if not value:
        pytest.skip(f"{env_var} is not set")
    root = Path(value)
    if not root.exists():
        pytest.skip(f"{env_var} does not exist: {root}")
    return root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dd_fixture_smoke(tmp_path):
    from cap_tools.calibrate_dd import run_cli

    root = _fixture_root("CAP_TOOLS_DD_FIXTURES")
    output_csv = tmp_path / "detector_distance.csv"
    report_dir = tmp_path / "reports"

    report = run_cli(str(root), output_csv=str(output_csv), report_dir=str(report_dir), quiet=True)

    assert not report.empty
    assert output_csv.exists()
    assert set([
        "Label",
        "Old DD (mm)",
        "DD segmented fit (mm)",
        "DD change (%)",
        "Report file",
    ]).issubset(report.columns)
    assert list(report_dir.glob("*.pdf"))


def test_dd_launch_config_dataset(tmp_path):
    from cap_tools.calibrate_dd import run_cli

    if not DD_LAUNCH_DATASET.exists():
        pytest.skip(f"DD launch dataset does not exist: {DD_LAUNCH_DATASET}")

    output_csv = tmp_path / "detector_distance.csv"
    report_dir = tmp_path / "reports"
    report = run_cli(
        str(DD_LAUNCH_DATASET),
        output_csv=str(output_csv),
        report_dir=str(report_dir),
        quiet=True,
    )

    assert len(report) > 0
    assert output_csv.exists()
    assert list(report_dir.glob("*.pdf"))
    assert report["DD segmented fit (mm)"].notna().all()
    assert report["Report file"].map(lambda value: str(report_dir) in value).all()


def test_learning_set_fixture_manifest(tmp_path):
    from cap_tools.learning_set import main

    root = _fixture_root("CAP_TOOLS_LEARNING_FIXTURES")
    expected_csv = root / "expected" / "info.csv"
    expected_hashes = root / "expected" / "sha256.txt"
    input_root = root / "input"
    out_dir = tmp_path / "learning"

    main([str(input_root)], str(out_dir), cmdline=True, zip_result=True)

    if expected_csv.exists():
        expected = pd.read_csv(expected_csv)
        actual = pd.read_csv(out_dir / "info.csv")
        pd.testing.assert_frame_equal(actual, expected)

    if expected_hashes.exists():
        expected = dict(
            line.strip().split(maxsplit=1)
            for line in expected_hashes.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        actual = {
            path.name: _sha256(path)
            for path in out_dir.iterdir()
            if path.is_file() and path.name in expected
        }
        assert actual == expected


def test_finalization_fixture_parses():
    from cap_tools.finalization import FinalizationCollection

    root = _fixture_root("CAP_TOOLS_FINALIZATION_FIXTURES")
    collection = FinalizationCollection.from_folder(str(root), include_subfolders=True, ignore_parse_errors=True)

    assert len(collection) > 0
    assert not collection.overall.empty
    assert not collection.highest_shell.empty
    assert not collection.overall_highest.empty
