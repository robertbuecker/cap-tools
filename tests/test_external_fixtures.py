import hashlib
import os
import shutil
import uuid
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest


DD_LAUNCH_DATASET = Path(r"C:\XcaliburData\DD\RESE\Tue-Jul-02-13-19-14-2024")


def _test_output_dir(name: str) -> Path:
    root = Path(__file__).resolve().parents[1] / ".tmp" / "test-output"
    path = root / f"{name}-{os.getpid()}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


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


def test_dd_fixture_smoke():
    from cap_tools.calibrate_dd import run_cli

    root = _fixture_root("CAP_TOOLS_DD_FIXTURES")
    out_dir = _test_output_dir("dd-fixture")
    try:
        output_csv = out_dir / "detector_distance.csv"
        report_dir = out_dir / "reports"

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
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_dd_launch_config_dataset():
    from cap_tools.calibrate_dd import run_cli

    if not DD_LAUNCH_DATASET.exists():
        pytest.skip(f"DD launch dataset does not exist: {DD_LAUNCH_DATASET}")

    out_dir = _test_output_dir("dd-launch")
    try:
        output_csv = out_dir / "detector_distance.csv"
        report_dir = out_dir / "reports"
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
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_learning_set_fixture_manifest():
    from cap_tools.learning_set import main

    root = _fixture_root("CAP_TOOLS_LEARNING_FIXTURES")
    expected_csv = root / "expected" / "info.csv"
    expected_hashes = root / "expected" / "sha256.txt"
    input_root = root / "input"
    out_dir = _test_output_dir("learning")
    try:
        main([str(input_root)], str(out_dir), cmdline=True, zip_result=True)

        info_csv = out_dir / "info.csv"
        zip_file = out_dir / "learning_set.zip"
        assert info_csv.exists()
        assert zip_file.exists()

        actual = pd.read_csv(info_csv)
        image_columns = [column for column in ("diff_img", "grain_img") if column in actual]
        for column in image_columns:
            for filename in actual[column].dropna():
                assert (out_dir / filename).exists()

        with ZipFile(zip_file) as archive:
            archived = set(archive.namelist())
        assert "info.csv" in archived
        for column in image_columns:
            for filename in actual[column].dropna():
                if Path(filename).suffix.lower() == ".tiff":
                    assert filename in archived

        if expected_csv.exists():
            expected = pd.read_csv(expected_csv)
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
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_finalization_fixture_parses():
    from cap_tools.finalization import FinalizationCollection

    root = _fixture_root("CAP_TOOLS_FINALIZATION_FIXTURES")
    collection = FinalizationCollection.from_folder(str(root), include_subfolders=True, ignore_parse_errors=True)

    assert len(collection) > 0
    assert not collection.overall.empty
    assert not collection.highest_shell.empty
    assert not collection.overall_highest.empty
