from __future__ import annotations

from pathlib import Path

from cap_tools import VERSION_TAG, __version__, get_version


ROOT = Path(__file__).resolve().parents[1]


def test_authoritative_version_matches_release_metadata() -> None:
    assert __version__ == "45.1"
    assert VERSION_TAG == "v45.1"
    assert get_version() == VERSION_TAG


def test_package_metadata_uses_the_authoritative_version_module() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'dynamic = ["version"]' in project
    assert 'version = {attr = "cap_tools._version.__version__"}' in project
