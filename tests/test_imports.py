import importlib
import tomllib
from pathlib import Path


def test_public_modules_import():
    for module_name in [
        "cap_tools.calibrate_dd",
        "cap_tools.finalization",
        "cap_tools.finalization_viewer",
        "cap_tools.learning_set",
        "cap_tools.screening",
        "cap_tools.screening_viewer",
    ]:
        importlib.import_module(module_name)


def test_cap_control_removed_from_active_code():
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in root.rglob("*.py"):
        if any(part in {"build", "dist", "dist-cx", "__pycache__", "tests"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        if "cap_tools.cap_control" in text:
            offenders.append(path.relative_to(root).as_posix())
    assert offenders == []


def test_compatibility_launchers_are_thin():
    root = Path(__file__).resolve().parents[1]
    for script in [
        "calibrate_dd.py",
        "finalization_viewer.py",
        "generate_learning_set.py",
        "screening_viewer.py",
    ]:
        text = (root / script).read_text(encoding="utf-8")
        assert "main_cli" in text
        assert len(text.splitlines()) <= 6


def test_console_script_targets_import():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = project["project"]["scripts"]
    assert set(scripts) == {
        "cap-tools-calibrate-dd",
        "cap-tools-finalization-viewer",
        "cap-tools-screening-viewer",
        "cap-tools-learning-set",
    }

    for target in scripts.values():
        module_name, attr_name = target.split(":", 1)
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attr_name))


def test_legacy_cell_clustering_surface_removed():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "cap_tools" / "cell_list.py").exists()
    assert not (root / "cap_tools" / "screening_report.py").exists()

    for module_name in [
        "cap_tools.widgets",
        "cap_tools.finalization_viewer",
        "cap_tools.screening_viewer",
    ]:
        module = importlib.import_module(module_name)
        assert not hasattr(module, "CellList")
        assert not hasattr(module, "ClusterWidget")
