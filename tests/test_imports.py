import importlib
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
