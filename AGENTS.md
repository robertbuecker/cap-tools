# AGENTS.md

## Project Context

`cap-tools` contains auxiliary CrysAlisPro workflow applications. The active maintained apps are:

- finalization viewer
- detector distance calibration
- screening viewer
- learning set generator

CAP automation and native `.rodhypix` file access live in the sibling `cap-auto` repository and are consumed as dependencies.

## Running Commands

Use the repository wrapper for Python commands instead of calling `python`, `pytest`, or installed console scripts directly. On this machine, invoke it through PowerShell with a process-local execution-policy bypass:

```powershell
powershell -ExecutionPolicy Bypass -File .\with-cap-tools.ps1 python -m pytest
powershell -ExecutionPolicy Bypass -File .\with-cap-tools.ps1 python -m pytest tests\test_imports.py -q
powershell -ExecutionPolicy Bypass -File .\with-cap-tools.ps1 python finalization_viewer.py
powershell -ExecutionPolicy Bypass -File .\with-cap-tools.ps1 python -m cap_tools.finalization_viewer
```

The canonical wrapper delegates to `with-cap-tools-pip.ps1`, activates the `cap-tools-pip` conda environment, and sets local temp/cache paths used by the test and runtime environment.

## Development Notes

- Keep changes scoped to the active maintained apps unless the task explicitly asks for broader cleanup.
- Prefer adding or updating focused tests in `tests/` for behavioral changes.
- Do not reintroduce direct `cap_tools.cap_control` usage; CAP control belongs in `cap-auto`.
- Do not reintroduce the removed cell clustering package surface; CAP now owns that workflow.
- The workspace may have a locked `.pytest_cache`; warnings about failing to write pytest cache are not usually test failures.
