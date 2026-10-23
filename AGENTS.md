# AGENTS.md

## Project Context

`cap-tools` contains auxiliary CrysAlisPro workflow applications. The active maintained apps are:

- finalization viewer
- detector distance calibration
- screening viewer
- learning set generator

CAP automation lives in the sibling `cap-auto` repository. Native `.rodhypix` file access lives in the sibling `rodhypix` repository. Both are consumed as dependencies.

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

- Build releases with `powershell -ExecutionPolicy Bypass -File .\with-cap-tools.ps1 python setup_cx.py build_exe` (with elevated execution when the Tcl sandbox restriction applies). Every successful build must create `dist-cx/cap-tools-<VERSION>.zip`, using the metadata version without the leading `v`. The archive contains a versioned folder with three portable `.cmd` launchers and an `app` runtime subfolder. Ship finalization viewer, detector distance calibration, and learning set generator; screening viewer remains source-only. Keep Windows DLLs beside their runtime executables. Verify ZIP contents and launch the viewer from a relocated extraction before publishing.

- In restricted agent sessions, Tcl can report that it cannot find a usable `init.tcl` even when the file exists. First compare `python -c "import tkinter; print(tkinter.Tcl().eval('info patchlevel'))"` through the wrapper with elevated execution. On this machine Tcl 8.6.13 works outside the filesystem sandbox. Run Tk-dependent tests and `python setup_cx.py build` with that access when needed; do not reinstall Tcl/Tk based on the sandbox error alone.

- Keep changes scoped to the active maintained apps unless the task explicitly asks for broader cleanup.
- Prefer adding or updating focused tests in `tests/` for behavioral changes.
- Do not reintroduce direct `cap_tools.cap_control` usage; CAP control belongs in `cap-auto`.
- Do not reintroduce the removed cell clustering package surface; CAP now owns that workflow.
- The workspace may have a locked `.pytest_cache`; warnings about failing to write pytest cache are not usually test failures.
