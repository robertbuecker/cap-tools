# cap-tools

Auxiliary electron diffraction tools for CrysAlisPro workflows.

This major branch is being refactored toward four maintained applications:

- finalization viewer
- detector distance calibration
- screening viewer
- learning set generator

CAP automation is delegated to `cap-auto`; native `.rodhypix` file access is delegated to `rodhypix`.

## Finalization viewer

The finalization viewer compares CrysAlisPro `*_red.sum` results and can load
the current, all, or glob-filtered finalizations from folders or DataExplorer
CSV exports. It also reads:

- Olex2 `R1_gt`, `wR_ref`, and `GOOF` values from
  `struct/olex2_<finalization>/<finalization>.res`;
- merged experiment membership from `expinfo/merged.ini`; and
- the experiments actually used from `expinfo/clustering.ini`.

Use **View Options** to choose overview columns, per-shell table columns, and
spider-plot axes and inversion. The two per-shell line plots have direct
upper/lower metric selectors. These choices are stored in
`%APPDATA%\cap-tools\finalization_viewer.json`.

Folder and CSV loading defaults to the current finalization. The CLI exposes
the same selection behavior:

```powershell
powershell -ExecutionPolicy Bypass -File .\with-cap-tools.ps1 python -m cap_tools.finalization_viewer --selection all <folder>
powershell -ExecutionPolicy Bypass -File .\with-cap-tools.ps1 python -m cap_tools.finalization_viewer --selection patterns --include "*C2*" --exclude "*auto*" <folder>
```
