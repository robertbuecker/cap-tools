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

## INI Wrangler foundation

`cap_tools.ini_wrangler` reads CAP MicroED configuration files into typed nested
objects and lists, writes editable JSON and normalized INI, and compares values.
It has no GUI, calibration judgments, automatic repairs, or instrument access.
All five supplied examples in `ini_files/` are regression fixtures; their readable
exports are in `ini_files/json/`. The template is an example, not a validation baseline.

```powershell
powershell -ExecutionPolicy Bypass -File .\with-cap-tools.ps1 python -m cap_tools.ini_wrangler to-json .\ini_files\CrysAlisMicroEDConfig_ver3-template.ini -o .\template.json
powershell -ExecutionPolicy Bypass -File .\with-cap-tools.ps1 python -m cap_tools.ini_wrangler to-ini .\template.json -o .\restored.ini
powershell -ExecutionPolicy Bypass -File .\with-cap-tools.ps1 python -m cap_tools.ini_wrangler diff .\ini_files\CrysAlisMicroEDConfig_ver3-template.ini .\restored.ini -o .\comparison.json
```

Output paths are required. Existing output files require `--overwrite`; CLI output
must differ from its inputs. Conversion and diff commands return 0 on success
(including a nonempty diff), or 2 on errors. Diagnostics are written to stderr;
diffs are written as JSON with `added_sections`, `removed_sections`, and `changes`.
Each change includes its kind, section, member path, and old/new values. A missing
side is `null`; an empty string and zero are distinct values. Arrays match by index,
with no preset matching or floating-point tolerance.

The Python API supports the same operations:

```python
from cap_tools.ini_wrangler import read_ini, read_json, write_ini, write_json, diff_documents

document = read_ini("configuration.ini")
preset = document.sections["Jeol Optics Presets"]["smicroedspec"]["saopticspresetplus_dcmode_multi_predef"][0]
preset["ddetectordistance"] = 650.0
write_json(document, "configuration.json")
restored = read_json("configuration.json")
write_ini(restored, "restored.ini")
changes = diff_documents(document, restored)
```

`loads_ini`/`dumps_ini` and `loads_json`/`dumps_json` work with strings. Writers
refuse existing paths by default; pass `overwrite=True` explicitly when needed.
`IniDocument.sections` is editable and `entries()` exposes `(section, path)` keys.
Diagnostics retain their code, path, source, and INI line number; JSON diagnostics
have no line number. They describe the last read and are regenerated on import,
rather than tracking subsequent edits.

JSON uses `{"format_version": 1, "sections": {...}}`. Dotted paths become
objects; bracket indices become lists. CAP names and section names remain literal,
including keys containing `->`. Numeric conventions (`i`, `l`, `w`, `e`, `dw`, `d`)
and observed geometry exceptions are decoded centrally; `c` fields are strings.
Enums and integer flags remain integers. Unknown fields and malformed numeric
values remain strings with diagnostics. JSON edits must retain those types:
use a decimal such as `650.0` for a double, and strings for unknown fields.

Round-trips preserve section/key sets, indices, empty strings, and typed values.
Ordering, whitespace, comments, and numeric spellings normalize. Values are literal:
there is no interpolation, inline comment stripping, special `[DEFAULT]` section,
or multiline continuation. Files are read as UTF-8 with optional BOM and written
as UTF-8. Floating-point fields use Python's finite double precision.
Sparse lists use `null` holes, omitted from INI output; indices above 100000 are
rejected to bound allocation. Duplicate sections/keys, conflicting tree shapes,
and malformed syntax are errors. Writers also reject structures INI cannot encode:
booleans, scalar nulls, empty nested containers, trailing list holes, dotted JSON
member names, and strings with surrounding whitespace or line breaks. Empty
top-level sections are supported.

Run the focused checks through the repository wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File .\with-cap-tools.ps1 python -m pytest tests\test_ini_wrangler.py tests\test_imports.py -q
```
