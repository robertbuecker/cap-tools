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
python -m cap_tools.finalization_viewer --selection all <folder>
python -m cap_tools.finalization_viewer --selection patterns --include "*C2*" --exclude "*auto*" <folder>
```

## INI Wrangler foundation

`cap_tools.ini_wrangler` reads CAP MicroED configuration files into typed nested
objects and lists, writes editable JSON and normalized INI, compares values,
and applies selected diff operations to other configurations.
It has no GUI, calibration judgments, automatic repairs, or instrument access.
All five supplied examples in `ini_files/` are regression fixtures; their readable
exports are in `ini_files/json/`. The template is an example, not a validation baseline.

```powershell
python -m cap_tools.ini_wrangler to-json .\ini_files\CrysAlisMicroEDConfig_ver3-template.ini -o .\template.json
python -m cap_tools.ini_wrangler to-ini .\template.json -o .\restored.ini
python -m cap_tools.ini_wrangler diff .\ini_files\CrysAlisMicroEDConfig_ver3-template.ini .\restored.ini -o .\comparison.json
```

Conversion and apply output paths are required. For `diff`, omitting `-o` uses
`default_diff_name(first, second)` in the current working directory. It discards
folders and the final extension, then removes the common filename prefix while
keeping any shared suffix. For example, `CrysAlisMicroEDConfig_ver3_120kV-RC_261023.ini`
versus `CrysAlisMicroEDConfig_ver3_200kV-RC_261023.ini` becomes
`DIFF-120kV-RC_261023-VS-200kV-RC_261023.json`. If prefix removal would empty either
name, both full stems are retained.

Existing output files require `--overwrite`; CLI output must differ from all its
inputs. Conversion, diff, and apply commands return 0 on success
(including a nonempty diff), or 2 on errors. Diagnostics are written to stderr;
diffs are written as JSON with `added_sections`, `removed_sections`, and `changes`.
Each change includes its kind, section, member path, and old/new values. A missing
side is `null`; an empty string and zero are distinct values. Arrays match by index,
with no preset matching or floating-point tolerance.

Apply any combination of `--additions`, `--deletions`, and `--changes`, or use
`--all`. At least one selector is required:

```powershell
python -m cap_tools.ini_wrangler diff old.ini new.ini
python -m cap_tools.ini_wrangler apply another.ini DIFF-old-VS-new.json --changes --additions -o updated.ini
python -m cap_tools.ini_wrangler apply another.ini DIFF-old-VS-new.json --all --check-old -o updated-strict.ini
```

The target may differ from both files that generated the diff. Changes replace
existing target values, even when they differ from the diff's old value. Deletions
remove only the named entries. Add `--check-old` to require matching old values
for changes and deletions. Missing change targets and additions conflicting with
existing values are errors; identical additions and missing deletions are no-ops
(missing deletions are errors with `--check-old`).

Array indices never shift after deletion. Removed sections disappear only when
empty; unrelated target entries survive. Selecting additions also creates declared
added sections, including empty ones. The complete diff is validated even when
only a subset is selected. Malformed diffs, value conflicts, and incompatible
tree shapes abort without writing output; input documents are never mutated.

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

`loads_diff`/`read_diff` and `dumps_diff`/`write_diff` validate and exchange the
same JSON shape emitted by `diff`. They reject duplicate operations, contradictory
section declarations, invalid paths/types, and identical old/new changes.

```python
from cap_tools.ini_wrangler import apply_diff, read_diff, read_ini, write_ini

target = read_ini("another.ini")
updated = apply_diff(target, read_diff("comparison.json"),
                     include_additions=True, include_changes=True)
write_ini(updated, "updated.ini")
# Or: apply_diff(target, patch, include_all=True, check_old=True)
```

`apply_diff` returns a new `IniDocument` with regenerated diagnostics. Value
conflicts raise `DiffConflictError`, whose `conflicts` attribute lists the
problems. Invalid structure raises `WranglerError`.

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

Run the focused checks:

```powershell
python -m pytest tests\test_ini_wrangler.py tests\test_imports.py -q
```

## INI validation policies

`cap_tools.ini_validation` evaluates declarative checks with `info`, `warning`,
or `error` severities. Findings contain the rule ID, section, concrete member
path, message, actual value, expected constraint, and explicit presence status.
All findings are collected. An error makes the report invalid; info and warning
findings do not. Parser diagnostics remain separate from policy findings.

The opt-in example `ini_files/policies/example-structure.json` checks that
diffraction presets have names and detector-distance entries. It is a structural
example, not an instrument correctness or calibration specification:

```powershell
python -m cap_tools.ini_wrangler validate configuration.ini ini_files/policies/example-structure.json
python -m cap_tools.ini_wrangler validate configuration.ini policy.json -o report.json
python -m cap_tools.ini_wrangler apply configuration.ini policy.json --all -o updated.ini
```

`validate` emits a JSON report to stdout, or to `-o` with the usual overwrite
protection. Exit status is 0 without error findings, 1 with error findings, and
2 for malformed policies, unknown custom checks, execution failures, or I/O
errors. `apply` accepts standalone diffs as before, or a policy containing a
`patch`. It applies the selected operations to a candidate, then runs its checks.
Error findings prevent output even with `--overwrite`; warnings/info are printed
to stderr and allow writing. Use `validate` separately to check the original file
before editing. Validation never executes the patch or mutates the input.

Policies use `format_version: 1`, a `checks` list, and an optional `patch` object
in the existing diff format. Check IDs must be unique; multiple checks may target
the same entry at different severities. This synthetic example demonstrates the
available kinds, without defining instrument requirements:

```json
{
  "format_version": 1,
  "checks": [
    {"id": "required", "kind": "present", "section": "Example", "path": "sa[*].dlimit", "severity": "error"},
    {"id": "deprecated", "kind": "absent", "section": "Example", "path": "ilegacy", "severity": "warning", "message": "Remove this deprecated setting."},
    {"id": "bounds", "kind": "range", "section": "Example", "path": "sa[*].dlimit", "min": 0.0, "max": 100.0, "severity": "warning"},
    {"id": "modes", "kind": "one_of", "section": "Example", "path": "emode", "values": [0, 2, 4], "severity": "error"},
    {"id": "consistency", "kind": "custom", "section": "Example", "path": "sa[*].dlimit", "check": "limit_consistency", "severity": "error"}
  ],
  "patch": {"added_sections": [], "removed_sections": [], "changes": []}
}
```

`present` checks existence, so an empty string counts as present. `absent` reports
any existing entry. These two kinds can also target whole objects or arrays.
`range` checks inclusive numeric bounds (one or both required); nonnumeric values
fail. `one_of` compares typed values, so `2` and `2.0` are distinct. Range and
allowed-value checks skip missing entries: pair them with `present` when required.
The optional `message` replaces the built-in finding text.

Array wildcards use `[*]` at any depth, for example
`sa_presets[*].sa_apertures[*].e_apt_size_id`. Each existing array element is
checked independently, in numeric index order, and findings name its index.
Missing properties within existing elements are retained as missing matches.
If an array is missing or has the wrong shape, a missing match retains `[*]` in
its path; a presence check reports it. Sparse holes are skipped by wildcards;
an explicit `[2]` selector can require that position. No section, member-name,
or recursive globs are supported. Wildcards select checks only; patches continue
to use exact paths.

The Python API includes `ValidationRule`, `ValidationPolicy`, `validate_document`,
`apply_policy`, `expand_entry_path`, and `loads/read/dumps/write_policy`:

```python
from cap_tools.ini_validation import read_policy, validate_document, apply_policy
from cap_tools.ini_wrangler import read_ini, write_ini

document = read_ini("configuration.ini")
policy = read_policy("policy.json")
report = validate_document(document, policy)
print(report.to_dict())
result = apply_policy(document, policy, include_all=True)
write_ini(result.document, "updated.ini")
```

`apply_policy` returns a `PolicyResult` containing the candidate document and
report. It raises `ValidationFailed` with a `.report` attribute on error findings;
malformed policies and broken custom checks raise `WranglerError`. Reports and
policies round-trip through ordinary JSON; report serialization uses `to_dict()`.

Custom checks are manually defined Python functions registered by name. They
receive a `ValidationContext` with the document, rule, and `EntryMatch` containing
the concrete path, presence flag, and value. Return an iterable of messages
(using the rule's severity) or `CustomIssue` objects with individual severities:

```python
from cap_tools.ini_validation import CustomIssue, register_custom_check

def limit_consistency(context):
    if not context.entry.present:
        return [CustomIssue("No limit is available for this preset", "warning")]
    # Inspect context.document.sections here for the related settings.
    return []

register_custom_check("limit_consistency", limit_consistency)
```

Registration is explicit Python code; JSON never imports or evaluates functions.
A Python launcher may register checks before invoking `main_cli()`. Alternatively,
pass a `custom_checks={"limit_consistency": limit_consistency}` mapping to
`validate_document` or `apply_policy`. Custom checks receive isolated copies, so
their edits cannot change the source document or later checks. Return messages
provide the custom finding text. Unknown functions and callback failures are
execution errors, including when their target entry is missing. Policies can
be loaded before their custom functions are registered; names resolve on evaluation.

```powershell
python -m pytest tests/test_ini_validation.py tests/test_ini_wrangler.py tests/test_imports.py -q
```

## INI Wrangler desktop editor

Launch the standalone Tkinter editor with optional current, comparison, and policy
paths. It uses the same readers, writers, diffs, and validation engine as the CLI:

```powershell
python -m cap_tools.ini_wrangler_gui
python -m cap_tools.ini_wrangler_gui configuration.ini --compare reference.ini --policy policy.json
```

Open the **current** configuration as INI or structured JSON. The aligned tree
shows literal CAP member names and JSON-style values beside a read-only comparison.
Arrays align by index, including sparse arrays. Sections, objects, and array
elements can be expanded independently; scrollbars and column dividers provide
space for long names and values. Without a comparison, its column is hidden.
File paths, unsaved status, the loaded policy, parser diagnostic count, and
validation status remain visible.

- Double-click a current value or press **F2** to edit an existing scalar. Text
  values are edited without JSON quotes; numeric fields use CAP's existing type
  decoder. Malformed numeric text is retained with a parser diagnostic.
- Select scalar rows and use **Copy comparison → current** to copy values or add
  missing entries. **Delete current entries** removes only the selected current
  scalar entries. Each operation is transactional; incompatible shapes do not
  partially change the file. Container editing is available through JSON.
- **Undo/Redo** cover cell edits, copies, deletions, JSON commits, and accepted
  patches. Save establishes a new saved state without clearing edit history.
- The **JSON** tab edits the complete current document. **Apply JSON** parses and
  commits it as one history operation; **Revert** restores the committed version.
  Pending drafts suspend saving, editing, and analysis. Invalid JSON leaves the
  committed document intact. Unsaved changes prompt on Open current and Close.
- **Save/Save As** writes the entire current document. Use a `.json` filename for
  structured JSON, or `.ini` for INI. Ordinary saves allow policy errors, so work
  in progress can be saved. Saves cannot overwrite the comparison or active
  policy. Existing round-trip and representability constraints still apply.

The independent filters combine with AND: presence (either file/OR, both/AND,
current only, comparison only), comparison (any/different/equal), findings
(all/with findings/errors/warnings/information), and a case-insensitive search of
full paths and displayed values. One-sided entries count as different. Ancestors
remain visible for context; filtering never limits saves, validation, or exported
diffs.

The resizable lower pane contains **Details** and **Findings**. Details shows full
values and paths, parser diagnostics, and applicable policy findings. Findings
lists every issue, including entries missing from both files. Selecting an issue
clears filters and reveals its entry or nearest existing ancestor; it never adds
data. Full issue details remain available even if its section is entirely absent.

**Load policy** starts validation automatically, and each committed edit triggers
a fresh background evaluation. Pending, ready, failed, and JSON-draft states are
explicit. Results from an older document/policy revision are discarded. **Validate
now** retries an evaluation; **Clear policy** removes its findings. Loading a
policy never applies its patch. Python launchers may register custom checks before
starting `main_cli()`, using the existing validation registration API.

**Export diff** chooses an explicit direction (default current → comparison) and
uses the existing filename-shortening function. It always compares complete
documents. **Apply diff / policy** opens a separate preview: choose additions,
deletions, changes, or all, optionally check old values, then generate a candidate.
The preview lists the candidate diff and all policy findings. Cancellation leaves
current unchanged; **Accept into current** creates one undoable edit and does not
write to disk. Error-level findings from the applied or active policy prevent
acceptance. Existing policy/diff semantics, including preservation of unrelated
target entries, are retained. Policy authoring and instrument-file installation
are outside this GUI.

The Tk-independent view model is in `cap_tools.ini_wrangler_view_model`. Run its
tests together with backend/import checks:

```powershell
python -m pytest tests/test_ini_wrangler_view_model.py tests/test_ini_validation.py tests/test_ini_wrangler.py tests/test_imports.py -q
python -m tests.gui_smoke_ini_wrangler --screenshots .tmp/ini-wrangler-gui
```

The second command requires a live Windows desktop. It exercises real-file
comparison, cell editing, history, filtering, JSON drafts, preview cancellation
and acceptance, missing-entry findings, and single-file/small-window layouts.
Screenshots are local QA artifacts; no source INIs are modified.
