# cap-auto Upstream Notes

`cap-tools` now treats `cap_auto.cap_files` as the target home for CAP file parsing and CAP command-generation helpers. These notes track gaps found while moving cap-tools toward that boundary without changing the sibling `cap-auto` repository in this cleanup pass.

## Diffraction Analysis

- `cap_auto.cap_files.get_diff_info` returns lists of dictionaries, while the existing `cap_tools.process.get_diff_info` callers expect pandas `DataFrame` objects. `cap-tools` currently keeps a compatibility wrapper that converts `shell_stats`, `reflections`, and `powder_data` into the historical DataFrame shapes.
- The cap-tools wrapper explicitly loads `path + ".par"` into a provided `CAPInstance` before delegating. If this is intended as canonical behavior, cap-auto could add an option to load the experiment internally when a `CAPInstance` is supplied.
- Historical cap-tools column names include `1/d`, `d-value`, and `intx`; cap-auto uses `inv_d`, `d_value`, and `intensity`. The wrapper currently adds the old aliases.

## Results Viewer CSV

- `cap_auto.cap_files.parse_cap_csv` returns `(experiments, cells, centrings)`. The old cap-tools helper returned an additional all-ones `weights` array for cell clustering. The clustering surface has been removed from cap-tools, so no upstream change is currently required.

## Finalization XML and Summary Parsing

- `cap_auto.cap_files.FinalizationXML` overlaps with `cap_tools.finalization.FinalizationXML`, but the active finalization viewer also needs parsing of `*_red.sum` summary tables and collection-level comparison helpers. Those viewer-specific models remain in cap-tools for now.
