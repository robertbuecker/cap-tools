# Specs for INI wrangler

The INI wrangler tool should allow to compare, mangle, and validate the initialization file of CrysAlisPro-ED concerned with the online interface of the XtaLAB Synergy-ED. The filename of the active file is fixed to `CrysAlisProMicroEDConfig_ver3.ini`, in practice, a lot of different versions are present on an instrument that get renamed and overwritten, often causeing a mess.

## File structure

### Contents
The file contents have historically grown and mix a lot of different concern: 
* Some trivial ones (screen arrangement etc)
* Presets for diffraction settings that are frequently edited
* Behavior configuration, some reflecting GUI elements, some only in the file. Includes several deprecated ones, that still may be used to enforce legacy behaviors
* Calibrated machine parameters, which are set during installation and quasi-immutable. However, on some instances, different calibration sets (e.g., for different beam voltages) are required.
Depending on which of those groups an entry belongs to, typically requested behavior of the tool may be different. Note that this is actually per-entry, not neccesarily per-section.

### Conventions
While formally an INI file, the structure is much richer and rather is equivalent to a type-annotated JSON, or similar:
* The individual entries define a hierarchy below the top-level sections by `.` symbols inside the entry name
* At various hierarchy levels, entries can be vectors/lists of identically-formed elements. Often, this relates to two-dimensional vectors, signaled by a `_xy2` postfix. Another common use are different presets of diffraction configurations, which cover a range of individual settings (basically, a list of dicts).
* The ordering of entries is often random for historical reasons in individual files. It has no semantic meaning and does not have to be preserved
* The first letters of the last level of the entry name defines its data type (int, char, double, enum, short, long, maybe some more)

## Functionality
To be defined and extended.
MVP for now: parse and write the INI files, and generate structured diffs.

## Implementation
* Use Python with Tkinter as UI framework
* Part of the `cap-tools` package, but mostly independent of the rest
* Internal representation should not strictly follow the entry name strings, but the semantics should be decoded into/encoded from Python structures

## Example files
Example INI files are in the `ini_files` folder. Those include:
* `RESE_261022`: from a well-calibrated running instrument (RESE). It may contain a lot of legacy entries that are unused or deprecated, as the machine is quite old.
* `200kV-RC_261023`: similar from an old but good instrument (RC), might even have older deprecated parameters, or sub-optimal flags enforcing legacy behaviors that never were changed.
* `200kV-RC_261022_bad`: very outdated file with missing, unset, or legacy, that can arise from a messy update history.
* `120kV-RC_261021`: recent, well working file from RC, for a different acceleration voltage; hence, electron-optical calibration values may be different