import json
from pathlib import Path

import pytest

from cap_tools.ini_wrangler import (
    DiffConflictError, DocumentDiff, EntryChange, IniDocument, WranglerError,
    apply_diff, default_diff_name, diff_documents, dumps_diff, dumps_ini, dumps_json,
    loads_diff, loads_ini, loads_json, main_cli, read_diff, read_ini, read_json,
    write_diff, write_ini, write_json,
)


EXAMPLES = Path(__file__).resolve().parents[1] / "ini_files"
NAMES = [
    "CrysAlisMicroEDConfig_ver3-RESE_261022",
    "CrysAlisMicroEDConfig_ver3-template",
    "CrysAlisMicroEDConfig_ver3_120kV-RC_261021",
    "CrysAlisMicroEDConfig_ver3_200kV-RC_261022_bad",
    "CrysAlisMicroEDConfig_ver3_200kV-RC_261023",
]


@pytest.mark.parametrize("name,count", zip(NAMES, [1463, 554, 1396, 561, 1402]))
def test_example_roundtrips(name, count, tmp_path):
    document = read_ini(EXAMPLES / f"{name}.ini")
    assert len(document.entries()) == count
    assert not [d for d in document.diagnostics if d.code == "invalid_value"]
    direct = tmp_path / "direct.ini"
    exported = tmp_path / "document.json"
    restored = tmp_path / "restored.ini"
    write_ini(document, direct)
    write_json(document, exported)
    write_ini(read_json(exported), restored)
    for result in (read_ini(direct), read_ini(restored)):
        assert result.sections == document.sections
        assert diff_documents(document, result).changes == []
    assert (EXAMPLES / "json" / f"{name}.json").read_text(encoding="utf-8") == dumps_json(document)


def test_nested_types_and_exceptions():
    document = loads_ini("""[Main]
s.presets[0].cpresetname_64=Standard
s.presets[0].ddetectordistance=650
s.presets[0].l_deflector_pla_xy2[1]=32768
s.presets[0].l_deflector_pla_xy2[0]=32767
s.presets[0].e_probemode=2
s.presets[0].iuse=1
s.size.cx=120
s.size.cy=135
s.rect.left=-10
psEDSyncUnitSettings->lsync_unit_frame_queue_maxitems=10
preset_num=2
preset_1=Small
[Empty]
""")
    preset = document.sections["Main"]["s"]["presets"][0]
    assert preset == {"cpresetname_64": "Standard", "ddetectordistance": 650.0,
                      "l_deflector_pla_xy2": [32767, 32768], "e_probemode": 2, "iuse": 1}
    assert type(preset["iuse"]) is int
    assert document.sections["Main"]["s"]["size"] == {"cx": 120, "cy": 135}
    assert document.sections["Main"]["s"]["rect"]["left"] == -10
    assert document.sections["Main"]["psEDSyncUnitSettings->lsync_unit_frame_queue_maxitems"] == 10
    assert document.sections["Main"]["preset_num"] == 2
    assert document.sections["Empty"] == {}
    assert loads_json(dumps_json(document)).sections == document.sections


def test_literals_and_diagnostics():
    document = loads_ini("\ufeff; comment\n# comment\n[S]\ncpath=C:\\100%\\a=b;#x\ncempty=\n"
                         "mystery=001\nibad=oops\ndbad=NaN\ndoverflow=1e999\n", source="example.ini")
    assert document.sections["S"]["cpath"] == "C:\\100%\\a=b;#x"
    assert document.sections["S"]["cempty"] == ""
    assert document.sections["S"]["mystery"] == "001"
    assert [(d.code, d.line) for d in document.diagnostics] == [
        ("unknown_type", 6), ("invalid_value", 7), ("invalid_value", 8), ("invalid_value", 9)]
    assert all(d.source == "example.ini" for d in document.diagnostics)
    restored = loads_json(dumps_json(document))
    assert restored.sections == document.sections
    assert sorted((d.code, d.path) for d in restored.diagnostics) == sorted(
        (d.code, d.path) for d in document.diagnostics)
    assert all(d.line is None for d in restored.diagnostics)


def test_sparse_lists_numeric_order_and_determinism():
    document = loads_ini("[Z]\nia[10]=10\nia[2]=2\n[A]\n")
    assert document.sections["Z"]["ia"] == [None, None, 2] + [None] * 7 + [10]
    output = dumps_ini(document)
    assert output == "[A]\n\n[Z]\nia[2]=2\nia[10]=10\n"
    assert loads_json(dumps_json(document)).sections == document.sections
    assert dumps_ini(loads_ini(output)) == output
    assert loads_ini("[S]\nda[0][1]=1e-3\n").sections["S"]["da"] == [[None, 0.001]]


@pytest.mark.parametrize("body", [
    "key=1", "[S", "[]", "[S][bad]", "[S]\nmissing", "[S]\ni=1\ni=2",
    "[S]\n[S]", "[S]\ns=1\ns.i=2", "[S]\ns.i=2\ns=1",
    "[S]\ns.i=2\ns[0]=1", "[S]\nia[-1]=1", "[S]\nia[01]=1",
    "[S]\ni..x=1", "[S]\nia[100001]=1",
])
def test_bad_ini_has_source_location(body):
    with pytest.raises(WranglerError, match=r"broken.ini:\d+:"):
        loads_ini(body, source="broken.ini")


@pytest.mark.parametrize("tree", [
    {"i": True}, {"i": None}, {"d": float("inf")}, {"i": 1.5},
    {"mystery": 1}, {"c": " leading"}, {"c": "a\nb"},
    {"s": {}}, {"ia": []}, {"ia": [1, None]}, {"a.b": "x"},
    {"ia[0]": 1}, {";comment": "x"},
])
def test_unrepresentable_json_edits_rejected(tree):
    document = IniDocument({"S": tree})
    with pytest.raises(WranglerError):
        dumps_json(document)
    with pytest.raises(WranglerError):
        dumps_ini(document)


@pytest.mark.parametrize("text", [
    '{"format_version":2,"sections":{}}',
    '{"format_version":true,"sections":{}}',
    '{"format_version":1,"sections":{"S":{"i":1,"i":2}}}',
    '{"format_version":1,"sections":[]}',
    '{"format_version":1,"sections":{"S":{"d":NaN}}}',
    '{"format_version":1,"sections":{},"extra":1}',
    'invalid',
])
def test_bad_json(text):
    with pytest.raises(WranglerError, match="edited.json"):
        loads_json(text, source="edited.json")


def test_diff_semantics():
    before = loads_ini("[S]\nd=0.20000000000000001\ni=0\nc=\nctext=old\n[Removed]\n")
    after = loads_ini("[Added]\n[S]\nctext=new\nd=0.2\ni=bad\ncnew=\n")
    diff = diff_documents(before, after)
    assert diff.added_sections == ["Added"]
    assert diff.removed_sections == ["Removed"]
    assert [(c.kind, c.path, c.old, c.new) for c in diff.changes] == [
        ("removed", "c", "", None), ("added", "cnew", None, ""),
        ("changed", "ctext", "old", "new"), ("changed", "i", 0, "bad")]
    assert diff_documents(loads_ini("[S]\ni=1\nd=1\n"),
                          loads_ini("[S]\nd=1.0\ni=1\n")).changes == []
    assert diff_documents(IniDocument({"S": {"d": 1}}),
                          IniDocument({"S": {"d": 1.0}})).changes[0].kind == "changed"
    assert [c.path for c in diff_documents(loads_ini("[S]\nia[0]=1\nia[1]=2\n"),
                                          loads_ini("[S]\nia[0]=2\nia[1]=1\n")).changes] == ["ia[0]", "ia[1]"]


def test_cli_and_overwrite_protection(tmp_path, capsys):
    ini = tmp_path / "source.ini"
    exported = tmp_path / "export.json"
    restored = tmp_path / "restored.ini"
    report = tmp_path / "diff.json"
    ini.write_text("[S]\ni=1\n", encoding="utf-8")
    assert main_cli(["to-json", str(ini), "-o", str(exported)]) == 0
    original = exported.read_bytes()
    assert main_cli(["to-json", str(ini), "-o", str(exported)]) == 2
    assert exported.read_bytes() == original
    assert main_cli(["to-json", str(ini), "-o", str(exported), "--overwrite"]) == 0
    data = json.loads(exported.read_text())
    data["sections"]["S"]["i"] = 2
    exported.write_text(json.dumps(data), encoding="utf-8")
    assert main_cli(["to-ini", str(exported), "-o", str(restored)]) == 0
    assert read_ini(restored).sections["S"]["i"] == 2
    assert main_cli(["diff", str(ini), str(restored), "-o", str(report)]) == 0
    assert json.loads(report.read_text())["changes"] == [
        {"kind": "changed", "section": "S", "path": "i", "old": 1, "new": 2}]
    assert main_cli(["to-json", str(ini), "-o", str(ini), "--overwrite"]) == 2
    with pytest.raises(FileExistsError):
        write_ini(read_ini(ini), ini)
    assert "Output must differ" in capsys.readouterr().err


@pytest.mark.parametrize("first,second,expected", [
    ("folder/CrysAlisMicroEDConfig_ver3_120kV-RC_261023.ini",
     "folder/CrysAlisMicroEDConfig_ver3_200kV-RC_261023.ini",
     "DIFF-120kV-RC_261023-VS-200kV-RC_261023.json"),
    ("one/Config_A_shared.ini", "two/Config_B_shared.ini", "DIFF-A_shared-VS-B_shared.json"),
    ("one/alpha.ini", "two/beta.ini", "DIFF-alpha-VS-beta.json"),
    ("one/same.ini", "two/same.ini", "DIFF-same-VS-same.json"),
    ("prefix.ini", "prefix_extra.ini", "DIFF-prefix-VS-prefix_extra.json"),
    ("a.multi.ini", "b.multi.ini", "DIFF-a.multi-VS-b.multi.json"),
])
def test_default_diff_name(first, second, expected):
    assert default_diff_name(Path(first), second) == expected


@pytest.mark.parametrize("first,second", [(NAMES[2], NAMES[4]), (NAMES[0], NAMES[3])])
def test_diff_apply_real_files(first, second, tmp_path):
    before = read_ini(EXAMPLES / f"{first}.ini")
    after = read_ini(EXAMPLES / f"{second}.ini")
    patch = diff_documents(before, after)
    output = tmp_path / "patch.json"
    write_diff(patch, output)
    assert read_diff(output) == patch
    assert loads_diff(dumps_diff(patch)) == patch
    assert apply_diff(before, patch, include_all=True, check_old=True).sections == after.sections
    third = loads_json(dumps_json(before))
    third.sections["Camera Calib"]["smicroedspec"]["scameracalibdata"]["dcameragain_e_to_adu"] = 123.0
    third.sections["Local target settings"] = {"i_keep": 77}
    transferred = apply_diff(third, patch, include_all=True)
    expected = loads_json(dumps_json(after))
    expected.sections["Local target settings"] = {"i_keep": 77}
    # A local edit is replaced only if that field actually occurs in the diff.
    if not any(c.section == "Camera Calib" for c in patch.changes):
        expected.sections["Camera Calib"]["smicroedspec"]["scameracalibdata"]["dcameragain_e_to_adu"] = 123.0
    assert transferred.sections == expected.sections
    assert read_ini(EXAMPLES / f"{first}.ini").sections == before.sections


@pytest.mark.parametrize("add,delete,change", [
    (True, False, False), (False, True, False), (False, False, True),
    (True, True, False), (True, False, True), (False, True, True), (True, True, True),
])
def test_apply_selection_on_different_target(add, delete, change):
    old = loads_ini("[S]\ni_remove=1\ni_change=2\ni_keep=7\nia[0]=10\nia[1]=20\n"
                    "[Removed]\ni=4\n[EmptyOld]\n")
    new = loads_ini("[S]\ni_change=3\ni_keep=7\nia[0]=10\ni_add=5\n[Added]\ni=6\n[EmptyNew]\n")
    target = loads_ini("[S]\ni_remove=99\ni_change=42\ni_keep=777\nia[0]=100\nia[1]=200\n"
                       "[Removed]\ni=44\ni_unrelated=8\n[EmptyOld]\n[TargetOnly]\n")
    original = dumps_json(target)
    result = apply_diff(target, diff_documents(old, new), include_additions=add,
                        include_deletions=delete, include_changes=change)
    assert result.sections["S"]["i_change"] == (3 if change else 42)
    assert result.sections["S"]["i_keep"] == 777
    assert result.sections["S"]["ia"] == ([100] if delete else [100, 200])
    assert ("i_remove" not in result.sections["S"]) == delete
    assert ("i_add" in result.sections["S"]) == add
    assert ("Added" in result.sections) == add
    assert ("EmptyNew" in result.sections) == add
    assert ("EmptyOld" not in result.sections) == delete
    assert result.sections["Removed"].get("i_unrelated") == 8
    assert ("i" not in result.sections["Removed"]) == delete
    assert result.sections["TargetOnly"] == {}
    assert dumps_json(target) == original
    assert loads_ini(dumps_ini(result)).sections == result.sections


def test_apply_missing_conflicting_and_strict_values():
    patch = diff_documents(loads_ini("[S]\ni=1\ni_remove=2\n"),
                           loads_ini("[S]\ni=3\ni_add=4\n"))
    target = loads_ini("[S]\ni_add=99\n")
    original = dumps_json(target)
    with pytest.raises(DiffConflictError) as caught:
        apply_diff(target, patch, include_all=True)
    assert len(caught.value.conflicts) == 2
    assert "target is missing" in str(caught.value)
    assert "addition conflicts" in str(caught.value)
    assert dumps_json(target) == original
    already = loads_ini("[S]\ni=3\ni_add=4\n")
    assert apply_diff(already, patch, include_all=True).sections == already.sections
    different = loads_ini("[S]\ni=55\ni_remove=66\n")
    with pytest.raises(DiffConflictError, match="expected old value"):
        apply_diff(different, patch, include_changes=True, check_old=True)
    with pytest.raises(DiffConflictError, match="target is missing"):
        apply_diff(already, patch, include_deletions=True, check_old=True)
    with pytest.raises(WranglerError, match="Select"):
        apply_diff(target, patch)


def test_apply_array_holes_and_shape_transitions():
    old = loads_ini("[S]\nia[0]=1\nia[1]=2\nia[2]=3\ns=legacy\n")
    new = loads_ini("[S]\nia[0]=1\nia[2]=3\ns.i=4\n")
    patch = diff_documents(old, new)
    result = apply_diff(old, patch, include_all=True)
    assert result.sections == new.sections
    assert result.sections["S"]["ia"] == [1, None, 3]
    with pytest.raises(WranglerError, match="Cannot apply diff"):
        apply_diff(old, patch, include_additions=True)
    reverse = diff_documents(new, old)
    assert apply_diff(new, reverse, include_all=True).sections == old.sections


@pytest.mark.parametrize("patch", [
    {"added_sections": [], "removed_sections": [], "changes": [], "extra": 1},
    {"added_sections": ["S", "S"], "removed_sections": [], "changes": []},
    {"added_sections": ["S"], "removed_sections": ["S"], "changes": []},
    {"added_sections": ["bad\nsection"], "removed_sections": [], "changes": []},
    {"added_sections": "S", "removed_sections": [], "changes": []},
    {"added_sections": [], "removed_sections": [], "changes": {}},
])
def test_invalid_diff_envelopes(patch):
    with pytest.raises(WranglerError, match="patch.json"):
        loads_diff(json.dumps(patch), source="patch.json")


@pytest.mark.parametrize("change", [
    EntryChange("unknown", "S", "i", 1, 2),
    EntryChange("added", "S", "i", 1, 2),
    EntryChange("added", "S", "i"),
    EntryChange("removed", "S", "i", 1, 2),
    EntryChange("removed", "S", "i"),
    EntryChange("changed", "S", "i", None, 2),
    EntryChange("changed", "S", "i", 1, 1),
    EntryChange("changed", "S", "i", True, 2),
    EntryChange("changed", "S", "i", 1, 2.0),
    EntryChange("changed", "S", "d", 1.0, float("nan")),
    EntryChange("changed", "S", "c", "old", " whitespace "),
    EntryChange("changed", "S", "ia[-1]", 1, 2),
    EntryChange("changed", "S", "i..bad", 1, 2),
])
def test_invalid_diff_operations(change):
    patch = DocumentDiff([], [], [change])
    with pytest.raises(WranglerError):
        dumps_diff(patch)
    # Excluding an operation does not make a malformed patch acceptable.
    with pytest.raises(WranglerError):
        apply_diff(loads_ini("[S]\n"), patch, include_additions=True)


def test_diff_duplicate_and_conflicting_operations():
    change = EntryChange("changed", "S", "i", 1, 2)
    for patch in (
        DocumentDiff([], [], [change, change]),
        DocumentDiff(["S"], [], [change]),
        DocumentDiff([], ["S"], [change]),
        DocumentDiff([], [], [EntryChange("added", "S", "s", new="x"),
                              EntryChange("added", "S", "s.i", new=1)]),
    ):
        with pytest.raises(WranglerError):
            dumps_diff(patch)
    for text in (
        '{"added_sections":[],"removed_sections":[],"changes":[],"changes":[]}',
        '{"added_sections":[],"removed_sections":[],"changes":[{"kind":"added"}]}',
    ):
        with pytest.raises(WranglerError):
            loads_diff(text)


def test_diff_default_cli_and_apply(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = Path("Config_A_shared.ini")
    after = Path("Config_B_shared.ini")
    target = Path("third.ini")
    output = Path("applied.ini")
    before.write_text("[S]\ni=1\ni_remove=2\n", encoding="utf-8")
    after.write_text("[S]\ni=3\ni_add=4\n", encoding="utf-8")
    target.write_text("[S]\ni=99\ni_remove=66\ni_keep=55\n", encoding="utf-8")
    name = default_diff_name(before, after)
    assert main_cli(["diff", str(before), str(after)]) == 0
    patch_bytes = Path(name).read_bytes()
    assert main_cli(["diff", str(before), str(after)]) == 2
    assert Path(name).read_bytes() == patch_bytes
    assert main_cli(["diff", str(before), str(after), "--overwrite"]) == 0
    assert main_cli(["apply", str(target), name, "--all", "-o", str(output)]) == 0
    assert read_ini(output).sections["S"] == {"i": 3, "i_add": 4, "i_keep": 55}
    output_bytes = output.read_bytes()
    assert main_cli(["apply", str(target), name, "--all", "-o", str(output)]) == 2
    assert main_cli(["apply", str(target), name, "--all", "--check-old",
                     "-o", str(output), "--overwrite"]) == 2
    assert output.read_bytes() == output_bytes
    assert main_cli(["apply", str(target), name, "--changes", "--deletions",
                     "-o", str(output), "--overwrite"]) == 0
    assert read_ini(output).sections["S"] == {"i": 3, "i_keep": 55}
    assert main_cli(["apply", str(target), name, "-o", str(output), "--overwrite"]) == 2
    assert main_cli(["apply", str(target), name, "--all", "-o", str(target), "--overwrite"]) == 2
    assert main_cli(["apply", str(target), name, "--all", "-o", name, "--overwrite"]) == 2
