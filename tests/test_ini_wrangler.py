import json
from pathlib import Path

import pytest

from cap_tools.ini_wrangler import (
    IniDocument, WranglerError, diff_documents, dumps_ini, dumps_json,
    loads_ini, loads_json, main_cli, read_ini, read_json, write_ini, write_json,
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
