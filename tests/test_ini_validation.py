import json
from pathlib import Path

import pytest

from cap_tools.ini_validation import (
    CUSTOM_CHECKS, CustomIssue, ValidationFailed, ValidationPolicy, ValidationRule,
    apply_policy, dumps_policy, expand_entry_path, loads_policy, read_policy,
    register_custom_check, validate_document, write_policy,
)
from cap_tools.ini_wrangler import (
    WranglerError, diff_documents, dumps_json, loads_ini, main_cli, read_ini,
)


def rule(kind, path="i", severity="error", **kwargs):
    return ValidationRule(id=kwargs.pop("id", kind), kind=kind, section="S", path=path,
                          severity=severity, **kwargs)


def test_presence_absence_bounds_and_levels():
    document = loads_ini("[S]\ncempty=\ni=2\ni_legacy=1\ndbad=bad\n")
    policy = ValidationPolicy([
        rule("present", "cempty", id="empty-is-present"),
        rule("present", "i_missing", message="Required setting"),
        rule("absent", "i_legacy", severity="warning"),
        rule("range", "i", min=0, max=1, severity="info"),
        rule("range", "dbad", min=0, id="numeric-required"),
        rule("range", "i_missing", min=0, id="skip-missing-range"),
        rule("one_of", "i_missing", values=[1], id="skip-missing-values"),
        rule("one_of", "i", values=[2.0], id="typed-values"),
    ])
    before = dumps_json(document)
    report = validate_document(document, policy)
    assert [(f.rule_id, f.severity) for f in report.findings] == [
        ("present", "error"), ("absent", "warning"), ("range", "info"),
        ("numeric-required", "error"), ("typed-values", "error")]
    assert report.findings[0].message == "Required setting"
    assert report.findings[0].present is False
    assert report.findings[0].actual is None
    assert report.findings[2].expected == {"min": 0, "max": 1}
    assert report.to_dict()["counts"] == {"info": 1, "warning": 1, "error": 3}
    assert report.has_errors
    assert dumps_json(document) == before


@pytest.mark.parametrize("value", [-1, 0, 1])
def test_inclusive_bounds(value):
    document = loads_ini(f"[S]\ni={value}\n")
    assert not validate_document(document, ValidationPolicy([rule("range", min=-1, max=1)])).findings
    assert not validate_document(document, ValidationPolicy([rule("range", min=-1)])).findings
    assert not validate_document(document, ValidationPolicy([rule("range", max=1)])).findings
    assert not validate_document(document, ValidationPolicy([rule("one_of", values=[-1, 0, 1])])).findings


def test_nested_wildcards_missing_fields_and_sparse_indices():
    document = loads_ini("[S]\nsa[0].ia[0]=1\nsa[0].ia[2]=9\nsa[0].cname=first\n"
                         "sa[2].cname=second\nsa[3].ia[0]=3\nsa[3].ia[1]=4\n")
    matches = expand_entry_path(document, "S", "sa[*].ia[*]")
    assert [(m.path, m.present, m.value) for m in matches] == [
        ("sa[0].ia[0]", True, 1), ("sa[0].ia[2]", True, 9),
        ("sa[2].ia[*]", False, None), ("sa[3].ia[0]", True, 3), ("sa[3].ia[1]", True, 4)]
    policy = ValidationPolicy([
        rule("present", "sa[*].ia[*]"),
        rule("range", "sa[*].ia[*]", min=0, max=5),
        rule("present", "sa[*].cname", id="names"),
        rule("absent", "sa[*].cname", severity="info"),
    ])
    assert [(f.rule_id, f.path) for f in validate_document(document, policy).findings] == [
        ("present", "sa[2].ia[*]"), ("range", "sa[0].ia[2]"),
        ("names", "sa[3].cname"), ("absent", "sa[0].cname"), ("absent", "sa[2].cname")]
    assert expand_entry_path(document, "S", "sa[1].cname")[0].present is False
    assert expand_entry_path(document, "S", "sa[0]")[0].present is True


@pytest.mark.parametrize("text", ["", "[S]\n", "[S]\nsa=legacy\n"])
def test_wildcard_missing_or_wrong_shape_array(text):
    document = loads_ini(text)
    path = "sa[*].i"
    assert [(m.path, m.present) for m in expand_entry_path(document, "S", path)] == [(path, False)]
    assert len(validate_document(document, ValidationPolicy([rule("present", path)])).findings) == 1
    for kind, options in (("absent", {}), ("range", {"min": 0}), ("one_of", {"values": [1]})):
        assert not validate_document(document, ValidationPolicy([rule(kind, path, **options)])).findings


def test_policy_roundtrip_and_validation_does_not_apply_patch(tmp_path):
    old, new = loads_ini("[S]\ni=0\n"), loads_ini("[S]\ni=1\n")
    policy = ValidationPolicy([rule("range", min=1)], diff_documents(old, new))
    path = tmp_path / "policy.json"
    write_policy(policy, path)
    assert read_policy(path) == policy
    assert loads_policy(dumps_policy(policy)) == policy
    assert validate_document(old, policy).has_errors
    assert old.sections["S"]["i"] == 0
    result = apply_policy(old, policy, include_all=True)
    assert result.document.sections == new.sections
    assert not result.report.findings
    assert old.sections["S"]["i"] == 0
    with pytest.raises(FileExistsError):
        write_policy(policy, path)
    write_policy(policy, path, overwrite=True)


def test_apply_errors_block_and_warnings_do_not():
    old, new = loads_ini("[S]\ni=0\n"), loads_ini("[S]\ni=1\n")
    patch = diff_documents(old, new)
    policy = ValidationPolicy([rule("range", min=2)], patch)
    with pytest.raises(ValidationFailed) as caught:
        apply_policy(old, policy, include_all=True)
    assert caught.value.report.findings[0].actual == 1
    assert old.sections["S"]["i"] == 0
    warning_policy = ValidationPolicy([rule("range", min=2, severity="warning")], patch)
    result = apply_policy(old, warning_policy, include_all=True)
    assert result.document.sections == new.sections
    assert not result.report.has_errors
    assert result.report.findings[0].severity == "warning"
    with pytest.raises(WranglerError, match="containing a patch"):
        apply_policy(old, ValidationPolicy(), include_all=True)


def test_custom_checks_context_levels_and_mutation_isolation(monkeypatch):
    monkeypatch.setitem(CUSTOM_CHECKS, "reserved", lambda context: [])

    def check(context):
        assert context.rule.id == "custom"
        assert context.document.sections["S"]["i_other"] == 7
        context.document.sections["S"]["i_other"] = 99
        if not context.entry.present:
            return [CustomIssue("Missing custom target", "info")]
        return ["Default severity", CustomIssue("Specific severity", "warning")]

    document = loads_ini("[S]\nsa[0].i=1\nsa[1].cname=no-value\ni_other=7\n")
    policy = ValidationPolicy([rule("custom", "sa[*].i", check="manual", severity="error")])
    report = validate_document(document, policy, custom_checks={"manual": check})
    assert [(f.path, f.severity) for f in report.findings] == [
        ("sa[0].i", "error"), ("sa[0].i", "warning"), ("sa[1].i", "info")]
    assert document.sections["S"]["i_other"] == 7
    with pytest.raises(WranglerError, match="Unknown custom check"):
        validate_document(document, policy)
    with pytest.raises(WranglerError, match="already registered"):
        register_custom_check("reserved", check)
    monkeypatch.delitem(CUSTOM_CHECKS, "reserved")
    register_custom_check("reserved", check)
    assert CUSTOM_CHECKS["reserved"] is check
    monkeypatch.delitem(CUSTOM_CHECKS, "reserved")


@pytest.mark.parametrize("callback", [
    lambda context: None, lambda context: "bad return", lambda context: [None],
    lambda context: [CustomIssue("bad level", "fatal")], lambda context: 1 / 0,
])
def test_broken_custom_check_is_execution_error(callback):
    policy = ValidationPolicy([rule("custom", check="broken")])
    with pytest.raises(WranglerError, match="Custom check.*failed"):
        validate_document(loads_ini("[S]\ni=1\n"), policy, custom_checks={"broken": callback})


@pytest.mark.parametrize("changes", [
    {"severity": "fatal"}, {"kind": "unknown"}, {"id": ""}, {"path": "sa[-1].i"},
    {"path": "sa[01].i"}, {"path": "sa*.i"}, {"path": "sa[**].i"},
    {"kind": "range"}, {"kind": "range", "min": True},
    {"kind": "range", "min": 2, "max": 1}, {"kind": "range", "min": float("inf")},
    {"kind": "one_of", "values": []}, {"kind": "one_of", "values": [True]},
    {"kind": "one_of", "values": [float("nan")]}, {"kind": "custom"},
    {"extra": 1}, {"min": 1}, {"values": [1]}, {"check": "x"}, {"message": None},
])
def test_bad_rule_json(changes):
    item = {"id": "rule", "kind": "present", "section": "S", "path": "i", "severity": "error"}
    item.update(changes)
    with pytest.raises(WranglerError, match="policy.json"):
        loads_policy(json.dumps({"format_version": 1, "checks": [item]}), source="policy.json")


@pytest.mark.parametrize("text", [
    '{"format_version":true,"checks":[]}', '{"format_version":2,"checks":[]}',
    '{"format_version":1,"checks":[],"unknown":1}',
    '{"format_version":1,"checks":[],"patch":null}',
    '{"format_version":1,"checks":[],"checks":[]}',
    '{"format_version":1,"checks":{}}',
])
def test_bad_policy_json(text):
    with pytest.raises(WranglerError):
        loads_policy(text)


def test_duplicate_rule_ids():
    with pytest.raises(WranglerError, match="duplicate"):
        dumps_policy(ValidationPolicy([rule("present"), rule("present")]))


@pytest.mark.parametrize("filename", [
    "CrysAlisMicroEDConfig_ver3-RESE_261022.ini",
    "CrysAlisMicroEDConfig_ver3-template.ini",
    "CrysAlisMicroEDConfig_ver3_120kV-RC_261021.ini",
    "CrysAlisMicroEDConfig_ver3_200kV-RC_261022_bad.ini",
    "CrysAlisMicroEDConfig_ver3_200kV-RC_261023.ini",
])
def test_example_policy_on_real_files(filename):
    root = Path(__file__).resolve().parents[1] / "ini_files"
    document = read_ini(root / filename)
    policy = read_policy(root / "policies" / "example-structure.json")
    assert len(expand_entry_path(document, policy.checks[1].section, policy.checks[1].path)) >= 10
    assert not validate_document(document, policy).findings


def test_cli_validate_and_policy_apply(tmp_path, capsys):
    source, policy_file = tmp_path / "source.ini", tmp_path / "policy.json"
    output, report_path = tmp_path / "output.ini", tmp_path / "report.json"
    source.write_text("[S]\ni=0\n", encoding="utf-8")
    old, new = read_ini(source), loads_ini("[S]\ni=1\n")
    policy = ValidationPolicy([rule("range", min=1)], diff_documents(old, new))
    write_policy(policy, policy_file)
    assert main_cli(["validate", str(source), str(policy_file)]) == 1
    assert json.loads(capsys.readouterr().out)["counts"]["error"] == 1
    assert main_cli(["validate", str(source), str(policy_file), "-o", str(report_path)]) == 1
    assert json.loads(report_path.read_text())["findings"][0]["actual"] == 0
    assert main_cli(["apply", str(source), str(policy_file), "--all", "-o", str(output)]) == 0
    assert read_ini(output).sections == new.sections
    assert main_cli(["validate", str(output), str(policy_file)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    preserved = output.read_bytes()
    write_policy(ValidationPolicy([rule("range", min=2)], policy.patch), policy_file, overwrite=True)
    assert main_cli(["apply", str(source), str(policy_file), "--all", "-o", str(output), "--overwrite"]) == 1
    assert output.read_bytes() == preserved
    assert "output was not written" in capsys.readouterr().err
    write_policy(ValidationPolicy([rule("custom", check="unknown")]), policy_file, overwrite=True)
    assert main_cli(["validate", str(source), str(policy_file)]) == 2
    assert main_cli(["validate", str(source), str(policy_file), "-o", str(source), "--overwrite"]) == 2
