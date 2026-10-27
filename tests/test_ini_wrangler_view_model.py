from copy import deepcopy
from pathlib import Path

import pytest

from cap_tools.ini_validation import (
    ValidationFinding, ValidationPolicy, ValidationReport, ValidationRule,
    ValidationFailed, validate_document,
)
from cap_tools.ini_wrangler import diff_documents, dumps_json, loads_ini, read_ini
from cap_tools.ini_wrangler_view_model import (
    RowFilters, WranglerModel, display_value, evaluate_preview, load_document,
)


def model():
    state = WranglerModel()
    state.set_current(loads_ini("[S]\ni=1\ni_keep=7\nsa[0].i=10\nsa[2].i=30\n[Empty]\n"))
    state.set_comparison(loads_ini("[S]\ni=2\ni_keep=7\ni_new=8\nsa[0].i=10\nsa[1].i=20\n"))
    return state


def key(path):
    from cap_tools.ini_wrangler import _tokens
    return "S", tuple(_tokens(path))


def test_alignment_sparse_arrays_types_and_shapes():
    state = model()
    rows = state.aligned_rows()
    assert rows[key("i")].status == "Different"
    assert rows[key("i_keep")].status == "Equal"
    assert rows[key("sa[1].i")].status == "Comparison only"
    assert rows[key("sa[2].i")].status == "Current only"
    assert [r.path for r in rows.values() if r.scalar and r.path.startswith("sa")] == ["sa[0].i", "sa[1].i", "sa[2].i"]
    assert rows["S", ()].differences == 4
    state.set_comparison(loads_ini("[S]\ni.cchild=shape-change\n"))
    rows = state.aligned_rows()
    assert rows[key("i")].scalar and rows[key("i")].status == "Different"
    assert rows[key("i.cchild")].status == "Comparison only"
    state.set_comparison(None)
    assert all(not r.status for r in state.aligned_rows().values())


@pytest.mark.parametrize("presence,expected", [
    ("Both files (AND)", {"i", "i_keep", "sa[0].i"}),
    ("Current only", {"sa[2].i"}),
    ("Comparison only", {"i_new", "sa[1].i"}),
])
def test_presence_filters(presence, expected):
    state = model()
    rows = state.filtered_rows(RowFilters(presence=presence))
    assert {r.path for r in rows if r.scalar} == expected
    assert ("S", ()) in {r.key for r in rows}


def test_combined_filters_and_ancestor_context():
    state = model()
    state.report = ValidationReport([
        ValidationFinding("check", "warning", "S", "sa[2].i", "bad", True, 30),
        ValidationFinding("missing", "error", "Missing section", "i", "missing", False, None),
    ])
    rows = state.filtered_rows(RowFilters("Current only", "Different", "Warnings", "SA"))
    assert {r.path for r in rows if r.scalar} == {"sa[2].i"}
    assert {r.path for r in rows} >= {"", "sa", "sa[2]", "sa[2].i"}
    assert not state.filtered_rows(RowFilters("Comparison only", "Equal", "Errors"))
    assert state.nearest_key("S", "sa[2].i_missing") == key("sa[2]")
    assert state.nearest_key("Missing section", "i") is None
    assert state.nearest_key("S", "sa[*].i") == key("sa")
    assert len(state.report.findings) == 2
    assert display_value("", True) == '""'
    assert display_value(None, False) == "— missing"


def test_transactional_edit_copy_delete_history():
    state = model()
    original, reference = dumps_json(state.current), dumps_json(state.comparison)
    state.edit_value(key("i"), "bad-number")
    assert state.current.sections["S"]["i"] == "bad-number"
    assert state.current.diagnostics[0].code == "invalid_value"
    state.undo()
    assert dumps_json(state.current) == original and not state.dirty
    state.redo()
    assert state.dirty
    state.copy_comparison([key("i"), key("i_new"), key("sa[1].i")])
    assert state.current.sections["S"]["i"] == 2
    assert state.current.sections["S"]["sa"][1] == {"i": 20}
    state.delete_entries([key("sa[0].i"), key("sa[2].i")])
    assert state.current.sections["S"]["sa"] == [None, {"i": 20}]
    state.undo()
    assert state.current.sections["S"]["sa"][2] == {"i": 30}
    assert dumps_json(state.comparison) == reference
    state.edit_value(key("i"), "123")
    assert not state.can_redo


def test_missing_child_finding_does_not_mark_its_siblings():
    state = model()
    state.report = ValidationReport([
        ValidationFinding("missing", "error", "S", "sa[0].i_missing", "missing", False, None),
    ])
    rows = state.filtered_rows(RowFilters(findings="Errors"))
    assert {r.key for r in rows} == {("S", ()), key("sa"), key("sa[0]")}
    assert not any(r.scalar for r in rows)


def test_shape_collision_and_invalid_selection_roll_back():
    state = model()
    state.set_comparison(loads_ini("[S]\ni.cchild=x\ni_new=9\n"))
    before, token = dumps_json(state.current), state.token
    with pytest.raises(ValueError, match="Conflicting"):
        state.copy_comparison([key("i_new"), key("i.cchild")])
    assert dumps_json(state.current) == before and state.token == token
    with pytest.raises(ValueError, match="scalar"):
        state.delete_entries([key("i"), key("sa")])
    assert dumps_json(state.current) == before


def test_json_draft_blocking_invalid_apply_and_undo():
    state = model()
    before = dumps_json(state.current)
    state.set_json_draft("invalid")
    assert state.draft_pending and state.dirty
    for operation in (lambda: state.edit_value(key("i"), "2"), state.undo,
                      state.export_diff, state.save, lambda: state.copy_comparison([key("i")])):
        with pytest.raises(ValueError, match="draft"):
            operation()
    with pytest.raises(ValueError):
        state.apply_json_draft()
    assert dumps_json(state.current) == before and state.json_draft == "invalid"
    state.revert_json_draft()
    assert not state.draft_pending and not state.dirty
    candidate = deepcopy(state.current)
    candidate.sections["S"]["i"] = 55
    state.set_json_draft(dumps_json(candidate))
    state.apply_json_draft()
    assert state.current.sections["S"]["i"] == 55 and state.can_undo
    state.undo()
    assert dumps_json(state.current) == before


def test_save_preserves_history_allows_policy_errors_and_formats(tmp_path):
    state = model()
    state.set_policy(ValidationPolicy([ValidationRule("bad", "range", "S", "i", "error", min=100)]))
    token, doc, policy = state.validation_request()
    state.accept_validation(token, validate_document(doc, policy))
    assert state.report.has_errors
    state.edit_value(key("i"), "3")
    target = tmp_path / "saved.ini"
    state.save(target)
    assert read_ini(target).sections == state.current.sections
    assert not state.dirty and state.can_undo
    state.undo()
    assert state.dirty
    state.save(tmp_path / "saved.json")
    assert load_document(tmp_path / "saved.json").sections == state.current.sections
    state.set_comparison(state.comparison, target)
    with pytest.raises(ValueError, match="read-only"):
        state.save(target)


def test_revision_checks_reject_stale_results_and_drafts():
    state = model()
    policy = ValidationPolicy([ValidationRule("present", "present", "S", "i", "error")])
    state.set_policy(policy)
    token, document, rules = state.validation_request()
    state.edit_value(key("i"), "2")
    assert not state.accept_validation(token, validate_document(document, rules))
    fresh, _, _ = state.validation_request()
    state.set_json_draft("draft")
    assert not state.accept_validation(fresh, ValidationReport())
    assert state.validation_request() is None
    state.revert_json_draft()
    assert state.token != fresh
    token, _, _ = state.validation_request()
    assert state.accept_validation(token, None, "callback failed")
    assert state.validation_state == "Failed"
    state.set_policy(policy)
    assert not state.accept_validation(token, ValidationReport())
    state.set_policy(None)
    assert state.validation_state == "No policy" and state.report is None


def test_preview_policy_errors_selection_and_stale_acceptance():
    state = model()
    before = dumps_json(state.current)
    patch = state.export_diff()
    request = state.preview_request(patch, include_changes=True)
    preview = evaluate_preview(request)
    assert dumps_json(state.current) == before
    state.accept_preview(preview)
    assert state.current.sections["S"]["i"] == 2 and "i_new" not in state.current.sections["S"]
    state.undo()
    state.edit_value(key("i"), "99")
    with pytest.raises(ValueError, match="new preview"):
        state.accept_preview(preview)
    state.set_policy(ValidationPolicy([ValidationRule("bad", "range", "S", "i", "error", min=100)]))
    with pytest.raises(ValidationFailed):
        evaluate_preview(state.preview_request(patch, include_changes=True))
    state.set_policy(None)
    combined = ValidationPolicy([ValidationRule("bad", "range", "S", "i", "error", min=100)], patch)
    with pytest.raises(ValidationFailed):
        evaluate_preview(state.preview_request(combined, include_all=True))


def test_export_direction_and_filter_does_not_change_documents():
    state = model()
    assert state.export_diff() == diff_documents(state.current, state.comparison)
    assert state.export_diff(True) == diff_documents(state.comparison, state.current)
    before = dumps_json(state.current)
    state.filtered_rows(RowFilters(search="no matches"))
    assert dumps_json(state.current) == before
    assert len(state.export_diff().changes) == 4


def test_real_file_rows_remain_matched_after_edit():
    root = Path(__file__).resolve().parents[1] / "ini_files"
    state = WranglerModel()
    state.set_current(read_ini(root / "CrysAlisMicroEDConfig_ver3_120kV-RC_261021.ini"))
    state.set_comparison(read_ini(root / "CrysAlisMicroEDConfig_ver3_200kV-RC_261023.ini"))
    rows = state.aligned_rows()
    assert sum(r.scalar for r in rows.values()) == 1402
    assert sum(r.scalar and r.status == "Different" for r in rows.values()) == 212
    assert sum(r.scalar and r.status == "Comparison only" for r in rows.values()) == 6
