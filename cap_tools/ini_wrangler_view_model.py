"""Tk-independent editing, alignment, history, and validation state."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path

from cap_tools.ini_wrangler import (
    DocumentDiff, IniDocument, Node, WranglerError, _decode, _insert, _tokens,
    apply_diff, diff_documents, dumps_ini, dumps_json, loads_ini, loads_json,
    read_ini, read_json, write_ini, write_json,
)
from cap_tools.ini_validation import (
    ValidationFailed, ValidationFinding, ValidationPolicy, ValidationReport,
    _selector_tokens, apply_policy, dumps_policy, validate_document,
)

RowKey = tuple[str, tuple[str | int, ...]]
Revision = tuple[int, int]


def member_path(tokens: tuple[str | int, ...]) -> str:
    path = ""
    for token in tokens:
        path += f"[{token}]" if isinstance(token, int) else ("." if path else "") + token
    return path


def key_order(key: RowKey):
    return key[0], tuple((1, token) if isinstance(token, int) else (0, token) for token in key[1])


def load_document(path: str | Path) -> IniDocument:
    return read_json(path) if Path(path).suffix.lower() == ".json" else read_ini(path)


def display_value(value: Node | None, present: bool) -> str:
    if not present:
        return "— missing"
    if isinstance(value, dict):
        return "{…}"
    if isinstance(value, list):
        return f"[{sum(v is not None for v in value)} elements]"
    return json.dumps(value, ensure_ascii=False)


@dataclass
class AlignedRow:
    key: RowKey
    current_present: bool
    comparison_present: bool
    current: Node | None
    comparison: Node | None
    status: str
    findings: list[ValidationFinding] = field(default_factory=list)
    differences: int = 0

    @property
    def path(self):
        return member_path(self.key[1])

    @property
    def parent(self):
        return (self.key[0], self.key[1][:-1]) if self.key[1] else None

    @property
    def scalar(self):
        return ((self.current_present and type(self.current) in (str, int, float))
                or (self.comparison_present and type(self.comparison) in (str, int, float)))


@dataclass(frozen=True)
class RowFilters:
    presence: str = "Either file (OR)"
    comparison: str = "Any"
    findings: str = "All entries"
    search: str = ""


@dataclass(frozen=True)
class PreviewRequest:
    revision: Revision
    current: IniDocument
    operation: DocumentDiff | ValidationPolicy
    active_policy: ValidationPolicy | None
    options: dict


@dataclass(frozen=True)
class PatchPreview:
    revision: Revision
    document: IniDocument
    diff: DocumentDiff
    report: ValidationReport


def evaluate_preview(request: PreviewRequest) -> PatchPreview:
    """Safe to run on a worker; no live model or Tk objects are accessed."""
    report = ValidationReport()
    if isinstance(request.operation, ValidationPolicy):
        applied = apply_policy(request.current, request.operation, **request.options)
        candidate, report = applied.document, applied.report
    else:
        candidate = apply_diff(request.current, request.operation, **request.options)
    active = request.active_policy
    if active is not None and (not isinstance(request.operation, ValidationPolicy)
                               or active.checks != request.operation.checks):
        report.findings.extend(validate_document(candidate, active).findings)
    if report.has_errors:
        raise ValidationFailed(report)
    return PatchPreview(request.revision, candidate, diff_documents(request.current, candidate), report)


class WranglerModel:
    def __init__(self):
        self.current: IniDocument | None = None
        self.comparison: IniDocument | None = None
        self.policy: ValidationPolicy | None = None
        self.current_path: Path | None = None
        self.comparison_path: Path | None = None
        self.policy_path: Path | None = None
        self.revision = 0
        self.policy_revision = 0
        self.report: ValidationReport | None = None
        self.validation_state = "No policy"
        self.validation_error = ""
        self.json_draft = ""
        self._committed_json = ""
        self._saved_json = ""
        self._undo: list[IniDocument] = []
        self._redo: list[IniDocument] = []

    @property
    def token(self) -> Revision:
        return self.revision, self.policy_revision

    @property
    def draft_pending(self) -> bool:
        return self.current is not None and self.json_draft != self._committed_json

    @property
    def dirty(self) -> bool:
        return self.current is not None and (self.draft_pending or self._committed_json != self._saved_json)

    @property
    def can_undo(self):
        return bool(self._undo) and not self.draft_pending

    @property
    def can_redo(self):
        return bool(self._redo) and not self.draft_pending

    def require_ready(self):
        if self.current is None:
            raise WranglerError("Open a current file first")
        if self.draft_pending:
            raise WranglerError("Apply or revert the JSON draft first")

    def _invalidate(self):
        self.report = None
        self.validation_error = ""
        self.validation_state = ("JSON draft pending" if self.draft_pending else "Pending") if self.policy else "No policy"

    def set_current(self, document: IniDocument, path: str | Path | None = None):
        text = dumps_json(document)
        self.current = deepcopy(document)
        self.current_path = Path(path) if path is not None else None
        self._committed_json = self._saved_json = self.json_draft = text
        self._undo.clear()
        self._redo.clear()
        self.revision += 1
        self._invalidate()

    def set_comparison(self, document: IniDocument | None, path: str | Path | None = None):
        self.require_ready()
        if document is not None:
            dumps_json(document)
        self.comparison = deepcopy(document)
        self.comparison_path = Path(path) if path is not None else None

    def set_policy(self, policy: ValidationPolicy | None, path: str | Path | None = None):
        self.require_ready()
        if policy is not None:
            dumps_policy(policy)
        self.policy = deepcopy(policy)
        self.policy_path = Path(path) if path is not None else None
        self.policy_revision += 1
        self._invalidate()

    def _commit(self, document: IniDocument):
        self.require_ready()
        # The same parser used for files regenerates diagnostics after each edit.
        document = loads_ini(dumps_ini(document), source="<edited>")
        text = dumps_json(document)
        if text == self._committed_json:
            return
        self._undo.append(deepcopy(self.current))
        self._redo.clear()
        self.current = document
        self._committed_json = self.json_draft = text
        self.revision += 1
        self._invalidate()

    def set_json_draft(self, text: str):
        if self.current is None:
            raise WranglerError("Open a current file first")
        changed = self.draft_pending
        self.json_draft = text
        if changed != self.draft_pending:
            self.revision += 1
            self._invalidate()

    def apply_json_draft(self):
        document = loads_json(self.json_draft, source="JSON editor")
        draft = self.json_draft
        self.json_draft = self._committed_json
        try:
            self._commit(document)
        except Exception:
            self.json_draft = draft
            raise
        self.json_draft = self._committed_json
        self._invalidate()

    def revert_json_draft(self):
        if self.draft_pending:
            self.revision += 1
        self.json_draft = self._committed_json
        self._invalidate()

    def _replace_entries(self, entries):
        result = IniDocument({section: {} for section in self.current.sections})
        for (section, path), value in entries.items():
            try:
                _insert(result.sections.setdefault(section, {}), path, value)
            except WranglerError as exc:
                raise WranglerError(f"[{section}] {path}: {exc}") from exc
        self._commit(result)

    def edit_value(self, key: RowKey, text: str):
        self.require_ready()
        path = member_path(key[1])
        entries = self.current.entries()
        if (key[0], path) not in entries:
            raise WranglerError("Select an existing current scalar entry")
        entries[key[0], path] = _decode(path, text)[0]
        self._replace_entries(entries)

    def copy_comparison(self, keys: list[RowKey]):
        self.require_ready()
        if self.comparison is None:
            raise WranglerError("Open a comparison file first")
        entries, source = self.current.entries(), self.comparison.entries()
        for section, tokens in keys:
            identity = section, member_path(tokens)
            if identity not in source:
                raise WranglerError("Select comparison scalar entries to copy")
            entries[identity] = source[identity]
        self._replace_entries(entries)

    def delete_entries(self, keys: list[RowKey]):
        self.require_ready()
        entries = self.current.entries()
        for section, tokens in keys:
            identity = section, member_path(tokens)
            if identity not in entries:
                raise WranglerError("Select current scalar entries to delete")
            del entries[identity]
        self._replace_entries(entries)

    def _history(self, source, destination):
        self.require_ready()
        if not source:
            return
        destination.append(deepcopy(self.current))
        self.current = source.pop()
        self._committed_json = self.json_draft = dumps_json(self.current)
        self.revision += 1
        self._invalidate()

    def undo(self):
        self._history(self._undo, self._redo)

    def redo(self):
        self._history(self._redo, self._undo)

    def save(self, path: str | Path | None = None):
        self.require_ready()
        target = Path(path) if path is not None else self.current_path
        if target is None:
            raise WranglerError("Choose a save path")
        if self.comparison_path and target.resolve() == self.comparison_path.resolve():
            raise WranglerError("Save must not overwrite the read-only comparison file")
        if self.policy_path and target.resolve() == self.policy_path.resolve():
            raise WranglerError("Save must not overwrite the active policy")
        writer = write_json if target.suffix.lower() == ".json" else write_ini
        writer(self.current, target, overwrite=True)
        self.current_path = target
        self._saved_json = self._committed_json

    def export_diff(self, reverse=False) -> DocumentDiff:
        self.require_ready()
        if self.comparison is None:
            raise WranglerError("Open a comparison file first")
        return (diff_documents(self.comparison, self.current) if reverse
                else diff_documents(self.current, self.comparison))

    def validation_request(self):
        if self.current is None or self.policy is None or self.draft_pending:
            return None
        self.validation_state = "Pending"
        return self.token, deepcopy(self.current), deepcopy(self.policy)

    def accept_validation(self, token: Revision, report: ValidationReport | None, error: str = "") -> bool:
        if token != self.token or self.draft_pending or self.policy is None:
            return False
        self.report = report
        self.validation_error = error
        self.validation_state = "Failed" if error else "Ready"
        return True

    def preview_request(self, operation, **options) -> PreviewRequest:
        self.require_ready()
        return PreviewRequest(self.token, deepcopy(self.current), deepcopy(operation), deepcopy(self.policy), options)

    def accept_preview(self, preview: PatchPreview):
        self.require_ready()
        if preview.revision != self.token:
            raise WranglerError("The file or policy changed; generate a new preview")
        self._commit(preview.document)

    def nearest_key(self, section: str, path: str, rows=None) -> RowKey | None:
        rows = self.aligned_rows() if rows is None else rows
        tokens = []
        for token in _selector_tokens(path):
            if token == "*":
                break
            tokens.append(token)
        while True:
            key = section, tuple(tokens)
            if key in rows:
                return key
            if not tokens:
                return None
            tokens.pop()

    def aligned_rows(self) -> dict[RowKey, AlignedRow]:
        def nodes(document):
            result = {}

            def walk(section, tokens, value):
                result[section, tokens] = value
                if isinstance(value, dict):
                    for name, child in value.items():
                        walk(section, tokens + (name,), child)
                elif isinstance(value, list):
                    for index, child in enumerate(value):
                        if child is not None:
                            walk(section, tokens + (index,), child)
            if document:
                for section, tree in document.sections.items():
                    walk(section, (), tree)
            return result

        left, right = nodes(self.current), nodes(self.comparison)
        rows = {}
        for key in sorted(left.keys() | right.keys(), key=key_order):
            a, b = left.get(key), right.get(key)
            status = ""
            if self.comparison is not None:
                status = ("Comparison only" if key not in left else "Current only" if key not in right
                          else "Equal" if type(a) is type(b) and a == b else "Different")
            rows[key] = AlignedRow(key, key in left, key in right, a, b, status)
        if self.report:
            for finding in self.report.findings:
                key = self.nearest_key(finding.section, finding.path, rows)
                if key is not None:
                    rows[key].findings.append(finding)
        for row in reversed(list(rows.values())):
            if row.scalar and row.status and row.status != "Equal":
                row.differences += 1
            if row.parent is not None:
                rows[row.parent].differences += row.differences
        return rows

    def filtered_rows(self, filters: RowFilters, rows=None) -> list[AlignedRow]:
        rows = self.aligned_rows() if rows is None else rows
        parents = {r.parent for r in rows.values() if r.parent is not None}
        visible = set()
        level = {"Errors": "error", "Warnings": "warning", "Information": "info"}.get(filters.findings)
        for key, row in rows.items():
            if not row.scalar and key in parents and not row.findings:
                continue
            presence = {"Either file (OR)": True,
                        "Both files (AND)": row.current_present and row.comparison_present,
                        "Current only": row.current_present and not row.comparison_present,
                        "Comparison only": row.comparison_present and not row.current_present}
            if not presence[filters.presence]:
                continue
            if filters.comparison == "Different" and row.status not in ("Different", "Current only", "Comparison only"):
                continue
            if filters.comparison == "Equal" and row.status != "Equal":
                continue
            findings = list(row.findings)
            if filters.findings == "Entries with findings" and not findings:
                continue
            if level and not any(f.severity == level for f in findings):
                continue
            text = f"{key[0]} {row.path} {display_value(row.current, row.current_present)} {display_value(row.comparison, row.comparison_present)}"
            if filters.search.casefold() not in text.casefold():
                continue
            visible.add(key)
            parent = row.parent
            while parent is not None:
                visible.add(parent)
                parent = rows[parent].parent
        return [row for key, row in rows.items() if key in visible]
