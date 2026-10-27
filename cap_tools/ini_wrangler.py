"""Reversible, typed CAP MicroED INI structures (no GUI or instrument access)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
from os.path import commonprefix
from pathlib import Path
import re
import sys
from typing import Iterator


Scalar = str | int | float
Node = Scalar | dict[str, "Node"] | list["Node | None"]
FORMAT_VERSION = 1
_PART = re.compile(r"([^\.\[\]=\s]+)((?:\[(?:0|[1-9][0-9]*)\])*)\Z")
_INTEGER = re.compile(r"[+-]?[0-9]+\Z")
_FLOAT = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_INT_EXCEPTIONS = {"left", "right", "top", "bottom", "x", "y", "cx", "cy", "preset_num"}


class WranglerError(ValueError):
    """Invalid syntax, ambiguous structure, or unrepresentable document."""


@dataclass(frozen=True)
class Diagnostic:
    code: str
    section: str
    path: str
    message: str
    source: str = "<memory>"
    line: int | None = None


def _tokens(path: str) -> list[str | int]:
    result: list[str | int] = []
    for part in path.split("."):
        match = _PART.fullmatch(part)
        if not match:
            raise WranglerError(f"Invalid member path: {path!r}")
        result.append(match[1])
        result.extend(int(i) for i in re.findall(r"\[([0-9]+)\]", match[2]))
    if str(result[0]).startswith((";", "#")):
        raise WranglerError(f"Member path would be an INI comment: {path!r}")
    return result


def _kind(path: str) -> str:
    # Windows geometry and autocenter fields predate the CAP type convention.
    leaf = next(t for t in reversed(_tokens(path)) if isinstance(t, str))
    leaf = leaf.rsplit("->", 1)[-1]
    if leaf in _INT_EXCEPTIONS:
        return "int"
    if leaf.startswith("c"):
        return "str"
    if leaf.startswith(("dw", "i", "l", "w", "e")):
        return "int"
    if leaf.startswith("d"):
        return "float"
    return "unknown"


def _decode(path: str, text: str) -> tuple[Scalar, str | None]:
    kind = _kind(path)
    if kind == "str":
        return text, None
    if kind == "unknown":
        return text, "unknown_type"
    if kind == "int" and _INTEGER.fullmatch(text):
        try:
            return int(text), None
        except ValueError:
            return text, "invalid_value"
    if kind == "float" and _FLOAT.fullmatch(text):
        value = float(text)
        if math.isfinite(value):
            return value, None
    return text, "invalid_value"


def _insert(tree: dict[str, Node], path: str, value: Scalar) -> None:
    tokens = _tokens(path)
    node = tree
    for index, token in enumerate(tokens):
        if isinstance(token, int):
            if not isinstance(node, list):
                raise WranglerError(f"Conflicting array/object shape at {path!r}")
            # Prevent a malformed input from allocating an unbounded sparse list.
            if token > 100_000:
                raise WranglerError(f"Array index exceeds supported limit (100000): {path!r}")
            while len(node) <= token:
                node.append(None)
            existing = node[token]
        else:
            if not isinstance(node, dict):
                raise WranglerError(f"Conflicting object/scalar shape at {path!r}")
            existing = node.get(token)
        if index == len(tokens) - 1:
            if existing is not None:
                raise WranglerError(f"Duplicate or conflicting member at {path!r}")
            node[token] = value
            return
        expected = list if isinstance(tokens[index + 1], int) else dict
        if existing is None:
            existing = expected()
            node[token] = existing
        elif not isinstance(existing, expected):
            raise WranglerError(f"Conflicting tree shape at {path!r}")
        node = existing


def _flatten(node: Node, path: str = "") -> Iterator[tuple[str, Scalar]]:
    if isinstance(node, dict):
        for name in sorted(node):
            yield from _flatten(node[name], f"{path}.{name}" if path else name)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            if value is not None:
                yield from _flatten(value, f"{path}[{index}]")
    else:
        yield path, node


@dataclass
class IniDocument:
    """Section trees; diagnostics are observations, never calibration judgments.

    Edit ``sections`` directly, then use the writers to validate the result.
    Diagnostics describe the last read, rather than tracking subsequent edits.
    """

    sections: dict[str, dict[str, Node]] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def entries(self) -> dict[tuple[str, str], Scalar]:
        return {(section, path): value for section, tree in self.sections.items()
                for path, value in _flatten(tree)}


def loads_ini(text: str, *, source: str = "<memory>") -> IniDocument:
    """Read literal INI values, without interpolation, defaults, or continuations."""
    document = IniDocument()
    section = None
    seen: dict[tuple[str, str], int] = {}
    for line_number, raw in enumerate(text.lstrip("\ufeff").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith((";", "#")):
            continue
        try:
            if line.startswith("["):
                if (not line.endswith("]") or not line[1:-1].strip()
                        or any(c in line[1:-1] for c in "[]")):
                    raise WranglerError("Malformed section header")
                section = line[1:-1]
                if section in document.sections:
                    raise WranglerError(f"Duplicate section {section!r}")
                document.sections[section] = {}
                continue
            if section is None or "=" not in line:
                raise WranglerError("Expected section header or key=value entry")
            path, value = (part.strip() for part in line.split("=", 1))
            identity = section, path
            if identity in seen:
                raise WranglerError(f"Duplicate key {path!r}; first at line {seen[identity]}")
            decoded, code = _decode(path, value)
            _insert(document.sections[section], path, decoded)
            seen[identity] = line_number
            if code:
                document.diagnostics.append(Diagnostic(
                    code, section, path,
                    "Unrecognized type; retained string" if code == "unknown_type"
                    else f"Expected {_kind(path)}; retained string",
                    source, line_number,
                ))
        except WranglerError as exc:
            raise WranglerError(f"{source}:{line_number}: {exc}") from exc
    return document


def read_ini(path: str | Path) -> IniDocument:
    return loads_ini(Path(path).read_text(encoding="utf-8-sig"), source=str(path))


def _validate(document: IniDocument) -> None:
    if not isinstance(document.sections, dict):
        raise WranglerError("Sections must be an object")
    for section, tree in document.sections.items():
        if (not isinstance(section, str) or not section.strip()
                or any(c in section for c in "\r\n[]")):
            raise WranglerError(f"Invalid section name: {section!r}")
        if not isinstance(tree, dict):
            raise WranglerError(f"Section {section!r} must contain an object")

        def visit(node, path=""):
            if isinstance(node, dict):
                if path and not node:
                    raise WranglerError(f"Empty nested object cannot be encoded: {section}/{path}")
                for name, value in node.items():
                    if (not isinstance(name, str) or not _PART.fullmatch(name)
                            or "." in name or "[" in name):
                        raise WranglerError(f"Invalid member name: {name!r}")
                    visit(value, f"{path}.{name}" if path else name)
            elif isinstance(node, list):
                if not node or node[-1] is None:
                    raise WranglerError(f"Empty list/trailing null cannot be encoded: {section}/{path}")
                for index, value in enumerate(node):
                    if index > 100_000:
                        raise WranglerError("Array index exceeds supported limit (100000)")
                    if value is not None:
                        visit(value, f"{path}[{index}]")
            else:
                if type(node) not in (str, int, float):
                    raise WranglerError(f"Expected string or number at {section}/{path}")
                if isinstance(node, str) and (node != node.strip() or any(c in node for c in "\r\n")):
                    raise WranglerError(f"Whitespace cannot round-trip at {section}/{path}")
                decoded, _ = _decode(path, str(node))
                if type(decoded) is not type(node) or decoded != node:
                    raise WranglerError(f"Value/type cannot round-trip at {section}/{path}: {node!r}")
        visit(tree)


def dumps_ini(document: IniDocument) -> str:
    _validate(document)
    blocks = []
    for section in sorted(document.sections):
        lines = [f"[{section}]"]
        lines.extend(f"{path}={value}" for path, value in _flatten(document.sections[section]))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def dumps_json(document: IniDocument) -> str:
    _validate(document)
    return json.dumps({"format_version": FORMAT_VERSION, "sections": document.sections},
                      ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise WranglerError(f"Duplicate JSON member: {key!r}")
        result[key] = value
    return result


def loads_json(text: str, *, source: str = "<memory>") -> IniDocument:
    try:
        data = json.loads(text, object_pairs_hook=_unique_object)
        if (not isinstance(data, dict) or set(data) != {"format_version", "sections"}
                or type(data["format_version"]) is not int or data["format_version"] != FORMAT_VERSION):
            raise WranglerError("Expected format_version=1 and sections")
        document = IniDocument(data["sections"])
        _validate(document)
        # Reuse the INI decoder to produce current diagnostics for retained strings.
        document.diagnostics = loads_ini(dumps_ini(document), source=source).diagnostics
        document.diagnostics = [Diagnostic(d.code, d.section, d.path, d.message, source)
                                for d in document.diagnostics]
        return document
    except (ValueError, TypeError, OverflowError) as exc:
        raise WranglerError(f"{source}: {exc}") from exc


def read_json(path: str | Path) -> IniDocument:
    return loads_json(Path(path).read_text(encoding="utf-8-sig"), source=str(path))


def _write(path: str | Path, text: str, overwrite: bool) -> None:
    with Path(path).open("w" if overwrite else "x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def write_ini(document: IniDocument, path: str | Path, *, overwrite: bool = False) -> None:
    _write(path, dumps_ini(document), overwrite)


def write_json(document: IniDocument, path: str | Path, *, overwrite: bool = False) -> None:
    _write(path, dumps_json(document), overwrite)


@dataclass(frozen=True)
class EntryChange:
    kind: str
    section: str
    path: str
    old: Scalar | None = None
    new: Scalar | None = None


@dataclass(frozen=True)
class DocumentDiff:
    added_sections: list[str]
    removed_sections: list[str]
    changes: list[EntryChange]


def default_diff_name(first: str | Path, second: str | Path) -> str:
    """Return a filename using stems minus their shared leading text.

    Folders and the final extension are discarded. Shared trailing text is kept.
    If removing the prefix would empty either stem, retain both full stems.
    """
    left, right = Path(first).stem, Path(second).stem
    prefix = commonprefix([left, right])
    if prefix and left != prefix and right != prefix:
        left, right = left[len(prefix):], right[len(prefix):]
    return f"DIFF-{left}-VS-{right}.json"


def _validate_diff(diff: DocumentDiff) -> None:
    if not isinstance(diff, DocumentDiff):
        raise WranglerError("Expected a DocumentDiff")
    sides = []
    for names in (diff.added_sections, diff.removed_sections):
        if not isinstance(names, list) or any(not isinstance(n, str) for n in names):
            raise WranglerError("Diff section lists must contain names")
        if len(set(names)) != len(names):
            raise WranglerError("Duplicate diff section name")
        _validate(IniDocument({name: {} for name in names}))
        sides.append(set(names))
    added_sections, removed_sections = sides
    if added_sections & removed_sections:
        raise WranglerError("A section cannot be both added and removed")
    if not isinstance(diff.changes, list):
        raise WranglerError("Diff changes must be a list")
    seen = set()
    before, after = IniDocument(), IniDocument()
    for change in diff.changes:
        if (not isinstance(change, EntryChange) or not isinstance(change.section, str)
                or not isinstance(change.path, str)
                or change.kind not in ("added", "removed", "changed")):
            raise WranglerError("Invalid diff change")
        identity = change.section, change.path
        if identity in seen:
            raise WranglerError(f"Duplicate diff member: {identity!r}")
        seen.add(identity)
        _tokens(change.path)
        if change.section in added_sections and change.kind != "added":
            raise WranglerError(f"Added section contains a non-addition: {identity!r}")
        if change.section in removed_sections and change.kind != "removed":
            raise WranglerError(f"Removed section contains a non-deletion: {identity!r}")
        if change.kind == "added" and (change.old is not None or change.new is None):
            raise WranglerError(f"Addition requires old=null and a new value: {identity!r}")
        if change.kind == "removed" and (change.old is None or change.new is not None):
            raise WranglerError(f"Deletion requires an old value and new=null: {identity!r}")
        if change.kind == "changed":
            if change.old is None or change.new is None:
                raise WranglerError(f"Change requires old and new values: {identity!r}")
            if type(change.old) is type(change.new) and change.old == change.new:
                raise WranglerError(f"Change has identical old/new values: {identity!r}")
        for document, value in ((before, change.old), (after, change.new)):
            if value is not None:
                if type(value) not in (str, int, float):
                    raise WranglerError(f"Diff values must be strings or numbers: {identity!r}")
                try:
                    _insert(document.sections.setdefault(change.section, {}), change.path, value)
                except WranglerError as exc:
                    raise WranglerError(f"Invalid diff member {identity!r}: {exc}") from exc
    _validate(before)
    _validate(after)


def loads_diff(text: str, *, source: str = "<memory>") -> DocumentDiff:
    """Read the existing diff JSON shape, rejecting malformed/ambiguous patches."""
    try:
        data = json.loads(text, object_pairs_hook=_unique_object)
        if not isinstance(data, dict) or set(data) != {"added_sections", "removed_sections", "changes"}:
            raise WranglerError("Expected added_sections, removed_sections, and changes")
        if not isinstance(data["changes"], list):
            raise WranglerError("Diff changes must be a list")
        changes = []
        for item in data["changes"]:
            if not isinstance(item, dict) or set(item) != {"kind", "section", "path", "old", "new"}:
                raise WranglerError("Expected kind, section, path, old, and new on each change")
            changes.append(EntryChange(**item))
        diff = DocumentDiff(data["added_sections"], data["removed_sections"], changes)
        _validate_diff(diff)
        return diff
    except (ValueError, TypeError, OverflowError) as exc:
        raise WranglerError(f"{source}: {exc}") from exc


def read_diff(path: str | Path) -> DocumentDiff:
    return loads_diff(Path(path).read_text(encoding="utf-8-sig"), source=str(path))


def dumps_diff(diff: DocumentDiff) -> str:
    _validate_diff(diff)
    return json.dumps(asdict(diff), indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def write_diff(diff: DocumentDiff, path: str | Path, *, overwrite: bool = False) -> None:
    _write(path, dumps_diff(diff), overwrite)


def diff_documents(old: IniDocument, new: IniDocument) -> DocumentDiff:
    """Compare exact typed values; no numeric tolerances or preset matching."""
    before, after = old.entries(), new.entries()
    changes = []
    for section, path in sorted(before.keys() | after.keys()):
        key = section, path
        if key not in before:
            changes.append(EntryChange("added", section, path, new=after[key]))
        elif key not in after:
            changes.append(EntryChange("removed", section, path, old=before[key]))
        elif type(before[key]) is not type(after[key]) or before[key] != after[key]:
            changes.append(EntryChange("changed", section, path, before[key], after[key]))
    return DocumentDiff(sorted(new.sections.keys() - old.sections.keys()),
                        sorted(old.sections.keys() - new.sections.keys()), changes)


class DiffConflictError(WranglerError):
    """The selected operations cannot all be applied; the target is untouched."""

    def __init__(self, conflicts: list[str]):
        self.conflicts = conflicts
        super().__init__("Diff conflicts:\n" + "\n".join(conflicts))


def apply_diff(
    target: IniDocument, diff: DocumentDiff, *, include_additions: bool = False,
    include_deletions: bool = False, include_changes: bool = False,
    include_all: bool = False, check_old: bool = False,
) -> IniDocument:
    """Apply selected operations to a new document, never mutating ``target``.

    Changes require an existing key, but need not match the diff's old value
    unless ``check_old`` is set. Additions never replace a different value.
    Already-present identical additions and already-missing deletions are no-ops
    (a missing deletion conflicts with ``check_old``). Removed sections are only
    dropped when empty, preserving unrelated keys in a different target file.
    Array positions never shift. All conflicts abort the operation.
    """
    _validate(target)
    _validate_diff(diff)
    kinds = set()
    for enabled, kind in ((include_additions, "added"), (include_deletions, "removed"),
                          (include_changes, "changed")):
        if include_all or enabled:
            kinds.add(kind)
    if not kinds:
        raise WranglerError("Select additions, deletions, changes, or all")
    entries = target.entries()
    sections = set(target.sections)
    conflicts = []
    for change in diff.changes:
        if change.kind not in kinds:
            continue
        key = change.section, change.path
        present = key in entries
        actual = entries.get(key)
        label = f"[{change.section}] {change.path}"
        if change.kind == "added":
            if present and (type(actual) is not type(change.new) or actual != change.new):
                conflicts.append(f"{label}: addition conflicts with target value {actual!r}")
            else:
                entries[key] = change.new
                sections.add(change.section)
        elif not present:
            if change.kind == "changed" or check_old:
                conflicts.append(f"{label}: {change.kind} target is missing")
        elif check_old and (type(actual) is not type(change.old) or actual != change.old):
            conflicts.append(f"{label}: expected old value {change.old!r}, found {actual!r}")
        elif change.kind == "removed":
            del entries[key]
        else:
            entries[key] = change.new
    if conflicts:
        raise DiffConflictError(conflicts)
    if "added" in kinds:
        sections.update(diff.added_sections)
    if "removed" in kinds:
        nonempty = {section for section, _ in entries}
        sections.difference_update(set(diff.removed_sections) - nonempty)
    result = IniDocument({section: {} for section in sorted(sections)})
    for (section, path), value in sorted(entries.items()):
        try:
            _insert(result.sections[section], path, value)
        except WranglerError as exc:
            raise WranglerError(f"Cannot apply diff at [{section}] {path}: {exc}") from exc
    # Rebuild diagnostics and ensure the final shape can be written/read safely.
    return loads_ini(dumps_ini(result), source="<applied>")


def main_cli(argv: list[str] | None = None) -> int:
    from cap_tools.ini_validation import (
        ValidationFailed, apply_policy, loads_policy, read_policy, validate_document,
    )

    def print_findings(report):
        for finding in report.findings:
            print(f"{finding.severity}: {finding.rule_id}: [{finding.section}] "
                  f"{finding.path}: {finding.message}", file=sys.stderr)

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("to-json", "to-ini", "diff", "apply", "validate"):
        command = commands.add_parser(name)
        command.add_argument("input", type=Path)
        if name == "diff":
            command.add_argument("other", type=Path)
        if name == "apply":
            command.add_argument("diff", type=Path, help="diff or policy JSON to apply to the input INI")
            for flag in ("additions", "deletions", "changes", "all"):
                command.add_argument(f"--{flag}", action="store_true", help=f"include {flag}")
            command.add_argument("--check-old", action="store_true",
                                 help="require changes/deletions to match their old values")
        if name == "validate":
            command.add_argument("policy", type=Path, help="validation policy JSON")
        command.add_argument("-o", "--output", type=Path, required=name not in ("diff", "validate"))
        command.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        inputs = [args.input] + ([args.other] if args.command == "diff" else [])
        if args.command == "apply":
            inputs.append(args.diff)
        if args.command == "validate":
            inputs.append(args.policy)
        if args.output is None and args.command == "diff":
            args.output = Path(default_diff_name(args.input, args.other))
        if args.output is not None and any(args.output.resolve() == path.resolve() for path in inputs):
            raise WranglerError("Output must differ from the input files")
        reader = read_json if args.command == "to-ini" else read_ini
        document = reader(args.input)
        documents = [document]
        status = 0
        if args.command == "diff":
            other = read_ini(args.other)
            documents.append(other)
            write_diff(diff_documents(document, other), args.output, overwrite=args.overwrite)
        elif args.command == "apply":
            text = args.diff.read_text(encoding="utf-8-sig")
            data = json.loads(text, object_pairs_hook=_unique_object)
            options = dict(include_additions=args.additions, include_deletions=args.deletions,
                           include_changes=args.changes, include_all=args.all, check_old=args.check_old)
            if isinstance(data, dict) and {"format_version", "checks"} & data.keys():
                applied = apply_policy(document, loads_policy(text, source=str(args.diff)), **options)
                result = applied.document
                print_findings(applied.report)
            else:
                result = apply_diff(document, loads_diff(text, source=str(args.diff)), **options)
            documents = [result]
            write_ini(result, args.output, overwrite=args.overwrite)
        elif args.command == "validate":
            report = validate_document(document, read_policy(args.policy))
            output = json.dumps(report.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            if args.output is None:
                print(output, end="")
            else:
                _write(args.output, output, args.overwrite)
            status = 1 if report.has_errors else 0
        elif args.command == "to-json":
            write_json(document, args.output, overwrite=args.overwrite)
        else:
            write_ini(document, args.output, overwrite=args.overwrite)
        for doc in documents:
            for diagnostic in doc.diagnostics:
                location = f":{diagnostic.line}" if diagnostic.line is not None else ""
                print(f"{diagnostic.source}{location}: {diagnostic.code}: "
                      f"[{diagnostic.section}] {diagnostic.path}: {diagnostic.message}", file=sys.stderr)
    except ValidationFailed as exc:
        print_findings(exc.report)
        print(f"ini-wrangler: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"ini-wrangler: {exc}", file=sys.stderr)
        return 2
    return status


if __name__ == "__main__":
    raise SystemExit(main_cli())
