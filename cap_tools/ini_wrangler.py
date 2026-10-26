"""Reversible, typed CAP MicroED INI structures (no GUI or instrument access)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
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


def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("to-json", "to-ini", "diff"):
        command = commands.add_parser(name)
        command.add_argument("input", type=Path)
        if name == "diff":
            command.add_argument("other", type=Path)
        command.add_argument("-o", "--output", type=Path, required=True)
        command.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        inputs = [args.input] + ([args.other] if args.command == "diff" else [])
        if any(args.output.resolve() == path.resolve() for path in inputs):
            raise WranglerError("Output must differ from the input files")
        reader = read_json if args.command == "to-ini" else read_ini
        document = reader(args.input)
        documents = [document]
        if args.command == "diff":
            other = read_ini(args.other)
            documents.append(other)
            output = json.dumps(asdict(diff_documents(document, other)), indent=2, ensure_ascii=False) + "\n"
            _write(args.output, output, args.overwrite)
        elif args.command == "to-json":
            write_json(document, args.output, overwrite=args.overwrite)
        else:
            write_ini(document, args.output, overwrite=args.overwrite)
        for doc in documents:
            for diagnostic in doc.diagnostics:
                location = f":{diagnostic.line}" if diagnostic.line is not None else ""
                print(f"{diagnostic.source}{location}: {diagnostic.code}: "
                      f"[{diagnostic.section}] {diagnostic.path}: {diagnostic.message}", file=sys.stderr)
    except (OSError, ValueError) as exc:
        print(f"ini-wrangler: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
