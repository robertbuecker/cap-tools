"""Declarative CAP INI checks, array selectors, and optional diff policies."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Callable, Iterable, Mapping

from cap_tools.ini_wrangler import (
    DocumentDiff, IniDocument, Node, Scalar, WranglerError, _unique_object,
    _validate, _write, apply_diff, dumps_diff, loads_diff,
)

_SELECTOR_PART = re.compile(r"([^.*\[\]=\s]+)((?:\[(?:\*|0|[1-9][0-9]*)\])*)\Z")
_LEVELS = ("info", "warning", "error")


def _selector_tokens(path: str) -> list[str | int]:
    if not isinstance(path, str):
        raise WranglerError("Check path must be a string")
    result = []
    for part in path.split("."):
        match = _SELECTOR_PART.fullmatch(part)
        if not match:
            raise WranglerError(f"Invalid check path: {path!r}; use [*] for array wildcards")
        result.append(match[1])
        result.extend("*" if i == "*" else int(i) for i in re.findall(r"\[([^]]+)\]", match[2]))
    if str(result[0]).startswith((";", "#")):
        raise WranglerError(f"Invalid check path: {path!r}")
    return result


def _append_path(path: str, token: str | int) -> str:
    if isinstance(token, int) or token == "*":
        return f"{path}[{token}]"
    return f"{path}.{token}" if path else token


@dataclass(frozen=True)
class EntryMatch:
    path: str
    present: bool
    value: Node | None = None


def expand_entry_path(document: IniDocument, section: str, path: str) -> list[EntryMatch]:
    """Expand [*] over existing non-hole elements, retaining missing child paths.

    An absent/non-array wildcard branch yields one missing match with [*] still
    in its path, allowing presence rules to report missing arrays as well.
    Exact paths may select scalar entries or containers. No other glob is used.
    """
    tokens = _selector_tokens(path)

    def walk(node, position, concrete):
        if position == len(tokens):
            yield EntryMatch(concrete, node is not None, node)
            return
        token = tokens[position]
        if token == "*":
            indices = [i for i, value in enumerate(node) if value is not None] if isinstance(node, list) else []
            if not indices:
                missing = concrete
                for remaining in tokens[position:]:
                    missing = _append_path(missing, remaining)
                yield EntryMatch(missing, False)
            for index in indices:
                yield from walk(node[index], position + 1, _append_path(concrete, index))
        else:
            if isinstance(token, int):
                child = node[token] if isinstance(node, list) and token < len(node) else None
            else:
                child = node.get(token) if isinstance(node, dict) else None
            yield from walk(child, position + 1, _append_path(concrete, token))

    return list(walk(document.sections.get(section), 0, ""))


@dataclass(frozen=True)
class ValidationRule:
    id: str
    kind: str
    section: str
    path: str
    severity: str
    message: str | None = None
    min: int | float | None = None
    max: int | float | None = None
    values: list[Scalar] | None = None
    check: str | None = None


@dataclass(frozen=True)
class ValidationFinding:
    rule_id: str
    severity: str
    section: str
    path: str
    message: str
    present: bool
    actual: Node | None
    expected: object = None


@dataclass
class ValidationReport:
    findings: list[ValidationFinding] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(f.severity == "error" for f in self.findings)

    def to_dict(self) -> dict:
        return {"valid": not self.has_errors,
                "counts": {level: sum(f.severity == level for f in self.findings) for level in _LEVELS},
                "findings": [asdict(f) for f in self.findings]}


@dataclass
class ValidationPolicy:
    checks: list[ValidationRule] = field(default_factory=list)
    patch: DocumentDiff | None = None


def _finite_number(value) -> bool:
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _validate_policy(policy: ValidationPolicy) -> None:
    if not isinstance(policy, ValidationPolicy) or not isinstance(policy.checks, list):
        raise WranglerError("Expected a ValidationPolicy with a checks list")
    seen = set()
    for rule in policy.checks:
        if not isinstance(rule, ValidationRule):
            raise WranglerError("Expected ValidationRule in checks")
        if not isinstance(rule.id, str) or not rule.id.strip() or rule.id in seen:
            raise WranglerError(f"Missing or duplicate check id: {rule.id!r}")
        seen.add(rule.id)
        if rule.kind not in ("present", "absent", "range", "one_of", "custom"):
            raise WranglerError(f"Unknown check kind: {rule.kind!r}")
        if rule.severity not in _LEVELS:
            raise WranglerError(f"Invalid severity on {rule.id!r}")
        if not isinstance(rule.section, str):
            raise WranglerError("Check section must be a string")
        _validate(IniDocument({rule.section: {}}))
        _selector_tokens(rule.path)
        if rule.message is not None and not isinstance(rule.message, str):
            raise WranglerError("Check message must be a string")
        if rule.kind != "range" and (rule.min is not None or rule.max is not None):
            raise WranglerError("Bounds are only allowed for range checks")
        if rule.kind != "one_of" and rule.values is not None:
            raise WranglerError("Values are only allowed for one_of checks")
        if rule.kind != "custom" and rule.check is not None:
            raise WranglerError("Function names are only allowed for custom checks")
        if rule.kind == "range":
            bounds = [v for v in (rule.min, rule.max) if v is not None]
            if not bounds or not all(_finite_number(v) for v in bounds):
                raise WranglerError("Range needs at least one finite numeric bound")
            if rule.min is not None and rule.max is not None and rule.min > rule.max:
                raise WranglerError("Range min must not exceed max")
        if rule.kind == "one_of":
            if (not isinstance(rule.values, list) or not rule.values
                    or any(type(v) is not str and not _finite_number(v) for v in rule.values)):
                raise WranglerError("one_of needs a nonempty list of strings/finite numbers")
        if rule.kind == "custom" and (not isinstance(rule.check, str) or not rule.check.strip()):
            raise WranglerError("Custom check needs a registered function name")
    if policy.patch is not None:
        dumps_diff(policy.patch)


def loads_policy(text: str, *, source: str = "<memory>") -> ValidationPolicy:
    try:
        data = json.loads(text, object_pairs_hook=_unique_object)
        if (not isinstance(data, dict) or not {"format_version", "checks"} <= data.keys()
                or data.keys() - {"format_version", "checks", "patch"}
                or type(data["format_version"]) is not int or data["format_version"] != 1
                or not isinstance(data["checks"], list)):
            raise WranglerError("Expected format_version=1, checks, and optional patch")
        rules = []
        for item in data["checks"]:
            if not isinstance(item, dict):
                raise WranglerError("Each check must be an object")
            allowed = {"id", "kind", "section", "path", "severity", "message"}
            allowed.update({"range": ("min", "max"), "one_of": ("values",),
                            "custom": ("check",)}.get(item.get("kind"), ()))
            if item.keys() - allowed or any(value is None for value in item.values()):
                raise WranglerError("Unexpected check fields or null values")
            rules.append(ValidationRule(**item))
        patch = loads_diff(json.dumps(data["patch"]), source=source) if "patch" in data else None
        policy = ValidationPolicy(rules, patch)
        _validate_policy(policy)
        return policy
    except (ValueError, TypeError, OverflowError) as exc:
        raise WranglerError(f"{source}: {exc}") from exc


def read_policy(path: str | Path) -> ValidationPolicy:
    return loads_policy(Path(path).read_text(encoding="utf-8-sig"), source=str(path))


def dumps_policy(policy: ValidationPolicy) -> str:
    _validate_policy(policy)
    checks = [{k: v for k, v in asdict(rule).items() if v is not None} for rule in policy.checks]
    data = {"format_version": 1, "checks": checks}
    if policy.patch is not None:
        data["patch"] = asdict(policy.patch)
    return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def write_policy(policy: ValidationPolicy, path: str | Path, *, overwrite: bool = False) -> None:
    _write(path, dumps_policy(policy), overwrite)


@dataclass(frozen=True)
class ValidationContext:
    document: IniDocument
    rule: ValidationRule
    entry: EntryMatch


@dataclass(frozen=True)
class CustomIssue:
    message: str
    severity: str | None = None


CustomCheck = Callable[[ValidationContext], Iterable[str | CustomIssue]]
CUSTOM_CHECKS: dict[str, CustomCheck] = {}


def register_custom_check(name: str, function: CustomCheck) -> None:
    if not isinstance(name, str) or not name.strip() or not callable(function):
        raise WranglerError("Custom checks need a name and callable")
    if name in CUSTOM_CHECKS:
        raise WranglerError(f"Custom check already registered: {name!r}")
    CUSTOM_CHECKS[name] = function


def validate_document(
    document: IniDocument, policy: ValidationPolicy, *,
    custom_checks: Mapping[str, CustomCheck] | None = None,
) -> ValidationReport:
    """Collect findings without applying the policy patch or mutating the input."""
    _validate(document)
    _validate_policy(policy)
    functions = {**CUSTOM_CHECKS, **(custom_checks or {})}
    for rule in policy.checks:
        if rule.kind == "custom" and not callable(functions.get(rule.check)):
            raise WranglerError(f"Unknown custom check: {rule.check!r}")
    report = ValidationReport()
    for rule in policy.checks:
        for entry in expand_entry_path(document, rule.section, rule.path):
            message, expected = None, None
            if rule.kind == "present" and not entry.present:
                message, expected = "Required entry or array is missing", "present"
            elif rule.kind == "absent" and entry.present:
                message, expected = "Entry must not be present", "absent"
            elif rule.kind == "range" and entry.present:
                expected = {k: v for k, v in (("min", rule.min), ("max", rule.max)) if v is not None}
                if (not _finite_number(entry.value)
                        or (rule.min is not None and entry.value < rule.min)
                        or (rule.max is not None and entry.value > rule.max)):
                    message = "Value is not numeric or is outside inclusive bounds"
            elif rule.kind == "one_of" and entry.present:
                expected = rule.values
                if not any(type(entry.value) is type(v) and entry.value == v for v in rule.values):
                    message = "Value is not in the typed allowed-value list"
            elif rule.kind == "custom":
                try:
                    context = deepcopy(ValidationContext(document, rule, entry))
                    issues = functions[rule.check](context)
                    if isinstance(issues, (str, bytes)):
                        raise TypeError("Return an iterable of messages/CustomIssue, not a string")
                    for issue in issues:
                        issue = CustomIssue(issue) if isinstance(issue, str) else issue
                        if (not isinstance(issue, CustomIssue) or not isinstance(issue.message, str)
                                or (issue.severity is not None and issue.severity not in _LEVELS)):
                            raise TypeError("Invalid custom check finding")
                        report.findings.append(ValidationFinding(
                            rule.id, issue.severity or rule.severity, rule.section, entry.path,
                            issue.message, entry.present, deepcopy(entry.value)))
                except Exception as exc:
                    raise WranglerError(f"Custom check {rule.check!r} failed at {entry.path}: {exc}") from exc
            if message is not None:
                report.findings.append(ValidationFinding(
                    rule.id, rule.severity, rule.section, entry.path, rule.message or message,
                    entry.present, deepcopy(entry.value), deepcopy(expected)))
    return report


class ValidationFailed(WranglerError):
    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__("Policy validation contains errors; output was not written")


@dataclass
class PolicyResult:
    document: IniDocument
    report: ValidationReport


def apply_policy(
    target: IniDocument, policy: ValidationPolicy, *,
    include_additions: bool = False, include_deletions: bool = False,
    include_changes: bool = False, include_all: bool = False, check_old: bool = False,
    custom_checks: Mapping[str, CustomCheck] | None = None,
) -> PolicyResult:
    """Apply the selected patch, then validate the candidate before returning it."""
    _validate_policy(policy)
    if policy.patch is None:
        raise WranglerError("Apply requires a policy containing a patch")
    result = apply_diff(target, policy.patch, include_additions=include_additions,
                        include_deletions=include_deletions, include_changes=include_changes,
                        include_all=include_all, check_old=check_old)
    report = validate_document(result, policy, custom_checks=custom_checks)
    if report.has_errors:
        raise ValidationFailed(report)
    return PolicyResult(result, report)
