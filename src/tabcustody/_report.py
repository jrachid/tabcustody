"""Turns the findings of a file into a report that names paths, shapes and fixes, and never a value of the table."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tabcustody._detect import Finding, find_tables
from tabcustody._reader import UnreadableFileError, read_members

_KEEP_CONFIDENTIAL = "keep this file as confidential as the data it was fitted on."
_INSTANCE_BASED = f"An instance-based model keeps training rows by design: {_KEEP_CONFIDENTIAL}"

HINTS = (
    (
        "tabicl.",
        (
            "TabICL can save without the rows: fit with kv_cache=True, then call "
            "save(path, save_training_data=False); the file then keeps the model's key-value cache instead."
        ),
    ),
    (
        "tabpfn.",
        (
            "TabPFN keeps the rows in its default fit mode. Releases that include PriorLabs/TabPFN "
            "pull request 1323 drop them once the caches are built when the model is fitted with "
            f"fit_mode='fit_with_cache'; otherwise {_KEEP_CONFIDENTIAL}"
        ),
    ),
    ("tabdpt.", f"TabDPT has no save function and keeps its training rows: {_KEEP_CONFIDENTIAL}"),
    ("sklearn.neighbors.", _INSTANCE_BASED),
    ("sklearn.svm.", _INSTANCE_BASED),
)


def _hint(owner: str | None) -> str | None:
    if owner is None:
        return None
    return next((hint for prefix, hint in HINTS if owner.startswith(prefix)), None)


@dataclass(frozen=True)
class Report:
    """What one file carries: its findings, or the reason it could not be read."""

    path: str
    format: str
    findings: tuple[Finding, ...]
    error: str | None = None

    @property
    def carries_training_data(self) -> bool:
        return bool(self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "format": self.format,
            "carries_training_data": self.carries_training_data,
            "error": self.error,
            "findings": [
                {
                    "path": finding.path,
                    "rows": finding.shape[0],
                    "columns": finding.shape[1],
                    "kind": "training table" if finding.companion else "copy",
                    "companion": finding.companion,
                    "reversible_by": finding.reversible_by,
                    "owner": finding.owner,
                    "evidence": list(finding.evidence),
                    "hint": finding.hint,
                }
                for finding in self.findings
            ],
        }

    def to_text(self) -> str:
        lines = [f"{self.path} ({self.format})"]
        if self.error:
            lines.append(f"  unreadable: {self.error}")
        elif not self.findings:
            lines.append("  no training table found")
        for group in _alike(self.findings):
            first, kind = group[0], "training table" if group[0].companion else "copy"
            rows = " or ".join(sorted({str(f.shape[0]) for f in group}, key=int, reverse=True))
            columns = " or ".join(sorted({str(f.shape[1]) for f in group}, key=int, reverse=True))
            count = f"  ({len(group)} alike)" if len(group) > 1 else ""
            path = _pattern(first.path) if len(group) > 1 else first.path
            lines.append(f"  {kind}  {path}  {rows} rows x {columns} columns{count}")
            evidence = [_pattern(reason) if len(group) > 1 else reason for reason in first.evidence]
            lines.extend(f"    why: {reason}" for reason in evidence)
            if first.hint:
                lines.append(f"    fix: {first.hint}")
        return "\n".join(lines)


_INDEX = re.compile(r"\[\d+\]")


def _pattern(text: str) -> str:
    return _INDEX.sub("[*]", text)


def _alike(findings: tuple[Finding, ...]) -> list[list[Finding]]:
    groups: dict[tuple[str, bool, str | None], list[Finding]] = {}
    for finding in findings:
        key = (_pattern(finding.path), finding.companion is None, finding.hint)
        groups.setdefault(key, []).append(finding)
    return list(groups.values())


def scan_file(path: str | Path, min_rows: int = 20) -> Report:
    """Scans one model file without running it; a file it cannot read gives a Report whose `error` says why."""
    try:
        file_format, members = read_members(path)
    except (UnreadableFileError, OSError) as error:
        return Report(str(path), "unknown", (), str(error))
    findings = tuple(
        replace(finding, hint=_hint(finding.owner))
        for member, tree in members
        for finding in find_tables(tree, min_rows=min_rows, prefix=f"{member}:" if member else "")
    )
    return Report(str(path), file_format, findings)
