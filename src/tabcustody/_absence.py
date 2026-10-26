"""Tells whether given rows are absent from a model file's training tables, without running the file and without ever printing a value of those rows."""

from __future__ import annotations

import json
import pickle
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tabcustody._detect import Finding, _walk, as_table, find_tables
from tabcustody._reader import Shell, UnreadableFileError, read_bytes, read_members
from tabcustody._reverse import reverse

TOLERANCE = 1e-9
DECOYS = 200


@dataclass(frozen=True)
class RowCheck:
    """Where one given row, known by its position in the given rows, was found: training tables holding all its values, and kept column extremes equal to one of its values that no searched table explains."""

    index: int
    tables: tuple[str, ...] = ()
    extremes: tuple[str, ...] = ()

    @property
    def present(self) -> bool:
        return bool(self.tables)

    @property
    def uncertain(self) -> bool:
        """True when no table holds the row but a kept extreme equals one of its values, which may be this person's or another's who shares it."""
        return not self.tables and bool(self.extremes)


@dataclass(frozen=True)
class AbsenceReport:
    """The answer for one file: each row's check, what was searched, what could not be, and how many decoys matched by chance."""

    path: str
    rows: tuple[RowCheck, ...] = ()
    searched: tuple[str, ...] = ()
    unverifiable: tuple[str, ...] = ()
    decoys: int = 0
    decoys_matched: int = 0
    error: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def present(self) -> tuple[int, ...]:
        return tuple(row.index for row in self.rows if row.present)

    @property
    def uncertain(self) -> tuple[int, ...]:
        return tuple(row.index for row in self.rows if row.uncertain)

    @property
    def verified(self) -> bool:
        """True when every place a row could hide was searched: no cache, no table too narrow to hold a row, no decoy matched, no unexplained extreme."""
        return (
            self.error is None
            and not self.unverifiable
            and self.decoys_matched == 0
            and not self.uncertain
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "rows": [
                {
                    "row": r.index,
                    "present": r.present,
                    "uncertain": r.uncertain,
                    "tables": list(r.tables),
                    "extremes": list(r.extremes),
                }
                for r in self.rows
            ],
            "searched": list(self.searched),
            "unverifiable": list(self.unverifiable),
            "decoys": self.decoys,
            "decoys_matched": self.decoys_matched,
            "verified": self.verified,
            "error": self.error,
        }

    def to_text(self) -> str:
        if self.error is not None:
            return f"{self.path}\n  cannot check: {self.error}"
        lines = [self.path]
        for row in self.rows:
            if row.tables:
                lines.append(f"  row {row.index}  present in {', '.join(row.tables)}")
            if row.uncertain:
                lines.append(
                    f"  row {row.index}  may be present: one of its values is a column extreme "
                    f"kept in {', '.join(row.extremes)}, and no table shows whose it is"
                )
            if not row.present and not row.uncertain:
                qualifier = "" if self.verified else ", but see what could not be checked"
                lines.append(f"  row {row.index}  absent from every training table{qualifier}")
        for place in self.unverifiable:
            lines.append(f"  not checked: {place}")
        lines += [f"  note: {note}" for note in self.notes]
        if self.decoys:
            lines.append(
                f"  decoys: {self.decoys_matched} of {self.decoys} matched by chance"
                + (
                    " — the tables cannot tell rows apart, absence is not shown"
                    if self.decoys_matched
                    else ""
                )
            )
        present = len(self.present)
        if present:
            verdict = f"{present} of {len(self.rows)} row(s) present"
        elif self.verified:
            verdict = f"all {len(self.rows)} row(s) absent"
        else:
            verdict = f"none of {len(self.rows)} row(s) found, but absence is not shown"
        lines.append(f"verdict: {verdict}")
        return "\n".join(lines)


def _sealed(path: str | Path) -> bool:
    if not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as archive:
        if "manifest.json" not in archive.namelist():
            return False
        try:
            manifest = json.loads(archive.read("manifest.json"))
        except ValueError:
            return False
    return isinstance(manifest, dict) and str(manifest.get("format", "")).startswith(
        "tabcustody-seal/"
    )


def _tolerance(stored: np.ndarray) -> float:
    """Returns the relative tolerance a table stored with this dtype allows: a few units of its precision, never less than TOLERANCE, so a float32 table is not called free of rows it holds."""
    if stored.dtype.kind != "f":
        return TOLERANCE
    return max(TOLERANCE, 4 * float(np.finfo(stored.dtype).eps))


def _holds(table: np.ndarray, rows: np.ndarray, tolerance: float = TOLERANCE) -> np.ndarray:
    """Tells, for each row, whether some line of `table` holds all its values, in any columns: a TabPFN member keeps the input columns among transformed ones."""
    table = np.asarray(table, dtype=float)
    found = np.zeros(len(rows), dtype=bool)
    for i, row in enumerate(rows):
        close = np.isclose(table[:, :, None], row[None, None, :], rtol=tolerance, atol=1e-12)
        found[i] = bool(close.any(axis=1).all(axis=1).any())
    return found


def _extremes(node: Any, scaler: Any) -> np.ndarray | None:
    """Returns the smallest and largest value of each column, brought back to the columns' scale by the standard scaler beside them, or None."""
    if not isinstance(node, Shell) or not isinstance(scaler, Shell):
        return None
    parts = [node.attributes.get(name) for name in ("X_min_", "X_max_")]
    scale = [scaler.attributes.get(name) for name in ("mean_", "scale_")]
    if any(part is None for part in parts + scale):
        return None
    low, high = (np.asarray(part, dtype=float).ravel() for part in parts)
    mean, factor = (np.asarray(part, dtype=float).ravel() for part in scale)
    return np.vstack([low * factor + mean, high * factor + mean])


def _decoys(table: np.ndarray, width: int, count: int) -> np.ndarray:
    """Draws rows whose every value comes from the table's own column but from different lines, so a match can only be chance; drawn with a fixed seed."""
    rng = np.random.default_rng(0)
    columns = np.asarray(table, dtype=float)[:, :width]
    picks = rng.integers(0, len(columns), (count, width))
    decoys = columns[picks, np.arange(width)]
    real = {tuple(line) for line in columns}
    return np.array([d for d in decoys if tuple(d) not in real]).reshape(-1, width)


def _explained(
    value: float, column: int, width: int, tables: list[tuple[np.ndarray, float]]
) -> bool:
    """Tells whether a searched table holds this value, in its column when the table is as wide as the rows: then the kept extreme is another person's, since this row is not in that table."""
    for table, tolerance in tables:
        cells = table[:, column] if table.shape[1] == width else table
        if np.isclose(cells, value, rtol=tolerance, atol=1e-12).any():
            return True
    return False


def _members(
    path: str | Path, keys: bytes | Iterable[bytes] | None
) -> list[tuple[str | None, Any]]:
    """Returns the object trees to search: the file's own, or, for a sealed file, the tree of the model its key rebuilds, which runs that model's pickle as unseal does."""
    if not _sealed(path):
        return read_members(path)[1]
    if keys is None:
        raise UnreadableFileError("the file is sealed; pass its key to check it")
    from tabcustody._seal import CannotUnseal, unseal

    try:
        model = unseal(path, keys)
    except CannotUnseal as error:
        raise UnreadableFileError(str(error)) from None
    return [(None, read_bytes(pickle.dumps(model)))]


def check_absent(
    path: str | Path,
    rows: np.ndarray,
    min_rows: int = 20,
    keys: bytes | Iterable[bytes] | None = None,
) -> AbsenceReport:
    """Searches the file for each given row, in every training table after reversing its scaler and among the column extremes; a clear file is never run, a sealed one is rebuilt with `keys`. The report never holds a value of the rows."""
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    try:
        members = _members(path, keys)
    except (UnreadableFileError, OSError) as error:
        return AbsenceReport(str(path), error=str(error))

    width = rows.shape[1]
    checks = [RowCheck(i) for i in range(len(rows))]
    searched: list[str] = []
    unverifiable: list[str] = []
    candidates: list[tuple[np.ndarray, float]] = []
    tables: list[tuple[np.ndarray, float]] = []
    hits: list[tuple[int, int, str]] = []
    notes: list[str] = []
    for member, tree in members:
        prefix = f"{member}:" if member else ""
        nodes: dict[str, Any] = {}
        for raw, node in _walk(tree):
            nodes[prefix + raw.removeprefix(".")] = node
        findings: list[Finding] = find_tables(tree, min_rows=min_rows, prefix=prefix)
        for finding in findings:
            if finding.cache:
                unverifiable.append(f"{finding.path} (key-value cache, derived from the rows)")
                continue
            if finding.extremes:
                kept = _extremes(nodes.get(finding.path), nodes.get(finding.reversible_by or ""))
                if kept is None or kept.shape[1] != width:
                    unverifiable.append(f"{finding.path} (column extremes)")
                    continue
                hit = np.isclose(rows[:, None, :], kept[None, :, :], rtol=1e-6, atol=1e-12)
                for i, column in np.argwhere(hit.any(axis=1)).tolist():
                    hits.append((i, column, finding.path))
                continue
            if finding.copy:
                continue
            table = as_table(nodes.get(finding.path))
            if table is None:
                unverifiable.append(f"{finding.path} (training table not readable)")
                continue
            tolerance = _tolerance(table)
            if finding.reversible_by:
                table = reverse(table, nodes[finding.reversible_by])
            if table.shape[1] < width:
                unverifiable.append(f"{finding.path} (training table narrower than the given rows)")
                continue
            searched.append(finding.path)
            tables.append((table, tolerance))
            if table.shape[1] == width:
                candidates.append((table, tolerance))
            for i in np.flatnonzero(_holds(table, rows, tolerance)).tolist():
                checks[i] = RowCheck(i, (*checks[i].tables, finding.path), checks[i].extremes)
    for i, column, place in hits:
        if checks[i].present or _explained(rows[i, column], column, width, tables):
            continue
        if place not in checks[i].extremes:
            checks[i] = RowCheck(i, checks[i].tables, (*checks[i].extremes, place))
    if not searched and not unverifiable:
        notes.append("the scanner finds no training data in this file")
    decoys = _decoys(candidates[0][0], width, DECOYS) if candidates else np.empty((0, width))
    matched = sum(int(_holds(t, decoys, tol).sum()) for t, tol in candidates) if len(decoys) else 0
    return AbsenceReport(
        str(path),
        tuple(checks),
        tuple(searched),
        tuple(unverifiable),
        len(decoys),
        matched,
        notes=tuple(notes),
    )


def read_rows(path: str | Path) -> np.ndarray:
    """Reads the rows to check from a .npy file or a CSV file with an optional header; an error names the line, never its content."""
    path = Path(path)
    if path.suffix == ".npy":
        loaded: np.ndarray = np.atleast_2d(np.load(path, allow_pickle=False)).astype(float)
        return loaded
    import csv

    values: list[list[float]] = []
    with open(path, newline="") as file:
        for number, cells in enumerate(csv.reader(file), start=1):
            if not cells:
                continue
            try:
                values.append([float(cell) for cell in cells])
            except ValueError:
                if number == 1:
                    continue
                raise ValueError(f"{path}: line {number} is not numeric") from None
    if not values or len({len(line) for line in values}) != 1:
        raise ValueError(f"{path}: expected rows of numbers, all of the same width")
    return np.array(values)
