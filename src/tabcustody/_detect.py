"""Finds training tables in an object tree built by the reader, without being given the training data."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from tabcustody._reader import Shell
from tabcustody._reverse import produced, scaler_kind

TRAINING_NAMES = frozenset(
    {"X_train", "X_", "_fit_X", "X_fit_", "support_vectors_", "X_transformed_"}
)


@dataclass(frozen=True)
class Finding:
    """A 2-D numeric array judged to hold training rows; `companion` is the 1-D array with one entry per row, if any."""

    path: str
    shape: tuple[int, ...]
    companion: str | None
    evidence: tuple[str, ...]
    reversible_by: str | None = None
    owner: str | None = None
    hint: str | None = None


def _is_numeric(value: Any, ndim: int) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == ndim and value.dtype.kind in "iuf"


def _is_lookup_grid(array: np.ndarray) -> bool:
    """Tells a lookup table, such as QuantileTransformer's quantiles_, where every column is sorted, from rows of data."""
    return array.shape[0] > 1 and bool(np.all(np.diff(array, axis=0) >= 0))


def _named_children(node: Any) -> list[tuple[str, Any]]:
    if isinstance(node, Shell):
        return [(f".{key}", value) for key, value in node.attributes.items()] + [
            (f"[{key!r}]", value) for key, value in node.entries.items()
        ]
    if isinstance(node, dict):
        return [(f"[{key!r}]", value) for key, value in node.items()]
    return []


def _positional_children(node: Any) -> list[tuple[str, Any]]:
    if isinstance(node, Shell):
        children = [(f"[{index}]", value) for index, value in enumerate(node.items)]
        children += [(f"<arg{index}>", value) for index, value in enumerate(node.args)]
        if node.state is not None:
            children.append(("<state>", node.state))
        return children
    if isinstance(node, list | tuple | set | frozenset):
        return [(f"[{index}]", value) for index, value in enumerate(node)]
    if isinstance(node, np.ndarray) and node.dtype.kind == "O":
        return [(f"[{index}]", value) for index, value in enumerate(node.ravel())]
    return []


def _walk(root: Any) -> Iterator[tuple[str, Any]]:
    seen: set[int] = set()
    stack: list[tuple[str, Any]] = [("", root)]
    while stack:
        path, node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield path, node
        children = _named_children(node) + _positional_children(node)
        stack.extend((path + suffix, child) for suffix, child in reversed(children))


_PIPELINE_STEP = re.compile(r"^(?P<pipeline>.*)\[(?P<index>\d+)\]\[1\]$")


def _parent(path: str) -> str:
    return path[: max(path.rfind("."), path.rfind("["), 0)]


def _feeds(scaler_path: str, table_path: str) -> bool:
    """Tells whether the scaler sits beside the table's owner, or is the pipeline step right before it."""
    owner = _parent(table_path)
    if _parent(scaler_path) == owner:
        return True
    step = _PIPELINE_STEP.match(owner)
    if step is None:
        return False
    return scaler_path == f"{step['pipeline']}[{int(step['index']) - 1}][1]"


def _attach_scalers(
    tables: list[Finding], arrays: dict[str, np.ndarray], scalers: list[tuple[str, Shell]]
) -> list[Finding]:
    attached = []
    for finding in tables:
        feeding = [
            (path, scaler)
            for path, scaler in scalers
            if _feeds(path, finding.path) and produced(arrays[finding.path], scaler)
        ]
        if len(feeding) != 1:
            attached.append(finding)
            continue
        path, scaler = feeding[0]
        note = f"its columns carry the fingerprint of the {scaler_kind(scaler)} at {path}, which reverses it"
        attached.append(replace(finding, reversible_by=path, evidence=(*finding.evidence, note)))
    return attached


def find_tables(tree: Any, min_rows: int = 20, prefix: str = "") -> list[Finding]:
    """Returns the training tables in `tree`: 2-D numeric arrays beside a 1-D array of the same length, then their copies."""

    def _clean(path: str) -> str:
        return prefix + path.removeprefix(".")

    tables: list[Finding] = []
    arrays: dict[str, np.ndarray] = {}
    scalers: list[tuple[str, Shell]] = []
    for path, node in _walk(tree):
        if _is_numeric(node, 2):
            arrays[_clean(path)] = node
        if scaler_kind(node) is not None:
            scalers.append((_clean(path), node))
        siblings = _named_children(node)
        for name, value in siblings:
            if not _is_numeric(value, 2) or value.shape[0] < min_rows or _is_lookup_grid(value):
                continue
            companion = next(
                (
                    other
                    for other, candidate in siblings
                    if _is_numeric(candidate, 1) and len(candidate) == value.shape[0]
                ),
                None,
            )
            if companion is None:
                continue
            evidence = [f"the 1-D array {_clean(path + companion)} has one entry per row"]
            if name.lstrip(".") in TRAINING_NAMES:
                evidence.append(f"{name.lstrip('.')} is a name libraries use for training rows")
            owner = node.name if isinstance(node, Shell) else None
            tables.append(
                Finding(
                    _clean(path + name),
                    value.shape,
                    _clean(path + companion),
                    tuple(evidence),
                    owner=owner,
                )
            )

    confirmed = {finding.path: finding.shape[0] for finding in tables}
    for path, array in arrays.items():
        if path in confirmed:
            continue
        source = next((other for other, rows in confirmed.items() if rows == array.shape[0]), None)
        if source is not None:
            copy_evidence = (f"it has as many rows as the training table {source}",)
            tables.append(Finding(path, array.shape, None, copy_evidence))
    return _attach_scalers(tables, arrays, scalers)
