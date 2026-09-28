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


_FRAMES = frozenset({"pandas.DataFrame", "pandas.core.frame.DataFrame"})
_SERIES = frozenset({"pandas.Series", "pandas.core.series.Series"})


def _manager(node: Shell) -> Any:
    return node.attributes.get("_mgr", node.attributes.get("_data"))


def _placement(locations: Any, count: int) -> list[int] | None:
    if isinstance(locations, slice):
        columns = list(range(locations.start or 0, locations.stop, locations.step or 1))
    elif _is_numeric(locations, 1):
        columns = [int(column) for column in locations]
    else:
        return None
    return columns if len(columns) == count else None


def _blocks(manager: Any) -> list[tuple[np.ndarray, Any]]:
    """Pairs each block of a pickled pandas BlockManager with the columns it holds, for pandas 1 to 3 layouts."""
    blocks: dict[int, tuple[np.ndarray, Any]] = {}
    for _, node in _walk(manager):
        if isinstance(node, Shell) and node.name.endswith("_unpickle_block") and node.args:
            values, locations = node.args[0], node.args[1] if len(node.args) > 1 else None
        elif isinstance(node, dict) and "values" in node and "mgr_locs" in node:
            values, locations = node["values"], node["mgr_locs"]
        else:
            continue
        if isinstance(values, np.ndarray):
            blocks[id(values)] = (values, locations)
    return list(blocks.values())


def as_table(node: Any) -> np.ndarray | None:
    """Returns `node` as a 2-D numeric array: a NumPy array as is, a pickled pandas DataFrame rebuilt from its numeric blocks."""
    if _is_numeric(node, 2):
        return node  # type: ignore[no-any-return]
    if not (isinstance(node, Shell) and node.name in _FRAMES):
        return None
    columns: dict[int, np.ndarray] = {}
    for values, locations in _blocks(_manager(node)):
        if not _is_numeric(values, 2):
            continue
        placement = _placement(locations, values.shape[0])
        if placement is None:
            return None
        columns.update(zip(placement, values, strict=True))
    if not columns or len({column.shape[0] for column in columns.values()}) != 1:
        return None
    return np.column_stack([columns[index] for index in sorted(columns)])


def as_vector(node: Any) -> np.ndarray | None:
    """Returns `node` as a 1-D numeric array: a NumPy array as is, a pickled pandas Series by its values."""
    if _is_numeric(node, 1):
        return node  # type: ignore[no-any-return]
    if not (isinstance(node, Shell) and node.name in _SERIES):
        return None
    return next((values for _, values in _walk(_manager(node)) if _is_numeric(values, 1)), None)


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
        table = as_table(node)
        if table is not None:
            arrays[_clean(path)] = table
        if scaler_kind(node) is not None:
            scalers.append((_clean(path), node))
        siblings = _named_children(node)
        for name, child in siblings:
            value = as_table(child)
            if value is None or value.shape[0] < min_rows or _is_lookup_grid(value):
                continue
            companion = next(
                (
                    other
                    for other, candidate in siblings
                    if (vector := as_vector(candidate)) is not None
                    and len(vector) == value.shape[0]
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
