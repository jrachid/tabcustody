"""Saves a live fitted model with its training data lifted out, and restores it, by pickling the model through a pickler that externalises exactly what the scanner would report: tables, target-less tables, copies, column extremes and caches."""

from __future__ import annotations

import io
import pickle
from typing import Any

import numpy as np

from tabcustody._detect import _TABICL_CACHE, find_tables
from tabcustody._reader import read_bytes

MIN_ROWS = 20
_TABPFN_CACHE_PREFIX = "tabpfn.architectures."


class CannotSeal(Exception):
    """Raised when sealing would leave training data a scan still finds, rather than write a file that only looks sealed."""


def _class_name(obj: Any) -> str:
    return f"{type(obj).__module__}.{type(obj).__qualname__}"


def _is_numeric(value: Any, ndim: int) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == ndim and value.dtype.kind in "iuf"


def _named_items(node: Any) -> list[tuple[Any, Any]]:
    """Returns a node's direct children by attribute or by key, the way the scanner reads them."""
    if isinstance(node, dict):
        return list(node.items())
    children: list[tuple[Any, Any]] = []
    slots = getattr(type(node), "__slots__", ())
    for name in (slots,) if isinstance(slots, str) else slots:
        if hasattr(node, name):
            children.append((name, getattr(node, name)))
    children += list(getattr(node, "__dict__", {}).items())
    return children


def _walk_children(node: Any) -> list[Any]:
    if isinstance(node, dict):
        return list(node.values())
    if isinstance(node, list | tuple):
        return list(node)
    if isinstance(node, np.ndarray):
        return []
    return [value for _, value in _named_items(node)]


def _is_cache(value: Any) -> bool:
    """Tells whether `value` is a cache container: a dict or list whose entries are TabICL or TabPFN cache objects."""
    entries = list(value.values()) if isinstance(value, dict) else value
    if not isinstance(value, dict | list) or not entries:
        return False
    names = {_class_name(entry) for entry in entries}
    return names == {_TABICL_CACHE} or all(n.startswith(_TABPFN_CACHE_PREFIX) for n in names)


def sealable_objects(model: Any, min_rows: int = MIN_ROWS) -> list[Any]:
    """Returns every object `seal` must lift from the live model, the live counterpart of what the scanner reports: a 2-D table beside a 1-D companion or as wide as its owner's n_features_in_, that companion, 2-D arrays of the same row count (copies), X_min_/X_max_ (column extremes), and whole cache containers."""
    lift: list[Any] = []
    taken: set[int] = set()
    tables: list[np.ndarray] = []

    def take(obj: Any) -> None:
        if id(obj) not in taken:
            taken.add(id(obj))
            lift.append(obj)

    seen: set[int] = set()
    stack = [model]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        items = [] if isinstance(node, list | tuple) else _named_items(node)
        companions = {len(v): v for _, v in items if _is_numeric(v, 1) and len(v) >= min_rows}
        declared = node.n_features_in_ if hasattr(node, "n_features_in_") else None
        low, high = getattr(node, "X_min_", None), getattr(node, "X_max_", None)
        if (
            isinstance(low, np.ndarray)
            and isinstance(high, np.ndarray)
            and _is_numeric(low, 2)
            and low.shape == high.shape
        ):
            take(low)
            take(high)
        for key, value in items:
            if _is_cache(value):
                take(value)
            elif _is_numeric(value, 2) and value.shape[0] >= min_rows:
                if value.shape[0] in companions:
                    take(value)
                    take(companions[value.shape[0]])
                    tables.append(value)
                elif declared == value.shape[1]:
                    take(value)
                    tables.append(value)
        for child in _walk_children(node):
            if id(child) not in seen:
                stack.append(child)

    rows = {table.shape[0] for table in tables}
    seen.clear()
    stack = [model]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if _is_numeric(node, 2) and node.shape[0] in rows:
            take(node)
        for child in _walk_children(node):
            if id(child) not in seen:
                stack.append(child)
    return lift


def lift(model: Any, min_rows: int = MIN_ROWS) -> tuple[bytes, list[Any]]:
    """Pickles the model with every sealable object replaced by a reference, so the bytes carry none of it; returns the bytes and the lifted objects in reference order. Another reference to the same object, even inside a C object's state, becomes the same reference through pickle's memo."""
    wanted = {id(obj) for obj in sealable_objects(model, min_rows)}
    slots: dict[int, int] = {}
    lifted: list[Any] = []

    class _Pickler(pickle.Pickler):
        def persistent_id(self, obj: Any) -> int | None:
            # A class with a custom __getstate__ (TabICL's) pickles a fresh cache object, so the
            # live id never matches; recognising a cache by its contents here catches that copy too.
            if id(obj) not in wanted and not _is_cache(obj):
                return None
            if id(obj) not in slots:
                slots[id(obj)] = len(lifted)
                lifted.append(obj)
            return slots[id(obj)]

    buffer = io.BytesIO()
    _Pickler(buffer, protocol=pickle.DEFAULT_PROTOCOL).dump(model)
    model_bytes = buffer.getvalue()
    left = [finding.path for finding in find_tables(read_bytes(model_bytes))]
    if left:
        raise CannotSeal(
            "training data is kept where sealing cannot reach it, likely inside a C object's state: "
            + ", ".join(left)
        )
    return model_bytes, lifted


def restore(model_bytes: bytes, lifted: list[Any]) -> Any:
    """Rebuilds the model from `lift`'s bytes, putting each lifted object back where its reference was."""

    class _Unpickler(pickle.Unpickler):
        def persistent_load(self, pid: Any) -> Any:
            return lifted[int(pid)]

    return _Unpickler(io.BytesIO(model_bytes)).load()
