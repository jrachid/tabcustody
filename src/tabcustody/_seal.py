"""Saves a live fitted model with its training tables lifted out, and restores it, by pickling the model through a pickler that externalises those arrays."""

from __future__ import annotations

import io
import pickle
from typing import Any

import numpy as np

MIN_ROWS = 20


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


def _is_numeric(value: Any, ndim: int) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == ndim and value.dtype.kind in "iuf"


def _walk_children(node: Any) -> list[Any]:
    if isinstance(node, dict):
        return list(node.values())
    if isinstance(node, list | tuple):
        return list(node)
    if isinstance(node, np.ndarray):
        return []
    return [value for _, value in _named_items(node)]


def training_arrays(model: Any, min_rows: int = MIN_ROWS) -> list[np.ndarray]:
    """Returns every training array in the live model: a 2-D numeric array beside a 1-D numeric array of the same length, and that companion."""
    found: list[np.ndarray] = []
    seen: set[int] = set()
    stack = [model]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        items = [] if isinstance(node, list | tuple) else _named_items(node)
        companions = {len(v): v for _, v in items if _is_numeric(v, 1) and len(v) >= min_rows}
        for _, value in items:
            if _is_numeric(value, 2) and value.shape[0] in companions:
                found += [value, companions[value.shape[0]]]
        for child in _walk_children(node):
            if id(child) not in seen:
                stack.append(child)
    return found


def lift(model: Any, min_rows: int = MIN_ROWS) -> tuple[bytes, list[np.ndarray]]:
    """Pickles the model with its training arrays replaced by a reference, so the bytes carry none of them; returns the bytes and the lifted arrays in reference order. Any other reference to the same array, even inside a C object's state, becomes the same reference."""
    wanted = {id(array) for array in training_arrays(model, min_rows)}
    slots: dict[int, int] = {}
    lifted: list[np.ndarray] = []

    class _Pickler(pickle.Pickler):
        def persistent_id(self, obj: Any) -> int | None:
            if id(obj) not in wanted:
                return None
            if id(obj) not in slots:
                slots[id(obj)] = len(lifted)
                lifted.append(obj)
            return slots[id(obj)]

    buffer = io.BytesIO()
    _Pickler(buffer, protocol=pickle.DEFAULT_PROTOCOL).dump(model)
    return buffer.getvalue(), lifted


def restore(model_bytes: bytes, lifted: list[np.ndarray]) -> Any:
    """Rebuilds the model from `lift`'s bytes, putting each lifted array back where its reference was."""

    class _Unpickler(pickle.Unpickler):
        def persistent_load(self, pid: Any) -> np.ndarray:
            return lifted[int(pid)]

    return _Unpickler(io.BytesIO(model_bytes)).load()
