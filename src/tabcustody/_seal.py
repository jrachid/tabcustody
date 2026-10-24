"""Finds the training tables inside a live fitted model and lifts them out for sealing, then puts them back, without a deep copy of the model."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

MIN_ROWS = 20


@dataclass(frozen=True)
class Sealed:
    """The marker left where a training object was, so the saved model carries none of it; `slot` points to the lifted object."""

    slot: int


@dataclass(frozen=True)
class _Ref:
    """One place a training object sits, read and written back the same way whatever holds it."""

    holder: Any
    key: Any
    is_item: bool

    def get(self) -> Any:
        return self.holder[self.key] if self.is_item else getattr(self.holder, self.key)

    def set(self, value: Any) -> None:
        if self.is_item:
            self.holder[self.key] = value
        else:
            setattr(self.holder, self.key, value)


def _named_items(node: Any) -> list[tuple[Any, bool, Any]]:
    """Returns the (key, is_item, value) of a node's direct children, by attribute or by key, the way the scanner reads them."""
    if isinstance(node, dict):
        return [(key, True, value) for key, value in node.items()]
    children: list[tuple[Any, bool, Any]] = []
    slots = getattr(type(node), "__slots__", ())
    for name in slots if isinstance(slots, tuple | list) else (slots,):
        if hasattr(node, name):
            children.append((name, False, getattr(node, name)))
    children += [(name, False, value) for name, value in getattr(node, "__dict__", {}).items()]
    return children


def _is_numeric(value: Any, ndim: int) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == ndim and value.dtype.kind in "iuf"


def _children_for_walk(node: Any) -> list[Any]:
    if isinstance(node, dict):
        return list(node.values())
    if isinstance(node, list | tuple):
        return list(node)
    if isinstance(node, np.ndarray):
        return []
    return [value for _, _, value in _named_items(node)]


def training_refs(model: Any, min_rows: int = MIN_ROWS) -> list[_Ref]:
    """Returns a reference to every training table in the live model: a 2-D numeric array beside a 1-D numeric array of the same length, and that companion."""
    refs: list[_Ref] = []
    seen: set[int] = set()
    stack = [model]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        items = [] if isinstance(node, list | tuple) else _named_items(node)
        companions = {
            len(value): (key, is_item)
            for key, is_item, value in items
            if _is_numeric(value, 1) and len(value) >= min_rows
        }
        for key, is_item, value in items:
            if _is_numeric(value, 2) and value.shape[0] in companions:
                refs.append(_Ref(node, key, is_item))
                companion_key, companion_item = companions[value.shape[0]]
                refs.append(_Ref(node, companion_key, companion_item))
        for child in _children_for_walk(node):
            if id(child) not in seen:
                stack.append(child)
    return _unique(refs)


def _unique(refs: list[_Ref]) -> list[_Ref]:
    out, seen = [], set()
    for ref in refs:
        mark = (id(ref.holder), ref.key, ref.is_item)
        if mark not in seen:
            seen.add(mark)
            out.append(ref)
    return out


@contextmanager
def lifted(model: Any, min_rows: int = MIN_ROWS) -> Iterator[list[Any]]:
    """Swaps each training table in the live model for a `Sealed` marker, yields the lifted objects in slot order, and puts every object back when the block ends, even on error."""
    refs = training_refs(model, min_rows)
    lifted_objects = [ref.get() for ref in refs]
    try:
        for slot, ref in enumerate(refs):
            ref.set(Sealed(slot))
        yield lifted_objects
    finally:
        for ref, original in zip(refs, lifted_objects, strict=True):
            ref.set(original)
