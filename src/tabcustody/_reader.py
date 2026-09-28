"""Reads a pickle without running it: NumPy arrays are rebuilt, every other class or function becomes a Shell."""

from __future__ import annotations

import collections
import importlib
import io
import pickle
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np


class Shell:
    """An object the file asked for, kept as its dotted name, arguments and state, with none of its code run."""

    dotted_name: ClassVar[str] = ""

    args: tuple[Any, ...]
    attributes: dict[str, Any]
    state: Any
    items: list[Any]
    entries: dict[Any, Any]

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        shell = object.__new__(cls)
        shell.args = args + tuple(kwargs.items())
        shell.attributes = {}
        shell.state = None
        shell.items = []
        shell.entries = {}
        return shell

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def name(self) -> str:
        return type(self).dotted_name

    def __setstate__(self, state: Any) -> None:
        if (
            isinstance(state, tuple)
            and len(state) == 2
            and all(p is None or isinstance(p, dict) for p in state)
        ):
            for part in state:
                self.attributes.update(part or {})
        elif isinstance(state, dict) and all(isinstance(key, str) for key in state):
            self.attributes.update(state)
        else:
            self.state = state

    def append(self, item: Any) -> None:
        self.items.append(item)

    def extend(self, items: Any) -> None:
        self.items.extend(items)

    def __setitem__(self, key: Any, value: Any) -> None:
        self.entries[key] = value

    def __repr__(self) -> str:
        return f"Shell({self.name})"


_shell_classes: dict[str, type[Shell]] = {}


def _shell_class(dotted_name: str) -> type[Shell]:
    if dotted_name not in _shell_classes:
        _shell_classes[dotted_name] = type(dotted_name, (Shell,), {"dotted_name": dotted_name})
    return _shell_classes[dotted_name]


def _numpy_attribute(submodule: str, name: str) -> Any:
    try:
        module = importlib.import_module(f"numpy._core.{submodule}")
    except ImportError:
        module = importlib.import_module(f"numpy.core.{submodule}")
    return getattr(module, name)


def _new_object(cls: Any, *args: Any) -> Any:
    if isinstance(cls, type) and issubclass(cls, Shell):
        return cls.__new__(cls, *args)
    return _shell_class("copyreg.__newobj__")(cls, *args)


def _reconstructor(cls: Any, base: Any, state: Any) -> Any:
    shell = _new_object(cls)
    if state is not None and isinstance(shell, Shell):
        shell.state = state
    return shell


def _encode(text: Any, encoding: str = "utf-8", *rest: Any) -> Any:
    """Stands in for `_codecs.encode`, which pickle protocol 2 uses to carry bytes; only turns text into bytes."""
    if (
        isinstance(text, str)
        and isinstance(encoding, str)
        and encoding.lower() in ("latin1", "latin-1", "utf-8")
    ):
        return text.encode(encoding)
    return _shell_class("_codecs.encode")(text, encoding, *rest)


_NUMPY_BUILDERS = {
    ("multiarray", "_reconstruct"),
    ("multiarray", "scalar"),
    ("numeric", "_frombuffer"),
}

_SAFE_GLOBALS: dict[tuple[str, str], Any] = {
    ("builtins", "set"): set,
    ("builtins", "frozenset"): frozenset,
    ("builtins", "bytearray"): bytearray,
    ("builtins", "complex"): complex,
    ("builtins", "slice"): slice,
    ("builtins", "object"): object,
    ("builtins", "list"): list,
    ("builtins", "dict"): dict,
    ("builtins", "tuple"): tuple,
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("copyreg", "_reconstructor"): _reconstructor,
    ("copyreg", "__newobj__"): _new_object,
    ("_codecs", "encode"): _encode,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _SAFE_GLOBALS:
            return _SAFE_GLOBALS[(module, name)]
        prefix, _, submodule = module.rpartition(".")
        if prefix in ("numpy.core", "numpy._core") and (submodule, name) in _NUMPY_BUILDERS:
            return _numpy_attribute(submodule, name)
        if module.startswith("numpy.dtypes") and name.endswith("DType"):
            return getattr(importlib.import_module("numpy.dtypes"), name)
        return _shell_class(f"{module}.{name}")

    def persistent_load(self, pid: Any) -> Shell:
        return _shell_class("pickle.persistent_id")(pid)


def read_bytes(data: bytes) -> Any:
    """Returns the object tree of a pickle, rebuilding NumPy arrays and turning every other class into a Shell."""
    return _Unpickler(io.BytesIO(data)).load()


def read(path: str | Path) -> Any:
    """Returns the object tree of the pickle at `path`; nothing the file names is imported or called, NumPy array builders aside."""
    with Path(path).open("rb") as stream:
        return _Unpickler(stream).load()
