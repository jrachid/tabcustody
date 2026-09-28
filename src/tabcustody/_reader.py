"""Reads a pickle without running it: NumPy arrays are rebuilt, every other class or function becomes a Shell."""

from __future__ import annotations

import bz2
import collections
import gzip
import importlib
import io
import lzma
import pickle
import zipfile
import zlib
from pathlib import Path
from typing import Any, BinaryIO, ClassVar, Self, cast

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


class UnreadableFileError(Exception):
    """Raised when a file is neither a pickle, a joblib file nor a zip archive of them, or is malformed."""


_JOBLIB_WRAPPER = "joblib.numpy_pickle.NumpyArrayWrapper"


def _read_exactly(stream: BinaryIO, size: int) -> bytes:
    chunks, remaining = [], size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise UnreadableFileError("the file ends inside an array")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class _Unpickler(pickle._Unpickler):
    # The C unpickler reads ahead, which would swallow the raw array bytes joblib writes between opcodes.

    dispatch: ClassVar[dict[int, Any]] = dict(pickle._Unpickler.dispatch)
    stack: list[Any]

    def __init__(self, stream: BinaryIO) -> None:
        super().__init__(stream)
        self._stream = stream

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

    def _load_build(self) -> None:
        pickle._Unpickler.load_build(self)  # type: ignore[attr-defined]
        top = self.stack[-1]
        if isinstance(top, Shell) and top.name == _JOBLIB_WRAPPER:
            self.stack[-1] = self._read_joblib_array(top)

    def _read_joblib_array(self, wrapper: Shell) -> np.ndarray:
        """Reads the array joblib writes right after its NumpyArrayWrapper, as joblib 1.6 `read_array` does."""
        try:
            dtype = np.dtype(wrapper.attributes["dtype"])
            shape = tuple(int(size) for size in wrapper.attributes["shape"])
        except (KeyError, TypeError, ValueError) as error:
            raise UnreadableFileError(f"malformed joblib array header: {error}") from error
        if dtype.hasobject:
            loaded = _Unpickler(self._stream).load()
            return np.asarray(loaded, dtype=object)
        if wrapper.attributes.get("numpy_array_alignment_bytes") is not None:
            padding = int.from_bytes(_read_exactly(self._stream, 1), "little")
            _read_exactly(self._stream, padding)
        count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        array = np.frombuffer(
            _read_exactly(self._stream, count * dtype.itemsize), dtype=dtype, count=count
        )
        if wrapper.attributes.get("order") == "F":
            return np.asarray(array.reshape(shape[::-1]).transpose())
        return np.asarray(array.reshape(shape))


_Unpickler.dispatch[pickle.BUILD[0]] = _Unpickler._load_build


class _ZlibReader(io.RawIOBase):
    """The zlib stream joblib writes with `compress=` an integer, decompressed as it is read."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._decompressor = zlib.decompressobj()
        self._pending = b""

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        while not self._pending and not self._decompressor.eof:
            compressed = self._stream.read(1 << 16)
            if not compressed:
                break
            self._pending = self._decompressor.decompress(compressed)
        size = min(len(buffer), len(self._pending))
        buffer[:size] = self._pending[:size]
        self._pending = self._pending[size:]
        return size


def _decompressed(stream: BinaryIO) -> BinaryIO:
    head = stream.peek(6)[:6] if hasattr(stream, "peek") else b""
    if head.startswith(b"\x1f\x8b"):
        return cast(BinaryIO, gzip.GzipFile(fileobj=stream))
    if head.startswith(b"BZh"):
        return cast(BinaryIO, bz2.BZ2File(stream))
    if head.startswith((b"\xfd7zXZ\x00", b"]\x00\x00")):
        return cast(BinaryIO, lzma.LZMAFile(stream))
    if head[:1] == b"\x78":
        return cast(BinaryIO, io.BufferedReader(_ZlibReader(stream)))
    return stream


def _load(stream: BinaryIO) -> Any:
    try:
        return _Unpickler(_decompressed(stream)).load()
    except UnreadableFileError:
        raise
    except Exception as error:
        raise UnreadableFileError(
            f"not a readable pickle or joblib file: {type(error).__name__}: {error}"
        ) from error


def read_bytes(data: bytes) -> Any:
    """Returns the object tree of a pickle, rebuilding NumPy arrays and turning every other class into a Shell."""
    return _load(io.BufferedReader(io.BytesIO(data)))


def read(path: str | Path) -> Any:
    """Returns the object tree of the pickle or joblib file at `path`; nothing it names is imported or called, NumPy array builders aside."""
    with Path(path).open("rb") as stream:
        return _load(stream)


def read_members(path: str | Path) -> tuple[str, list[tuple[str | None, Any]]]:
    """Returns the file's format and the object tree of each pickle or joblib member; a lone file has one member named None."""
    if zipfile.is_zipfile(path):
        trees: list[tuple[str | None, Any]] = []
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir() or member.filename.endswith(".json"):
                    continue
                with archive.open(member) as stream:
                    trees.append((member.filename, _load(stream)))  # type: ignore[arg-type]
        return "zip", trees
    return "pickle", [(None, read(path))]
