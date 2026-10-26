"""Saves a live fitted model with its training data lifted out, and restores it, by pickling the model through a pickler that externalises exactly what the scanner would report: tables, target-less tables, copies, column extremes and caches."""

from __future__ import annotations

import base64
import io
import json
import os
import pickle
import platform
import sys
import warnings
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from tabcustody._detect import _TABICL_CACHE, find_tables
from tabcustody._reader import read_bytes
from tabcustody._trace import merkle_root, table_leaves

MIN_ROWS = 20
_TABPFN_CACHE_PREFIX = "tabpfn.architectures."
_FORMAT = "tabcustody-seal/2"
_FORMATS = ("tabcustody-seal/1", _FORMAT)
_HKDF_INFO = b"tabcustody-seal/1"
_AUDIT_INFO = b"tabcustody-audit/1"
_COMMITMENT = "rfc9162-hmac-sha256/1"
_LIBRARIES = {
    "numpy": "numpy",
    "scikit-learn": "sklearn",
    "torch": "torch",
    "tabpfn": "tabpfn",
    "tabicl": "tabicl",
    "tabdpt": "tabdpt",
}


class CannotSeal(Exception):
    """Raised when sealing would leave training data a scan still finds, rather than write a file that only looks sealed."""


class CannotUnseal(Exception):
    """Raised when the key does not open the file, or the file was changed since it was sealed."""


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
    return _sealable(model, min_rows)[0]


def training_tables(model: Any, min_rows: int = MIN_ROWS) -> list[tuple[np.ndarray, Any]]:
    """Returns each training table of the live model with its 1-D target, or None for a target-less table; copies and caches are left out, they derive from a table."""
    tables = _sealable(model, min_rows)[1]
    targeted = {len(table) for table, target in tables if target is not None}
    return [(t, target) for t, target in tables if target is not None or len(t) not in targeted]


def _sealable(model: Any, min_rows: int) -> tuple[list[Any], list[tuple[np.ndarray, Any]]]:
    lift: list[Any] = []
    taken: set[int] = set()
    tables: list[tuple[np.ndarray, Any]] = []

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
                    tables.append((value, companions[value.shape[0]]))
                elif declared == value.shape[1]:
                    take(value)
                    tables.append((value, None))
        for child in _walk_children(node):
            if id(child) not in seen:
                stack.append(child)

    rows = {table.shape[0] for table, _ in tables}
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
    return lift, tables


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


def _derive(key: bytes, salt: bytes) -> tuple[bytes, bytes, bytes, str, bytes]:
    """Derives, from the sealing key and a salt, the encryption key, the fingerprint key, a commitment to the key, a key id, and a key for the whole-file authentication tag, each under its own label so none reveals another."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    material = HKDF(hashes.SHA256(), 136, salt, _HKDF_INFO).derive(key)
    return (
        material[:32],
        material[32:64],
        material[64:96],
        material[96:104].hex(),
        material[104:136],
    )


def audit_key(key: bytes) -> bytes:
    """Derives from the sealing key, without the file's salt, the key under which rows are committed: the same table sealed twice under one key keeps its root, and this key verifies proofs but decrypts nothing."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    return HKDF(hashes.SHA256(), 32, None, _AUDIT_INFO).derive(key)


def _versions() -> dict[str, str]:
    """Returns the versions of Python, tabcustody and the model libraries loaded now, which decide whether a model predicts the same."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        own = version("tabcustody")
    except PackageNotFoundError:
        own = "unknown"
    found = {"python": platform.python_version(), "tabcustody": own}
    for name, module in _LIBRARIES.items():
        if module in sys.modules:
            try:
                found[name] = version(name)
            except PackageNotFoundError:
                found[name] = str(getattr(sys.modules[module], "__version__", "unknown"))
    return found


def _table_fingerprints(model: Any, lifted: list[Any], key: bytes, min_rows: int) -> dict[int, Any]:
    audit = audit_key(key)
    fingerprints: dict[int, Any] = {}
    for table, target in training_tables(model, min_rows):
        slot = next(i for i, obj in enumerate(lifted) if obj is table)
        root = merkle_root(table_leaves(audit, table, target))
        fingerprints[slot] = {"scheme": _COMMITMENT, "rows": len(table), "root": root.hex()}
    return fingerprints


def _hmac(fp_key: bytes, data: bytes) -> str:
    from cryptography.hazmat.primitives import hashes, hmac

    tag = hmac.HMAC(fp_key, hashes.SHA256())
    tag.update(data)
    return tag.finalize().hex()


def _file_tag(mac_key: bytes, model_bytes: bytes, manifest: dict[str, Any]) -> str:
    """Authenticates the clear model skeleton and the manifest together, so a changed model.pkl or manifest is refused before anything is decrypted or unpickled."""
    body = {name: value for name, value in manifest.items() if name != "mac"}
    return _hmac(mac_key, model_bytes + json.dumps(body, sort_keys=True).encode())


def seal(model: Any, path: str | Path, key: bytes, min_rows: int = MIN_ROWS) -> None:
    """Writes a model file whose training data is encrypted with `key`: reloaded with the key by `unseal`, the model predicts exactly as before; without it, `tabcustody scan` finds neither a training table nor a cache. Raises CannotSeal if some training data cannot be reached."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    model_bytes, lifted = lift(model, min_rows)
    fingerprints = _table_fingerprints(model, lifted, key, min_rows)
    salt = os.urandom(16)
    enc_key, fp_key, commitment, key_id, mac_key = _derive(key, salt)
    cipher = AESGCM(enc_key)
    objects = []
    for slot, obj in enumerate(lifted):
        blob = pickle.dumps(obj, protocol=pickle.DEFAULT_PROTOCOL)
        nonce = os.urandom(12)
        ciphertext = cipher.encrypt(nonce, blob, str(slot).encode())
        objects.append(
            {
                "slot": slot,
                "key_id": key_id,
                "nonce": nonce.hex(),
                "ciphertext": base64.b64encode(ciphertext).decode(),
                "fingerprint": _hmac(fp_key, blob),
                "table_fingerprint": fingerprints.get(slot),
            }
        )
    manifest = {
        "format": _FORMAT,
        "cipher": "AES-256-GCM",
        "salt": salt.hex(),
        "commitment": commitment.hex(),
        "key_id": key_id,
        "versions": _versions(),
        "objects": objects,
    }
    manifest["mac"] = _file_tag(mac_key, model_bytes, manifest)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("model.pkl", model_bytes)
        archive.writestr("manifest.json", json.dumps(manifest))


def unseal(path: str | Path, key: bytes) -> Any:
    """Rebuilds the model `seal` wrote, with `key`; raises CannotUnseal if the key is wrong or the file was changed."""
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.constant_time import bytes_eq

    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        model_bytes = archive.read("model.pkl")
    if manifest.get("format") not in _FORMATS:
        raise CannotUnseal(f"unknown sealed format {manifest.get('format')!r}")
    enc_key, _, commitment, _, mac_key = _derive(key, bytes.fromhex(manifest["salt"]))
    if not bytes_eq(commitment, bytes.fromhex(manifest["commitment"])):
        raise CannotUnseal("the key does not open this file")
    expected = _file_tag(mac_key, model_bytes, manifest)
    if "mac" not in manifest or not bytes_eq(
        bytes.fromhex(expected), bytes.fromhex(manifest["mac"])
    ):
        raise CannotUnseal("the file was changed since it was sealed")
    cipher = AESGCM(enc_key)
    lifted: list[Any] = [None] * len(manifest["objects"])
    for entry in manifest["objects"]:
        ciphertext = base64.b64decode(entry["ciphertext"])
        try:
            blob = cipher.decrypt(
                bytes.fromhex(entry["nonce"]), ciphertext, str(entry["slot"]).encode()
            )
        except InvalidTag as error:
            raise CannotUnseal("the file was changed since it was sealed") from error
        lifted[int(entry["slot"])] = pickle.loads(blob)
    model = restore(model_bytes, lifted)
    _warn_on_other_versions(manifest.get("versions", {}))
    return model


def _warn_on_other_versions(sealed: dict[str, str]) -> None:
    now = _versions()
    changed = [
        f"{name} {sealed[name]} -> {now[name]}"
        for name in sealed
        if name in now and name not in ("python", "tabcustody") and now[name] != sealed[name]
    ]
    if changed:
        warnings.warn(
            "this model was sealed with other library versions and may predict differently: "
            + ", ".join(changed),
            stacklevel=3,
        )
