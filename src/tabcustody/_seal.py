"""Saves a live fitted model with its training data lifted out, and restores it, by pickling the model through a pickler that externalises exactly what the scanner would report: tables, target-less tables, copies, column extremes and caches."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import pickle
import platform
import sys
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tabcustody._detect import (
    _TABICL_CACHE,
    _TABPFN_ENTRY,
    _TABPFN_QUANTIZED_ENTRY,
    find_tables,
)
from tabcustody._reader import read_bytes
from tabcustody._trace import identifier, inclusion_path, merkle_root, table_leaves

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


class CannotTrace(Exception):
    """Raised when a sealed file carries no row commitment (format 1), or has no such table or row."""


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


def _is_tabpfn_cache(entry: Any) -> bool:
    """Tells a TabPFN cache from the network, whose classes share its module prefix, by what the scanner reads: per-layer key-value entries and the training shape."""
    if not _class_name(entry).startswith(_TABPFN_CACHE_PREFIX):
        return False
    layers = getattr(entry, "kv", None)
    if not isinstance(layers, dict) or not layers or getattr(entry, "train_shape", None) is None:
        return False
    return {_class_name(layer) for layer in layers.values()} <= {
        _TABPFN_ENTRY,
        _TABPFN_QUANTIZED_ENTRY,
    }


def _is_cache(value: Any) -> bool:
    """Tells whether `value` is a cache container: a dict of TabICL caches, or a list of TabPFN caches."""
    entries = list(value.values()) if isinstance(value, dict) else value
    if not isinstance(value, dict | list) or not entries:
        return False
    if {_class_name(entry) for entry in entries} == {_TABICL_CACHE}:
        return True
    return isinstance(value, list) and all(_is_tabpfn_cache(entry) for entry in entries)


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
        target_slot = next((i for i, obj in enumerate(lifted) if obj is target), None)
        root = merkle_root(table_leaves(audit, table, target))
        fingerprints[slot] = {
            "scheme": _COMMITMENT,
            "rows": len(table),
            "target_slot": target_slot,
            "root": root.hex(),
        }
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


def _predict(model: Any, inputs: np.ndarray) -> np.ndarray:
    method = getattr(model, "predict_proba", None) or getattr(model, "predict", None)
    if method is None:
        raise CannotSeal("probes need a model with predict_proba or predict")
    return np.asarray(method(inputs))


def _probe_inputs(model: Any, probes: int | np.ndarray, min_rows: int) -> np.ndarray:
    """Returns the probe inputs: the given array, or that many midpoints between two rows of the first training table, drawn with a fixed seed."""
    if not isinstance(probes, int):
        return np.asarray(probes)
    tables = training_tables(model, min_rows)
    if not tables:
        raise CannotSeal("no training table to draw probes from; pass the probe inputs")
    table = tables[0][0]
    width = getattr(model, "n_features_in_", table.shape[1])
    if table.shape[1] != width:
        raise CannotSeal(
            f"the table is {table.shape[1]} columns wide, the model takes {width}; "
            "pass the probe inputs"
        )
    rng = np.random.default_rng(0)
    first, second = rng.integers(0, len(table), probes), rng.integers(0, len(table), probes)
    return np.asarray((table[first] + table[second]) / 2, dtype=float)


def seal(
    model: Any,
    path: str | Path,
    key: bytes,
    min_rows: int = MIN_ROWS,
    probes: int | np.ndarray = 0,
) -> None:
    """Writes a model file whose training data is encrypted with `key`: reloaded with the key by `unseal`, the model predicts exactly as before; without it, `tabcustody scan` finds neither a training table nor a cache; with `probes`, a count or an array of inputs, the model is run on them and they are sealed with its predictions, for `verify`. Raises CannotSeal if some training data cannot be reached."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    model_bytes, lifted = lift(model, min_rows)
    fingerprints = _table_fingerprints(model, lifted, key, min_rows)
    checks = None
    if not isinstance(probes, int) or probes > 0:
        inputs = _probe_inputs(model, probes, min_rows)
        checks = {"inputs": inputs, "outputs": _predict(model, inputs)}
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
    if checks is not None:
        nonce = os.urandom(12)
        blob = pickle.dumps(checks, protocol=pickle.DEFAULT_PROTOCOL)
        manifest["probes"] = {
            "count": len(checks["inputs"]),
            "nonce": nonce.hex(),
            "ciphertext": base64.b64encode(cipher.encrypt(nonce, blob, b"probes")).decode(),
        }
    manifest["mac"] = _file_tag(mac_key, model_bytes, manifest)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("model.pkl", model_bytes)
        archive.writestr("manifest.json", json.dumps(manifest))


def unseal(path: str | Path, key: bytes) -> Any:
    """Rebuilds the model `seal` wrote, with `key`; raises CannotUnseal if the key is wrong or the file was changed."""
    manifest, model_bytes, lifted = _open(path, key)
    model = restore(model_bytes, lifted)
    _warn_on_other_versions(manifest.get("versions", {}))
    return model


@dataclass
class Verification:
    """What `verify` measured: whether the probes came back bit for bit, the largest deviation, and which library versions differ from the sealing ones."""

    probes: int
    identical: bool
    largest_deviation: float
    tolerance: float
    versions_changed: list[str] = field(default_factory=list)

    @property
    def holds(self) -> bool:
        return self.identical or self.largest_deviation <= self.tolerance

    def to_text(self) -> str:
        if self.identical:
            line = f"predicts identically, to the bit, on {self.probes} probes"
        else:
            verdict = "within" if self.holds else "beyond"
            line = (
                f"largest deviation {self.largest_deviation:.3g} on {self.probes} probes, "
                f"{verdict} the tolerance of {self.tolerance:g}"
            )
        if self.versions_changed:
            line += "\nversions changed since sealing: " + ", ".join(self.versions_changed)
        return line


def _deviation(sealed: np.ndarray, now: np.ndarray) -> float:
    if sealed.shape != now.shape:
        return float("inf")
    if sealed.dtype.kind in "biuf" and now.dtype.kind in "biuf":
        return float(np.max(np.abs(sealed.astype(float) - now.astype(float)), initial=0.0))
    return 0.0 if np.array_equal(sealed, now) else float("inf")


def verify(path: str | Path, key: bytes, tolerance: float = 0.0) -> Verification:
    """Rebuilds the sealed model here, runs it on the probes sealed with it, and reports how far its predictions moved; raises CannotTrace if the file was sealed without probes."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    manifest, model_bytes, lifted = _open(path, key)
    if "probes" not in manifest:
        raise CannotTrace("this file was sealed without probes; seal it again with probes")
    enc_key = _derive(key, bytes.fromhex(manifest["salt"]))[0]
    sealed = manifest["probes"]
    blob = AESGCM(enc_key).decrypt(
        bytes.fromhex(sealed["nonce"]), base64.b64decode(sealed["ciphertext"]), b"probes"
    )
    checks = pickle.loads(blob)
    model = restore(model_bytes, lifted)
    now = _predict(model, checks["inputs"])
    expected = np.asarray(checks["outputs"])
    return Verification(
        probes=len(checks["inputs"]),
        identical=expected.shape == now.shape
        and expected.dtype == now.dtype
        and expected.tobytes() == now.tobytes(),
        largest_deviation=_deviation(expected, now),
        tolerance=tolerance,
        versions_changed=_changed_versions(manifest.get("versions", {})),
    )


def _open(
    path: str | Path, key: bytes, slots: set[int] | None = None
) -> tuple[dict[str, Any], bytes, list[Any]]:
    """Checks the key and the whole-file tag, then decrypts the lifted objects, or only those in `slots`."""
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
        if slots is not None and int(entry["slot"]) not in slots:
            continue
        ciphertext = base64.b64decode(entry["ciphertext"])
        try:
            blob = cipher.decrypt(
                bytes.fromhex(entry["nonce"]), ciphertext, str(entry["slot"]).encode()
            )
        except InvalidTag as error:
            raise CannotUnseal("the file was changed since it was sealed") from error
        lifted[int(entry["slot"])] = pickle.loads(blob)
    return manifest, model_bytes, lifted


def _committed(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = [entry for entry in manifest["objects"] if entry.get("table_fingerprint")]
    if not entries:
        raise CannotTrace("this file carries no row commitment; it was sealed before format 2")
    return sorted(entries, key=lambda entry: int(entry["slot"]))


def _identity(manifest: dict[str, Any], model_bytes: bytes) -> tuple[bytes, list[bytes]]:
    roots = [bytes.fromhex(e["table_fingerprint"]["root"]) for e in _committed(manifest)]
    return hashlib.sha256(model_bytes).digest(), roots


def file_id(path: str | Path) -> str:
    """Returns the identifier to log beside each prediction: readable without the key, it changes with the model or any of its rows, and stays the same when the same model is sealed again under the same key."""
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        model_bytes = archive.read("model.pkl")
    return identifier(*_identity(manifest, model_bytes))


def _exact(values: Any) -> dict[str, str]:
    values = np.ascontiguousarray(values)
    values = values.astype(values.dtype.newbyteorder("<"))
    return {"dtype": values.dtype.str, "hex": values.tobytes().hex()}


def prove(path: str | Path, key: bytes, position: int, table: int = 0) -> dict[str, Any]:
    """Returns the proof that the row at `position` of the file's `table`-th training table was in it: the row and its target, and nothing of the other rows. Raises CannotTrace if there is no such table or row."""
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    entries = _committed(manifest)
    if not 0 <= table < len(entries):
        raise CannotTrace(f"this file has {len(entries)} committed table(s), not a table {table}")
    commitment = entries[table]["table_fingerprint"]
    slot, target_slot = int(entries[table]["slot"]), commitment["target_slot"]
    wanted = {slot} if target_slot is None else {slot, int(target_slot)}
    manifest, model_bytes, lifted = _open(path, key, wanted)
    rows = lifted[slot]
    target = None if target_slot is None else lifted[int(target_slot)]
    if not 0 <= position < len(rows):
        raise CannotTrace(f"table {table} has {len(rows)} rows, not a row {position}")
    leaves = table_leaves(audit_key(key), rows, target)
    if merkle_root(leaves).hex() != commitment["root"]:
        raise CannotTrace("the table does not match its commitment")
    skeleton, roots = _identity(manifest, model_bytes)
    return {
        "format": "tabcustody-proof/1",
        "scheme": commitment["scheme"],
        "skeleton_sha256": skeleton.hex(),
        "roots": [root.hex() for root in roots],
        "table": table,
        "rows": len(rows),
        "position": position,
        "row": _exact(rows[position]),
        "target": None if target is None else _exact(target[position]),
        "row_values": rows[position].tolist(),
        "target_value": None if target is None else target[position].item(),
        "path": [sibling.hex() for sibling in inclusion_path(leaves, position)],
    }


def _changed_versions(sealed: dict[str, str]) -> list[str]:
    now = _versions()
    return [
        f"{name} {sealed[name]} -> {now[name]}"
        for name in sealed
        if name in now and name not in ("python", "tabcustody") and now[name] != sealed[name]
    ]


def _warn_on_other_versions(sealed: dict[str, str]) -> None:
    changed = _changed_versions(sealed)
    if changed:
        warnings.warn(
            "this model was sealed with other library versions and may predict differently: "
            + ", ".join(changed),
            stacklevel=3,
        )
