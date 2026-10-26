"""Removes a person's rows from a fitted in-context model and refits a clone, so the result predicts exactly as a model fitted without them, and refuses whenever that cannot be shown."""

from __future__ import annotations

import hashlib
import hmac
import json
import pickle
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tabcustody._absence import TOLERANCE, _sealed, _tolerance, check_absent
from tabcustody._detect import _walk, as_table, as_vector, find_tables
from tabcustody._reader import read_bytes
from tabcustody._reverse import reverse
from tabcustody._seal import (
    CannotTrace,
    Keys,
    _versions,
    audit_key,
    file_id,
    opens,
    seal,
    unseal,
)

PROBES = 32


class CannotErase(Exception):
    """Raised when the rows are not in the training data, or the training data cannot be shown to be the one the model was fitted on; the model is left as it was."""


@dataclass(frozen=True)
class Erasure:
    """The refitted model, how many training rows were removed and kept, and where the training data came from: recovered from the model, or given."""

    model: Any
    removed: int
    kept: int
    source: str


def _clone(model: Any) -> Any:
    import importlib

    try:
        clone = importlib.import_module("sklearn.base").clone
    except ImportError:
        if not hasattr(model, "get_params"):
            raise CannotErase("the model has no get_params, so it cannot be cloned") from None
        return type(model)(**model.get_params())
    return clone(model)


def _fit(model: Any, X: np.ndarray, y: np.ndarray) -> Any:
    fresh = _clone(model)
    fitted = fresh.fit(X, y)
    return fresh if fitted is None else fitted


def _outputs(model: Any, inputs: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    """Returns what must not change: the classes, and the probabilities, or the predictions of a model without them."""
    classes = getattr(model, "classes_", None)
    method = getattr(model, "predict_proba", None) or model.predict
    return (None if classes is None else np.asarray(classes)), np.asarray(method(inputs))


def _same(
    first: tuple[np.ndarray | None, np.ndarray], second: tuple[np.ndarray | None, np.ndarray]
) -> bool:
    (classes, out), (other_classes, other_out) = first, second
    if classes is None or other_classes is None:
        if classes is not other_classes:
            return False
    elif not np.array_equal(classes, other_classes):
        return False
    return (
        out.shape == other_out.shape
        and out.dtype == other_out.dtype
        and out.tobytes() == other_out.tobytes()
    )


def _probes(X: np.ndarray) -> np.ndarray:
    """Draws midpoints between two training rows with a fixed seed: inputs that exercise the model without being its own rows."""
    rng = np.random.default_rng(0)
    first, second = rng.integers(0, len(X), PROBES), rng.integers(0, len(X), PROBES)
    return np.asarray((X[first] + X[second]) / 2, dtype=float)


def _candidates(model: Any, width: int, min_rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Returns each (X, y) the model may have been fitted on, read from its own training tables after reversing their scaler, with the target mapped through classes_ first; duplicates removed."""
    tree = read_bytes(pickle.dumps(model))
    nodes = {path.removeprefix("."): node for path, node in _walk(tree)}
    classes = getattr(model, "classes_", None)
    found: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[bytes] = set()
    for finding in find_tables(tree, min_rows=min_rows):
        if finding.cache or finding.extremes or finding.copy or finding.companion is None:
            continue
        table, target = as_table(nodes.get(finding.path)), as_vector(nodes.get(finding.companion))
        if table is None or target is None:
            continue
        if finding.reversible_by:
            table = reverse(table, nodes[finding.reversible_by])
        if table.shape[1] != width:
            continue
        targets = [target]
        if (
            classes is not None
            and target.dtype.kind in "iu"
            and 0 <= target.min()
            and target.max() < len(classes)
        ):
            targets.insert(0, np.asarray(classes)[target])
        for y in targets:
            key = np.ascontiguousarray(table).tobytes() + np.ascontiguousarray(y).tobytes()
            if key not in seen:
                seen.add(key)
                found.append((np.asarray(table), y))
    return found


def _matches(X: np.ndarray, rows: np.ndarray) -> list[np.ndarray]:
    """Returns, for each given row, the positions of the training rows equal to it, column by column."""
    tolerance = _tolerance(X) if X.dtype.kind == "f" else TOLERANCE
    data = np.asarray(X, dtype=float)
    return [
        np.flatnonzero(np.isclose(data, row, rtol=tolerance, atol=1e-12).all(axis=1))
        for row in rows
    ]


def erase(
    model: Any,
    rows: np.ndarray,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    min_rows: int = 20,
) -> Erasure:
    """Returns a clone of the model refitted without every training row equal to one of `rows`, which predicts exactly as a model fitted from scratch without them. The training data is the given X and y, or else recovered from the model's own tables; either way it must first refit to a model that predicts bit for bit as this one, or CannotErase is raised. The model passed in is never changed."""
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    width = rows.shape[1]
    declared = getattr(model, "n_features_in_", width)
    if declared != width:
        raise CannotErase(f"the rows have {width} columns, the model takes {declared}")
    if X is not None and y is not None:
        sources = [(np.asarray(X), np.asarray(y))]
        origin = "given"
    elif X is None and y is None:
        sources = _candidates(model, width, min_rows)
        origin = "recovered"
        if not sources:
            raise CannotErase("no training table with its target in the model; pass X and y")
    else:
        raise CannotErase("pass both X and y, or neither")

    probes = _probes(np.asarray(sources[0][0], dtype=float))
    expected = _outputs(model, probes)
    data = next(
        (pair for pair in sources if _same(_outputs(_fit(model, *pair), probes), expected)), None
    )
    if data is None:
        reason = (
            "X and y do not refit to this model"
            if origin == "given"
            else "the data kept in the model does not refit to it; pass X and y"
        )
        raise CannotErase(reason)

    X_data, y_data = data
    hits = _matches(X_data, rows)
    missing = [i for i, found in enumerate(hits) if len(found) == 0]
    if missing:
        raise CannotErase(f"row(s) {missing} of the given rows are not in the training data")
    keep = np.ones(len(X_data), dtype=bool)
    for found in hits:
        keep[found] = False
    refitted = _fit(model, X_data[keep], y_data[keep])
    return Erasure(refitted, int((~keep).sum()), int(keep.sum()), origin)


_RECEIPT = "tabcustody-erasure/1"


def _receipt_tag(audit: bytes, receipt: dict[str, Any]) -> str:
    body = {name: value for name, value in receipt.items() if name != "mac"}
    return hmac.new(audit, json.dumps(body, sort_keys=True).encode(), hashlib.sha256).hexdigest()


def verify_receipt(receipt: dict[str, Any], audit_key: bytes) -> bool:
    """Tells whether the receipt is the one tabcustody wrote, unchanged, with the audit key of the new version; it needs neither the sealing key nor the files."""
    tag = receipt.get("mac")
    return isinstance(tag, str) and hmac.compare_digest(_receipt_tag(audit_key, receipt), tag)


def _manifest(path: str | Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        manifest: dict[str, Any] = json.loads(archive.read("manifest.json"))
    return manifest


def _load(path: str | Path, keys: Keys | None) -> tuple[Any, dict[str, Any]]:
    """Opens the old version: a sealed file with its keys, or a clear pickle, which runs it as `seal` does; returns the model and what the receipt says of that version."""
    if _sealed(path):
        if keys is None:
            raise CannotErase("the old version is sealed; pass its key")
        model = unseal(path, keys)
        manifest = _manifest(path)
        try:
            identifier = file_id(path)
        except CannotTrace:
            identifier = None
        return model, {"sealed": True, "id": identifier, "key_id": manifest.get("key_id")}
    with open(path, "rb") as file:
        model = pickle.load(file)
    return model, {"sealed": False, "id": None, "key_id": None}


def erase_file(
    old: str | Path,
    new: str | Path,
    rows: np.ndarray,
    new_key: bytes,
    old_keys: Keys | None = None,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    receipt: str | Path | None = None,
    min_rows: int = 20,
) -> dict[str, Any]:
    """Writes a new version of the model without the given rows, sealed under `new_key`, checks that the rows are absent from it, and writes a receipt beside it that holds nothing of the rows; returns the receipt. Raises CannotErase, and writes nothing, if the rows cannot be removed or `new_key` still opens the old version."""
    model, before = _load(old, old_keys)
    if before["sealed"] and opens(old, new_key):
        raise CannotErase("the new key opens the old version; seal the new one under a fresh key")
    erasure = erase(model, rows, X, y, min_rows=min_rows)
    old_probes = 0
    if before["sealed"]:
        manifest = _manifest(old)
        old_probes = int(manifest.get("probes", {}).get("count", 0))
    seal(erasure.model, new, new_key, min_rows=min_rows, probes=old_probes)
    report = check_absent(new, rows, min_rows=min_rows, keys=new_key)
    if report.present or report.error is not None:
        Path(new).unlink()
        raise CannotErase("the rows are still found in the new version; nothing was written")
    caches = [place for place in report.unverifiable if "cache" in place]
    written = {
        "format": _RECEIPT,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "tabcustody": _versions()["tabcustody"],
        "before": before,
        "after": {
            "id": file_id(new),
            "key_id": _manifest(new)["key_id"],
        },
        "rows_given": len(np.atleast_2d(rows)),
        "removed": erasure.removed,
        "kept": erasure.kept,
        "training_data": erasure.source,
        "check": {
            "absent_from_tables": not report.uncertain,
            "verified": report.verified,
            "not_checked": list(report.unverifiable),
            "caches": "rebuilt without the rows by refitting" if caches else None,
        },
        "old_version": (
            "destroy the key whose key_id is above to make every copy of the old version "
            "unreadable; check it with tabcustody check --revoked"
            if before["sealed"]
            else "the old version was not sealed: its copies keep the rows in clear and no key "
            "can make them unreadable; delete them"
        ),
    }
    written["mac"] = _receipt_tag(audit_key(new_key), written)
    target = Path(receipt) if receipt is not None else Path(f"{new}.receipt.json")
    target.write_text(json.dumps(written, indent=2))
    return written
