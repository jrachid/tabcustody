"""Reverses scikit-learn scalers from their stored arrays alone, without importing scikit-learn."""

from __future__ import annotations

from typing import Any

import numpy as np

from tabcustody._reader import Shell

SCALERS = ("StandardScaler", "MinMaxScaler", "RobustScaler")
TOLERANCE = 1e-6


def scaler_kind(node: Any) -> str | None:
    """Returns the scaler class name when `node` is the Shell of a scikit-learn scaler this module can reverse."""
    if not isinstance(node, Shell) or not node.name.startswith("sklearn.preprocessing"):
        return None
    kind = node.name.rsplit(".", 1)[-1]
    return kind if kind in SCALERS else None


def _vector(scaler: Shell, name: str, width: int, default: float) -> np.ndarray:
    value = scaler.attributes.get(name)
    return np.full(width, default) if value is None else np.asarray(value, dtype=float)


def _width(scaler: Shell) -> int | None:
    scale = scaler.attributes.get("scale_")
    return None if scale is None else int(np.asarray(scale).shape[0])


def reverse(table: np.ndarray, scaler: Shell) -> np.ndarray:
    """Returns the rows `scaler` was given before it produced `table`; raises ValueError for a scaler it cannot reverse."""
    kind, width = scaler_kind(scaler), table.shape[1]
    scale = _vector(scaler, "scale_", width, 1.0)
    if kind == "StandardScaler":
        return np.asarray(table * scale + _vector(scaler, "mean_", width, 0.0), dtype=float)
    if kind == "MinMaxScaler":
        return np.asarray((table - _vector(scaler, "min_", width, 0.0)) / scale, dtype=float)
    if kind == "RobustScaler":
        return np.asarray(table * scale + _vector(scaler, "center_", width, 0.0), dtype=float)
    raise ValueError(f"no reversal for {scaler.name}")


def produced(table: np.ndarray, scaler: Shell) -> bool:
    """Tells whether `table` carries the fingerprint `scaler` leaves on the very rows it was fitted on."""
    kind = scaler_kind(scaler)
    if kind is None or _width(scaler) != table.shape[1] or table.shape[0] < 2:
        return False
    varying = np.ptp(table, axis=0) > TOLERANCE
    if not varying.any():
        return False
    columns = table[:, varying]
    if kind == "StandardScaler":
        centred = scaler.attributes.get("mean_") is None or np.allclose(
            columns.mean(axis=0), 0, atol=TOLERANCE
        )
        return centred and np.allclose(columns.std(axis=0), 1, atol=TOLERANCE)
    if kind == "MinMaxScaler":
        low, high = scaler.attributes.get("feature_range", (0, 1))
        return np.allclose(columns.min(axis=0), low, atol=TOLERANCE) and np.allclose(
            columns.max(axis=0), high, atol=TOLERANCE
        )
    centred = scaler.attributes.get("center_") is None or np.allclose(
        np.median(columns, axis=0), 0, atol=TOLERANCE
    )
    if scaler.attributes.get("scale_") is None:
        return centred
    low, high = scaler.attributes.get("quantile_range", (25.0, 75.0))
    spread = np.percentile(columns, high, axis=0) - np.percentile(columns, low, axis=0)
    return centred and np.allclose(spread, 1, atol=TOLERANCE)
