"""Reverses scikit-learn scalers and power or quantile transforms from their stored arrays alone, without importing scikit-learn."""

from __future__ import annotations

from typing import Any

import numpy as np

from tabcustody._reader import Shell

SCALERS = ("StandardScaler", "MinMaxScaler", "RobustScaler")
TRANSFORMS = ("PowerTransformer", "QuantileTransformer")
TOLERANCE = 1e-6


def scaler_kind(node: Any) -> str | None:
    """Returns the class name when `node` is the Shell of a scikit-learn scaler or transform this module can reverse."""
    if not isinstance(node, Shell) or not node.name.startswith("sklearn.preprocessing"):
        return None
    kind = node.name.rsplit(".", 1)[-1]
    return kind if kind in SCALERS + TRANSFORMS else None


def _vector(scaler: Shell, name: str, width: int, default: float) -> np.ndarray:
    value = scaler.attributes.get(name)
    return np.full(width, default) if value is None else np.asarray(value, dtype=float)


def _width(scaler: Shell) -> int | None:
    for name in ("scale_", "lambdas_"):
        value = scaler.attributes.get(name)
        if value is not None:
            return int(np.asarray(value).shape[0])
    quantiles = scaler.attributes.get("quantiles_")
    return None if quantiles is None else int(np.asarray(quantiles).shape[1])


def _yeo_johnson_inverse(column: np.ndarray, lam: float) -> np.ndarray:
    out = np.empty_like(column)
    high = column >= 0
    up, down = column[high], column[~high]
    out[high] = np.expm1(up) if abs(lam) < TOLERANCE else np.power(up * lam + 1, 1 / lam) - 1
    if abs(lam - 2) < TOLERANCE:
        out[~high] = -np.expm1(-down)
    else:
        out[~high] = 1 - np.power(-(2 - lam) * down + 1, 1 / (2 - lam))
    return out


def _box_cox_inverse(column: np.ndarray, lam: float) -> np.ndarray:
    return np.exp(column) if abs(lam) < TOLERANCE else np.power(column * lam + 1, 1 / lam)


def _power_inverse(table: np.ndarray, scaler: Shell, width: int) -> np.ndarray:
    standardiser = scaler.attributes.get("_scaler")
    values = table
    if isinstance(standardiser, Shell):
        values = table * _vector(standardiser, "scale_", width, 1.0) + _vector(
            standardiser, "mean_", width, 0.0
        )
    one_per_column = (
        _box_cox_inverse if scaler.attributes.get("method") == "box-cox" else _yeo_johnson_inverse
    )
    lambdas = np.asarray(scaler.attributes["lambdas_"], dtype=float)
    columns = [one_per_column(values[:, j], float(lambdas[j])) for j in range(width)]
    return np.asarray(np.column_stack(columns), dtype=float)


def _quantile_inverse(table: np.ndarray, scaler: Shell, width: int) -> np.ndarray:
    quantiles = np.asarray(scaler.attributes["quantiles_"], dtype=float)
    references = np.asarray(scaler.attributes["references_"], dtype=float)
    columns = [np.interp(table[:, j], references, quantiles[:, j]) for j in range(width)]
    return np.asarray(np.column_stack(columns), dtype=float)


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
    if kind == "PowerTransformer":
        return _power_inverse(table, scaler, width)
    if kind == "QuantileTransformer" and scaler.attributes.get("output_distribution") == "uniform":
        return _quantile_inverse(table, scaler, width)
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
        return bool(centred and np.allclose(columns.std(axis=0), 1, atol=TOLERANCE))
    if kind == "MinMaxScaler":
        low, high = scaler.attributes.get("feature_range", (0, 1))
        return bool(
            np.allclose(columns.min(axis=0), low, atol=TOLERANCE)
            and np.allclose(columns.max(axis=0), high, atol=TOLERANCE)
        )
    if kind == "RobustScaler":
        centred = scaler.attributes.get("center_") is None or np.allclose(
            np.median(columns, axis=0), 0, atol=TOLERANCE
        )
        if scaler.attributes.get("scale_") is None:
            return bool(centred)
        low, high = scaler.attributes.get("quantile_range", (25.0, 75.0))
        spread = np.percentile(columns, high, axis=0) - np.percentile(columns, low, axis=0)
        return bool(centred and np.allclose(spread, 1, atol=TOLERANCE))
    if kind == "PowerTransformer":
        if not isinstance(scaler.attributes.get("_scaler"), Shell):
            return False
        return bool(
            np.allclose(columns.mean(axis=0), 0, atol=TOLERANCE)
            and np.allclose(columns.std(axis=0), 1, atol=TOLERANCE)
        )
    if kind == "QuantileTransformer" and scaler.attributes.get("output_distribution") == "uniform":
        return bool(
            np.allclose(columns.min(axis=0), 0, atol=TOLERANCE)
            and np.allclose(columns.max(axis=0), 1, atol=TOLERANCE)
        )
    return False
