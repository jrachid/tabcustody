"""Finds training tables in an object tree built by the reader, without being given the training data."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from tabcustody._reader import Shell
from tabcustody._reverse import produced, scaler_kind

TRAINING_NAMES = frozenset(
    {"X_train", "X_", "_fit_X", "X_fit_", "support_vectors_", "X_transformed_"}
)


@dataclass(frozen=True)
class Finding:
    """A 2-D numeric array judged to hold training rows, or a cache built from them; `companion` is the 1-D array with one entry per row, if any.

    `shape` is (rows,) for a cache whose columns the file does not record."""

    path: str
    shape: tuple[int, ...]
    companion: str | None
    evidence: tuple[str, ...]
    reversible_by: str | None = None
    owner: str | None = None
    hint: str | None = None
    cache: bool = False
    extremes: bool = False
    copy: bool = False
    columns: tuple[str, ...] | None = None

    @property
    def kind(self) -> str:
        if self.extremes:
            return "column extremes"
        if self.cache:
            return "key-value cache"
        return "copy" if self.copy else "training table"


def _is_numeric(value: Any, ndim: int) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == ndim and value.dtype.kind in "iuf"


_FRAMES = frozenset({"pandas.DataFrame", "pandas.core.frame.DataFrame"})
_SERIES = frozenset({"pandas.Series", "pandas.core.series.Series"})


def _manager(node: Shell) -> Any:
    return node.attributes.get("_mgr", node.attributes.get("_data"))


def _placement(locations: Any, count: int) -> list[int] | None:
    if isinstance(locations, slice):
        columns = list(range(locations.start or 0, locations.stop, locations.step or 1))
    elif _is_numeric(locations, 1):
        columns = [int(column) for column in locations]
    else:
        return None
    return columns if len(columns) == count else None


def _blocks(manager: Any) -> list[tuple[np.ndarray, Any]]:
    """Pairs each block of a pickled pandas BlockManager with the columns it holds, for pandas 1 to 3 layouts."""
    blocks: dict[int, tuple[np.ndarray, Any]] = {}
    for _, node in _walk(manager):
        if isinstance(node, Shell) and node.name.endswith("_unpickle_block") and node.args:
            values, locations = node.args[0], node.args[1] if len(node.args) > 1 else None
        elif isinstance(node, dict) and "values" in node and "mgr_locs" in node:
            values, locations = node["values"], node["mgr_locs"]
        else:
            continue
        if isinstance(values, np.ndarray):
            blocks[id(values)] = (values, locations)
    return list(blocks.values())


def as_table(node: Any) -> np.ndarray | None:
    """Returns `node` as a 2-D numeric array: a NumPy array as is, a pickled pandas DataFrame rebuilt from its numeric blocks."""
    if _is_numeric(node, 2):
        return node  # type: ignore[no-any-return]
    if not (isinstance(node, Shell) and node.name in _FRAMES):
        return None
    columns: dict[int, np.ndarray] = {}
    for values, locations in _blocks(_manager(node)):
        if not _is_numeric(values, 2):
            continue
        placement = _placement(locations, values.shape[0])
        if placement is None:
            return None
        columns.update(zip(placement, values, strict=True))
    if not columns or len({column.shape[0] for column in columns.values()}) != 1:
        return None
    return np.column_stack([columns[index] for index in sorted(columns)])


def as_vector(node: Any) -> np.ndarray | None:
    """Returns `node` as a 1-D numeric array: a NumPy array as is, a pickled pandas Series by its values."""
    if _is_numeric(node, 1):
        return node  # type: ignore[no-any-return]
    if not (isinstance(node, Shell) and node.name in _SERIES):
        return None
    return next((values for _, values in _walk(_manager(node)) if _is_numeric(values, 1)), None)


_TABICL_CACHE = "tabicl._model.kv_cache.TabICLCache"
_TABPFN_ENTRY = "tabpfn.architectures.kv_cache.KVCacheEntry"
_TABPFN_QUANTIZED_ENTRY = "tabpfn.architectures.kv_cache.QuantizedKVCacheEntry"


def _tabicl_cache(node: Any) -> tuple[int, int] | None:
    """Returns the rows and columns a dict of TabICL caches was built from, as their train_shape records them."""
    if not isinstance(node, dict) or not node:
        return None
    shapes = set()
    for cache in node.values():
        if not (isinstance(cache, Shell) and cache.name == _TABICL_CACHE):
            return None
        shape = cache.attributes.get("train_shape")
        if cache.attributes.get("icl_cache") is None or not isinstance(shape, tuple | list):
            return None
        shapes.add((int(shape[-2]), int(shape[-1])))
    return shapes.pop() if len(shapes) == 1 else None


def _tabpfn_cache(node: Any) -> tuple[int, set[str]] | None:
    """Returns the rows a list of TabPFN caches was built from, as their train_shape records them, and the classes of their entries."""
    if not isinstance(node, list) or not node:
        return None
    rows, entries = set(), set()
    for cache in node:
        if not (isinstance(cache, Shell) and cache.name.startswith("tabpfn.architectures.")):
            return None
        layers, shape = cache.attributes.get("kv"), cache.attributes.get("train_shape")
        if not isinstance(layers, dict) or not layers or not isinstance(shape, tuple | list):
            return None
        names = {entry.name if isinstance(entry, Shell) else "" for entry in layers.values()}
        if not names <= {_TABPFN_ENTRY, _TABPFN_QUANTIZED_ENTRY}:
            return None
        rows.add(int(shape[-1]))
        entries |= names
    return (rows.pop(), entries) if len(rows) == 1 else None


def _tabpfn_columns(executor: Any) -> int | None:
    """Returns how many columns the TabPFN model was fitted on, as its ensemble preprocessor's feature schema lists them."""
    preprocessor = (
        executor.attributes.get("ensemble_preprocessor") if isinstance(executor, Shell) else None
    )
    schema = (
        preprocessor.attributes.get("feature_schema") if isinstance(preprocessor, Shell) else None
    )
    features = schema.attributes.get("features") if isinstance(schema, Shell) else None
    return len(features) if isinstance(features, list) else None


def _cache(node: Any, child: Any) -> tuple[tuple[int, ...], tuple[str, ...]] | None:
    """Returns the shape of the training rows `child`, an attribute of `node`, caches, and why it is a cache."""
    tabicl = _tabicl_cache(child)
    if tabicl is not None:
        return tabicl, ("TabICL's in-context cache keeps one key and one value per training row",)
    tabpfn = _tabpfn_cache(child)
    if tabpfn is None:
        return None
    rows, entries = tabpfn
    columns = _tabpfn_columns(node)
    evidence = [
        "TabPFN's in-context cache keeps one key and one value per training row, in every layer"
    ]
    if _TABPFN_QUANTIZED_ENTRY in entries:
        evidence.append("its keys and values are quantized to 8 bits, still one per training row")
    return ((rows,) if columns is None else (rows, columns)), tuple(evidence)


def _extremes(node: Any) -> tuple[int, str] | None:
    """Returns how many columns the X_min_ and X_max_ of `node` cover, and the attribute holding the scaler that brings them back to the columns' scale."""
    if not isinstance(node, Shell):
        return None
    low, high = node.attributes.get("X_min_"), node.attributes.get("X_max_")
    if not (isinstance(low, np.ndarray) and isinstance(high, np.ndarray)):
        return None
    if not (_is_numeric(low, 2) and low.shape == high.shape and low.shape[0] == 1):
        return None
    width = low.shape[1]
    for name, scaler in node.attributes.items():
        if not isinstance(scaler, Shell):
            continue
        parts = [scaler.attributes.get(part) for part in ("mean_", "scale_")]
        if all(isinstance(part, np.ndarray) and part.shape == (width,) for part in parts):
            return width, name
    return None


def _is_lookup_grid(array: np.ndarray) -> bool:
    """Tells a lookup table, such as QuantileTransformer's quantiles_, where every column is sorted, from rows of data."""
    return array.shape[0] > 1 and bool(np.all(np.diff(array, axis=0) >= 0))


def _declared_width(node: Any) -> int | None:
    """Returns the n_features_in_ a scikit-learn style estimator declares, the number of columns it was fitted on."""
    width = node.attributes.get("n_features_in_") if isinstance(node, Shell) else None
    return int(width) if isinstance(width, int | np.integer) else None


def _derived(array: np.ndarray, declared: int | None) -> bool:
    """Tells an array computed from the rows, such as a kernel's Cholesky factor or neighbour distances, from the rows themselves."""
    if declared is not None and declared != array.shape[1]:
        return True
    if array.shape[0] != array.shape[1]:
        return False
    triangular = not np.triu(array, 1).any() or not np.tril(array, -1).any()
    return triangular or bool(np.allclose(array, array.T))


def _named_children(node: Any) -> list[tuple[str, Any]]:
    if isinstance(node, Shell):
        return [(f".{key}", value) for key, value in node.attributes.items()] + [
            (f"[{key!r}]", value) for key, value in node.entries.items()
        ]
    if isinstance(node, dict):
        return [(f"[{key!r}]", value) for key, value in node.items()]
    return []


def _positional_children(node: Any) -> list[tuple[str, Any]]:
    if isinstance(node, Shell):
        children = [(f"[{index}]", value) for index, value in enumerate(node.items)]
        children += [(f"<arg{index}>", value) for index, value in enumerate(node.args)]
        if node.state is not None:
            children.append(("<state>", node.state))
        return children
    if isinstance(node, list | tuple | set | frozenset):
        return [(f"[{index}]", value) for index, value in enumerate(node)]
    if isinstance(node, np.ndarray) and node.dtype.kind == "O":
        return [(f"[{index}]", value) for index, value in enumerate(node.ravel())]
    return []


def _walk(root: Any) -> Iterator[tuple[str, Any]]:
    seen: set[int] = set()
    stack: list[tuple[str, Any]] = [("", root)]
    while stack:
        path, node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield path, node
        children = _named_children(node) + _positional_children(node)
        stack.extend((path + suffix, child) for suffix, child in reversed(children))


_PIPELINE_STEP = re.compile(r"^(?P<pipeline>.*)\[(?P<index>\d+)\]\[1\]$")


def _parent(path: str) -> str:
    return path[: max(path.rfind("."), path.rfind("["), 0)]


def _pipeline_step_before(scaler_path: str, table_path: str) -> bool:
    step = _PIPELINE_STEP.match(_parent(table_path))
    if step is None:
        return False
    return scaler_path == f"{step['pipeline']}[{int(step['index']) - 1}][1]"


# A power or quantile transform is credited only as a pipeline's last step before the table: beside another
# table it cannot be told from the scaler that the transform's own output was then fed to.
_PIPELINE_ONLY = ("PowerTransformer", "QuantileTransformer")


def _feeds(scaler_path: str, table_path: str, kind: str | None) -> bool:
    """Tells whether the scaler sits beside the table's owner, or is the pipeline step right before it."""
    if _pipeline_step_before(scaler_path, table_path):
        return True
    if kind in _PIPELINE_ONLY:
        return False
    return _parent(scaler_path) == _parent(table_path)


def _attach_scalers(
    tables: list[Finding], arrays: dict[str, np.ndarray], scalers: list[tuple[str, Shell]]
) -> list[Finding]:
    attached = []
    for finding in tables:
        if finding.path not in arrays:
            attached.append(finding)
            continue
        feeding = [
            (path, scaler)
            for path, scaler in scalers
            if _feeds(path, finding.path, scaler_kind(scaler))
            and produced(arrays[finding.path], scaler)
        ]
        if len(feeding) != 1:
            attached.append(finding)
            continue
        path, scaler = feeding[0]
        note = f"its columns carry the fingerprint of the {scaler_kind(scaler)} at {path}, which reverses it"
        attached.append(replace(finding, reversible_by=path, evidence=(*finding.evidence, note)))
    return attached


def _column_names(node: Any) -> tuple[str, ...] | None:
    """Returns the column names a scikit-learn style estimator records in feature_names_in_, when it keeps them as text."""
    if not isinstance(node, Shell):
        return None
    names = node.attributes.get("feature_names_in_")
    if isinstance(names, np.ndarray) and names.ndim == 1 and names.dtype.kind in "UO":
        return tuple(str(name) for name in names)
    return None


def _fitted_width(raw: str, widths: dict[str, tuple[int, str]]) -> tuple[int, str, str] | None:
    """Returns the n_features_in_ of the closest estimator above the raw path `raw`, with its own raw path and class."""
    while raw:
        raw = _parent(raw)
        if raw in widths:
            return widths[raw][0], raw, widths[raw][1]
    return None


def _copies(
    tables: list[Finding], arrays: dict[str, np.ndarray], declared: dict[str, int | None]
) -> list[Finding]:
    confirmed = {
        finding.path: finding.shape[0]
        for finding in tables
        if not (finding.cache or finding.extremes or finding.copy)
    }
    found = {finding.path for finding in tables}
    copies = []
    for path, array in arrays.items():
        if path in found or _derived(array, declared.get(path)):
            continue
        source = next((other for other, rows in confirmed.items() if rows == array.shape[0]), None)
        if source is not None:
            evidence = (f"it has as many rows as the training table {source}",)
            copies.append(Finding(path, array.shape, None, evidence, copy=True))
    return copies


_REBUILD_TENSOR = "torch._utils._rebuild_tensor_v2"


def _tensor_shape(node: Any) -> tuple[int, ...] | None:
    """Returns the shape of a PyTorch tensor, read from its rebuild call's size argument without importing torch."""
    if not (isinstance(node, Shell) and node.name == _REBUILD_TENSOR):
        return None
    for arg in node.args:
        if isinstance(arg, tuple) and arg and all(isinstance(axis, int) for axis in arg):
            return arg
    return None


def _row_counts(node: Any, min_rows: int) -> set[int]:
    """Returns the training-row counts a node records: a 1-D array's length, a train_shape, an n_samples_fit_."""
    if _is_numeric(node, 1) and node.shape[0] >= min_rows:
        return {int(node.shape[0])}
    if not isinstance(node, Shell):
        return set()
    counts = set()
    shape = node.attributes.get("train_shape")
    if isinstance(shape, tuple | list):
        counts |= {int(axis) for axis in shape if isinstance(axis, int) and axis >= min_rows}
    samples = node.attributes.get("n_samples_fit_")
    if isinstance(samples, int) and samples >= min_rows:
        counts.add(samples)
    return counts


def _common_ancestor(paths: list[str]) -> str:
    """Returns the deepest path that is an ancestor of every path in `paths`, cut at a name boundary."""
    prefix = paths[0]
    for path in paths[1:]:
        while not path.startswith(prefix):
            prefix = _parent(prefix)
    return prefix


def _shape_cache(
    tensors: list[tuple[str, tuple[int, ...]]], row_counts: set[int], min_rows: int
) -> list[Finding]:
    """Flags, as a last resort when no cache class was recognised, PyTorch tensors whose one axis is a known training-row count: the shape of a key-value cache from an unknown library."""
    matching = [(path, shape) for path, shape in tensors if row_counts & set(shape)]
    if not matching:
        return []
    rows = max(count for count in row_counts if any(count in shape for _, shape in matching))
    if rows < min_rows:
        return []
    reason = (
        f"PyTorch tensors here have an axis of {rows}, the training-row count the file records, "
        "the shape of a key-value cache even though their class is not a known one"
    )
    return [
        Finding(
            _common_ancestor([path for path, _ in matching]), (rows,), None, (reason,), cache=True
        )
    ]


def _attach_column_names(tables: list[Finding], named: dict[str, tuple[str, ...]]) -> list[Finding]:
    """Names a table's columns when the estimator that holds it directly records feature_names_in_ of the same width, never from an ancestor, so a preprocessed table keeps no names."""
    attached = []
    for finding in tables:
        names = named.get(_parent(finding.path))
        if finding.cache or finding.extremes or names is None or len(names) != finding.shape[-1]:
            attached.append(finding)
            continue
        reason = f"the columns are named {', '.join(names)}"
        attached.append(replace(finding, columns=names, evidence=(*finding.evidence, reason)))
    return attached


def find_tables(tree: Any, min_rows: int = 20, prefix: str = "") -> list[Finding]:
    """Returns the training tables in `tree`: 2-D arrays beside a 1-D array of the same length or as wide as their estimator's n_features_in_, caches, column extremes, copies."""

    def _clean(path: str) -> str:
        return prefix + path.removeprefix(".")

    tables: list[Finding] = []
    arrays: dict[str, np.ndarray] = {}
    declared: dict[str, int | None] = {}
    widths: dict[str, tuple[int, str]] = {}
    raw_paths: dict[str, str] = {}
    named: dict[str, tuple[str, ...]] = {}
    scalers: list[tuple[str, Shell]] = []
    tensors: list[tuple[str, tuple[int, ...]]] = []
    row_counts: set[int] = set()
    for path, node in _walk(tree):
        table = as_table(node)
        if table is not None:
            arrays[_clean(path)] = table
            raw_paths[_clean(path)] = path
        shape = _tensor_shape(node)
        if shape is not None:
            tensors.append((_clean(path), shape))
        row_counts |= _row_counts(node, min_rows)
        names = _column_names(node)
        if names is not None:
            named[_clean(path)] = names
        if scaler_kind(node) is not None:
            scalers.append((_clean(path), node))
        width = _declared_width(node)
        if width is not None and isinstance(node, Shell):
            widths[path] = (width, node.name)
        extremes = _extremes(node)
        if extremes is not None:
            width, scaler = extremes
            reason = (
                f"X_min_ and X_max_ keep the smallest and largest value of each column, "
                f"which the scaler at {_clean(path + '.' + scaler)} brings back to their scale"
            )
            tables.append(
                Finding(
                    _clean(path),
                    (2, width),
                    None,
                    (reason,),
                    reversible_by=_clean(path + "." + scaler),
                    owner=node.name,
                    extremes=True,
                )
            )
        siblings = _named_children(node)
        for name, child in siblings:
            cached = _cache(node, child)
            if cached is not None and cached[0][0] >= min_rows:
                tables.append(
                    Finding(
                        _clean(path + name),
                        cached[0],
                        None,
                        cached[1],
                        owner=node.name if isinstance(node, Shell) else None,
                        cache=True,
                    )
                )
                continue
            value = as_table(child)
            if value is None:
                continue
            declared[_clean(path + name)] = _declared_width(node)
            if value.shape[0] < min_rows or _is_lookup_grid(value):
                continue
            if _derived(value, _declared_width(node)):
                continue
            companion = next(
                (
                    other
                    for other, candidate in siblings
                    if (vector := as_vector(candidate)) is not None
                    and len(vector) == value.shape[0]
                ),
                None,
            )
            if companion is None:
                continue
            evidence = [f"the 1-D array {_clean(path + companion)} has one entry per row"]
            if name.lstrip(".") in TRAINING_NAMES:
                evidence.append(f"{name.lstrip('.')} is a name libraries use for training rows")
            owner = node.name if isinstance(node, Shell) else None
            tables.append(
                Finding(
                    _clean(path + name),
                    value.shape,
                    _clean(path + companion),
                    tuple(evidence),
                    owner=owner,
                )
            )

    tables += _copies(tables, arrays, declared)
    found = {finding.path for finding in tables}
    for path, array in arrays.items():
        if path in found or array.shape[0] < min_rows or _is_lookup_grid(array):
            continue
        fitted = _fitted_width(raw_paths[path], widths)
        if fitted is None or fitted[0] != array.shape[1] or _derived(array, declared.get(path)):
            continue
        estimator = _clean(fitted[1]) if fitted[1] else "the model"
        reason = f"it has {fitted[0]} columns, the n_features_in_ {estimator} declares"
        tables.append(Finding(path, array.shape, None, (reason,), owner=fitted[2]))
    tables += _copies(tables, arrays, declared)
    if not any(finding.cache for finding in tables):
        tables += _shape_cache(tensors, row_counts, min_rows)
    tables = _attach_column_names(tables, named)
    return _attach_scalers(tables, arrays, scalers)
