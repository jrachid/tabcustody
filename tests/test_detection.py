from pathlib import Path

import numpy as np
import pytest

from tabcustody._detect import as_table, find_tables
from tabcustody._reader import _shell_class, read, read_members

FIXTURES = Path(__file__).parent / "fixtures"
ROWS = 300


def findings(name: str, large: bool = False):
    path = FIXTURES / ("large" if large else "") / f"{name}.pkl"
    if not path.exists():
        pytest.skip(f"{path} is built by `tests/fixtures/build.py --large`")
    return find_tables(read(path))


IN_CONTEXT = [
    pytest.param("tabicl", False, "ensemble_generator_.X_", "ensemble_generator_.y_", id="tabicl"),
    pytest.param(
        "tabpfn-v2",
        True,
        "executor_.ensemble_members[0].X_train",
        "executor_.ensemble_members[0].y_train",
        id="tabpfn-v2",
        marks=pytest.mark.integration,
    ),
    pytest.param(
        "tabpfn-v3.5",
        True,
        "executor_.ensemble_members[0].X_train",
        "executor_.ensemble_members[0].y_train",
        id="tabpfn-v3.5",
        marks=pytest.mark.integration,
    ),
    pytest.param("tabdpt", True, "X_train", "y_train", id="tabdpt", marks=pytest.mark.integration),
]


@pytest.mark.parametrize(("name", "large", "table", "target"), IN_CONTEXT)
def test_the_training_table_of_each_in_context_model_is_found(name, large, table, target):
    found = {finding.path: finding for finding in findings(name, large)}

    assert table in found
    assert found[table].shape[0] == ROWS
    assert found[table].companion == target
    assert all(finding.shape[0] == ROWS for finding in found.values())


def tabicl_saved_with_its_cache_only():
    cache = _shell_class("tabicl._model.kv_cache.TabICLCache")()
    cache.attributes.update(
        icl_cache=_shell_class("tabicl._model.kv_cache.KVCache")(), train_shape=(4, ROWS, 3)
    )
    model = _shell_class("tabicl._sklearn.classifier.TabICLClassifier")()
    model.attributes.update(model_kv_cache_={"none": cache, "power": cache})
    return model


def test_a_tabicl_key_value_cache_is_reported_as_the_training_rows_it_was_built_from():
    found = find_tables(tabicl_saved_with_its_cache_only())

    assert [(finding.path, finding.shape, finding.kind) for finding in found] == [
        ("model_kv_cache_", (ROWS, 3), "key-value cache")
    ]


@pytest.mark.integration
def test_a_tabicl_file_saved_without_its_training_data_still_reports_its_cache():
    found = findings("tabicl-kv-cache", large=True)

    assert [(finding.path, finding.shape, finding.kind) for finding in found] == [
        ("model_kv_cache_", (ROWS, 3), "key-value cache")
    ]


def tabpfn_fitted_with_cache(entry="KVCacheEntry", columns=3):
    entries = {}
    for layer in range(2):
        entries[layer] = _shell_class(f"tabpfn.architectures.kv_cache.{entry}")()
        entries[layer].attributes.update(key=None, value=None)
    cache = _shell_class("tabpfn.architectures.tabpfn_v2.TabPFNV2Cache")()
    cache.attributes.update(kv=entries, train_shape=(1, ROWS))
    executor = _shell_class("tabpfn.inference.InferenceEngineExplicitKVCache")()
    executor.attributes.update(kv_caches=[cache, cache])
    if columns is not None:
        schema = _shell_class("tabpfn.preprocessing.datamodel.FeatureSchema")()
        schema.attributes.update(features=[object()] * columns)
        preprocessor = _shell_class("tabpfn.preprocessing.ensemble.TabPFNEnsemblePreprocessor")()
        preprocessor.attributes.update(feature_schema=schema)
        executor.attributes.update(ensemble_preprocessor=preprocessor)
    return executor


@pytest.mark.parametrize("entry", ["KVCacheEntry", "QuantizedKVCacheEntry"])
def test_a_tabpfn_key_value_cache_is_reported_as_the_training_rows_it_was_built_from(entry):
    found = find_tables(tabpfn_fitted_with_cache(entry))

    assert [(finding.path, finding.shape, finding.kind) for finding in found] == [
        ("kv_caches", (ROWS, 3), "key-value cache")
    ]
    assert "TabPFN" in found[0].evidence[0]


def test_a_quantized_tabpfn_cache_says_it_is_quantized():
    found = find_tables(tabpfn_fitted_with_cache("QuantizedKVCacheEntry"))

    assert any("8 bits" in reason for reason in found[0].evidence)


def test_a_tabpfn_cache_without_its_feature_schema_is_still_reported_by_its_rows():
    found = find_tables(tabpfn_fitted_with_cache(columns=None))

    assert [(finding.path, finding.shape) for finding in found] == [("kv_caches", (ROWS,))]


TABPFN_CACHE_FILES = [
    pytest.param(f"tabpfn-{weights}-cache.{suffix}", id=f"{weights}-{suffix}")
    for weights, suffix in [
        ("v2", "tabpfn_fit"),
        ("v2", "pkl"),
        ("v3.5", "tabpfn_fit"),
        ("v3.5-adaptive", "tabpfn_fit"),
    ]
]


@pytest.mark.integration
@pytest.mark.parametrize("name", TABPFN_CACHE_FILES)
def test_a_tabpfn_file_fitted_with_cache_reports_its_cache(name):
    path = FIXTURES / "large" / name
    if not path.exists():
        pytest.skip(f"{path} is built by `tests/fixtures/build.py --tabpfn-cache`")

    found = [
        finding
        for member, tree in read_members(path)[1]
        for finding in find_tables(tree, prefix=f"{member}:" if member else "")
    ]

    assert [(finding.path.split(":")[-1], finding.shape, finding.kind) for finding in found] == [
        (("" if name.endswith("fit") else "executor_.") + "kv_caches", (ROWS, 3), "key-value cache")
    ]


def test_a_transformed_copy_is_reported_next_to_the_table_it_copies():
    copies = [finding for finding in findings("tabicl") if finding.companion is None]

    assert copies
    assert all("ensemble_generator_.X_" in " ".join(copy.evidence) for copy in copies)


@pytest.mark.parametrize(
    ("name", "table", "companion"),
    [("kneighbors", "_fit_X", "_y"), ("svc", "support_vectors_", "support_")],
)
def test_instance_based_models_are_reported_like_sacroml_does(name, table, companion):
    found = {finding.path: finding for finding in findings(name)}

    assert found[table].companion == companion


@pytest.mark.parametrize(
    "name",
    [
        "xgboost",
        "lightgbm",
        "logistic-regression",
        "random-forest",
        "gradient-boosting",
        "mlp",
        "quantile-pipeline",
    ],
)
def test_classic_models_report_no_table(name):
    assert findings(name) == []


def test_a_square_weight_matrix_next_to_its_bias_is_not_a_table():
    tree = read(FIXTURES / "mlp.pkl")
    coefs, intercepts = tree.attributes["coefs_"], tree.attributes["intercepts_"]

    assert coefs[1].shape == (100, 100) and intercepts[0].shape == (100,)
    assert find_tables(tree) == []


def test_a_quantile_grid_beside_its_levels_is_not_a_table():
    tree = read(FIXTURES / "quantile-pipeline.pkl")
    transformer = tree.attributes["steps"][0][1]

    assert transformer.attributes["quantiles_"].shape == (100, 3)
    assert find_tables(tree) == []


@pytest.mark.parametrize("pandas", ["pandas2", "pandas3"])
def test_a_table_kept_as_a_pandas_dataframe_is_found(pandas):
    found = {finding.path: finding for finding in findings(f"frame-{pandas}")}

    assert found["X_train"].shape == (ROWS, 3)
    assert found["X_train"].companion == "y_train"


@pytest.mark.parametrize("pandas", ["pandas2", "pandas3"])
def test_a_dataframe_is_rebuilt_column_by_column_in_its_own_order(pandas):
    import sys

    sys.path.insert(0, str(FIXTURES))
    from customers import customers

    tree = read(FIXTURES / f"frame-{pandas}.pkl")

    np.testing.assert_array_equal(as_table(tree.attributes["X_train"]), customers()[0])
