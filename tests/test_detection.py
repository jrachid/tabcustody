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
    assert all(finding.shape[0] == ROWS for finding in found.values() if not finding.extremes)


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
    found = [finding for finding in findings("tabicl-kv-cache", large=True) if not finding.extremes]

    assert [(finding.path, finding.shape, finding.kind) for finding in found] == [
        ("model_kv_cache_", (ROWS, 3), "key-value cache")
    ]


def cache_of_an_unknown_class(rows=ROWS, cols=8):
    entries = {}
    for layer in range(2):
        key = _shell_class("mylib.cache.Entry")()
        tensor = _shell_class("torch._utils._rebuild_tensor_v2")(
            None, 0, (4, rows, cols), (0, 0, 0), False, {}
        )
        key.attributes.update(key=tensor, value=tensor)
        entries[layer] = key
    holder = _shell_class("mylib.model.Classifier")()
    holder.attributes.update(cache_={"kv": entries}, train_shape=(1, rows))
    return holder


def test_a_cache_of_an_unknown_class_is_found_by_its_shape():
    found = find_tables(cache_of_an_unknown_class())

    assert [(f.path, f.shape, f.kind) for f in found] == [
        ("cache_['kv']", (ROWS,), "key-value cache")
    ]
    assert "not a known one" in found[0].evidence[0]


def test_the_shape_rule_is_silent_without_a_known_row_count():
    holder = _shell_class("mylib.model.Classifier")()
    tensor = _shell_class("torch._utils._rebuild_tensor_v2")(
        None, 0, (4, ROWS, 8), (0, 0, 0), False, {}
    )
    holder.attributes.update(cache_={"kv": {0: tensor}})

    assert find_tables(holder) == []


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


TABICL_EXTREMES = "ensemble_generator_.preprocessors_['power']"


def test_tabicl_column_extremes_are_reported_beside_the_table():
    found = [finding for finding in findings("tabicl") if finding.extremes]

    assert [(finding.path, finding.shape, finding.kind) for finding in found] == [
        (TABICL_EXTREMES, (2, 3), "column extremes")
    ]
    assert found[0].reversible_by == f"{TABICL_EXTREMES}.standard_scaler_"


@pytest.mark.integration
def test_tabicl_column_extremes_are_reported_even_without_the_table():
    found = {finding.kind: finding for finding in findings("tabicl-kv-cache", large=True)}

    assert set(found) == {"key-value cache", "column extremes"}
    assert found["column extremes"].path == TABICL_EXTREMES


def preprocessor_with_extremes(width=3, scaler_width=3):
    scaler = _shell_class("tabicl._sklearn.preprocessing.CustomStandardScaler")()
    scaler.attributes.update(mean_=np.zeros(scaler_width), scale_=np.ones(scaler_width))
    pipeline = _shell_class("tabicl._sklearn.preprocessing.PreprocessingPipeline")()
    pipeline.attributes.update(
        standard_scaler_=scaler, X_min_=np.zeros((1, width)), X_max_=np.ones((1, width))
    )
    return pipeline


def test_column_extremes_need_a_scaler_of_their_width():
    assert [finding.kind for finding in find_tables(preprocessor_with_extremes())] == [
        "column extremes"
    ]
    assert find_tables(preprocessor_with_extremes(scaler_width=4)) == []


def test_a_preprocessed_table_is_not_given_the_input_column_names():
    found = findings("tabpfn-v2", large=True)

    assert found
    assert all(finding.columns is None for finding in found)


def test_a_transformed_copy_is_reported_next_to_the_table_it_copies():
    copies = [finding for finding in findings("tabicl") if finding.kind == "copy"]

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
    ("name", "table"),
    [
        ("nearest-neighbors", "_fit_X"),
        ("kernel-density", "tree_<state>[0]"),
        ("kernel-pca", "X_fit_"),
        ("isomap", "nbrs_._fit_X"),
    ],
)
def test_models_without_a_target_are_reported_by_the_width_they_were_fitted_on(name, table):
    found = {finding.path: finding for finding in findings(name)}

    assert found[table].kind == "training table"
    assert found[table].shape == (ROWS, 3)
    assert any("n_features_in_" in reason for reason in found[table].evidence)


def test_arrays_of_another_width_beside_rows_without_a_target_are_not_reported():
    found = {finding.path for finding in findings("isomap")}

    assert found == {"nbrs_._fit_X"}


@pytest.mark.parametrize(
    ("name", "rows", "factor"),
    [
        ("gaussian-process-regressor", "X_train_", "L_"),
        ("gaussian-process-classifier", "base_estimator_.X_train_", "base_estimator_.L_"),
    ],
)
def test_a_cholesky_factor_is_not_called_a_training_table(name, rows, factor):
    found = {finding.path: finding for finding in findings(name)}

    assert found[rows].kind == "training table"
    assert factor not in found


def test_neighbour_distances_are_not_called_a_training_table():
    found = {finding.path: finding for finding in findings("local-outlier-factor")}

    assert found["_fit_X"].kind == "training table"
    assert "_distances_fit_X_" not in found


def test_out_of_bag_probabilities_are_not_a_copy():
    found = {finding.path: finding for finding in findings("stacking")}

    assert found["estimators_[0]._fit_X"].kind == "training table"
    assert "estimators_[1].oob_decision_function_" not in found


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
