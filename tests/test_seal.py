import pickle

import numpy as np

from tabcustody._detect import find_tables
from tabcustody._reader import read_bytes
from tabcustody._seal import Sealed, lifted, training_refs

ROWS = 300


class Estimator:
    def __init__(self):
        self._fit_X = np.arange(ROWS * 3, dtype=float).reshape(ROWS, 3)
        self._y = np.arange(ROWS)
        self.n_features_in_ = 3


class Pipeline:
    def __init__(self):
        self.steps = [("scale", object()), ("knn", Estimator())]


def test_the_training_table_and_its_target_are_found_in_a_live_model():
    model = Estimator()

    refs = training_refs(model)

    assert {ref.key for ref in refs} == {"_fit_X", "_y"}


def test_sealing_swaps_the_table_for_a_marker_and_the_saved_model_carries_no_table():
    model = Estimator()

    with lifted(model) as objects:
        assert isinstance(model._fit_X, Sealed) and isinstance(model._y, Sealed)
        assert [o.shape for o in objects] == [(ROWS, 3), (ROWS,)]
        assert find_tables(read_bytes(pickle.dumps(model))) == []


def test_the_model_is_whole_again_after_sealing_even_when_the_block_raises():
    model = Estimator()
    before = model._fit_X.copy()

    try:
        with lifted(model):
            raise RuntimeError("saving failed")
    except RuntimeError:
        pass

    assert np.array_equal(model._fit_X, before)
    assert not isinstance(model._fit_X, Sealed)


def test_a_table_nested_in_a_pipeline_is_found_and_restored():
    model = Pipeline()
    knn = model.steps[1][1]

    with lifted(model):
        assert isinstance(knn._fit_X, Sealed)
    assert knn._fit_X.shape == (ROWS, 3)
