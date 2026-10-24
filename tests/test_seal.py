import numpy as np

from tabcustody._detect import find_tables
from tabcustody._reader import read_bytes
from tabcustody._seal import lift, restore, training_arrays

ROWS = 300


class Tree:
    __slots__ = ("data",)


class Estimator:
    def __init__(self):
        self._fit_X = np.arange(ROWS * 3, dtype=float).reshape(ROWS, 3)
        self._y = np.arange(ROWS)
        self.n_features_in_ = 3
        # The fitted tree shares the same array object, as a real kNN's KD tree does.
        self._tree = Tree()
        self._tree.data = self._fit_X


class Pipeline:
    def __init__(self):
        self.steps = [("scale", object()), ("knn", Estimator())]


def test_the_training_table_and_its_target_are_found_in_a_live_model():
    arrays = training_arrays(Estimator())

    shapes = sorted((a.shape for a in arrays), key=len)
    assert (ROWS,) in shapes and (ROWS, 3) in shapes


def test_the_sealed_bytes_carry_no_training_table_even_the_shared_copy():
    model_bytes, lifted = lift(Estimator())

    assert [a.shape for a in lifted] == [(ROWS, 3), (ROWS,)]
    assert find_tables(read_bytes(model_bytes)) == []


def test_restoring_gives_the_exact_arrays_back_and_keeps_them_shared():
    model = Estimator()

    model_bytes, lifted = lift(model)
    back = restore(model_bytes, lifted)

    assert np.array_equal(back._fit_X, model._fit_X)
    assert np.array_equal(back._y, model._y)
    assert back._tree.data is back._fit_X


def test_the_live_model_is_not_touched_by_sealing():
    model = Estimator()
    before = model._fit_X.copy()

    lift(model)

    assert np.array_equal(model._fit_X, before)
    assert isinstance(model._fit_X, np.ndarray)


def test_a_table_nested_in_a_pipeline_is_sealed_and_restored():
    model = Pipeline()

    model_bytes, lifted = lift(model)
    back = restore(model_bytes, lifted)

    assert find_tables(read_bytes(model_bytes)) == []
    assert np.array_equal(back.steps[1][1]._fit_X, model.steps[1][1]._fit_X)
