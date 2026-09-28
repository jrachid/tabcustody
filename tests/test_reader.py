import pickle
from collections import OrderedDict

import numpy as np
import pytest

from tabcustody._reader import Shell, read_bytes


class Estimator:
    def __init__(self) -> None:
        self.X_train = np.arange(12.0).reshape(4, 3)
        self.y_train = np.array([0, 1, 0, 1])
        self.params = {"n": 3}


class Records(list):
    pass


class Registry(dict):
    pass


class Slotted:
    __slots__ = ("weights",)

    def __init__(self) -> None:
        self.weights = np.ones((2, 2), dtype=np.float32)


@pytest.mark.parametrize("protocol", range(2, pickle.HIGHEST_PROTOCOL + 1))
def test_numpy_arrays_come_back_equal_whatever_the_pickle_protocol(protocol):
    arrays = {
        "floats": np.linspace(0, 1, 7),
        "float32_matrix": np.arange(6, dtype=np.float32).reshape(2, 3),
        "ints": np.array([3, 1, 2], dtype=np.int64),
        "fortran": np.asfortranarray(np.arange(6.0).reshape(2, 3)),
        "scalar": np.float64(2.5),
    }

    tree = read_bytes(pickle.dumps(arrays, protocol=protocol))

    for name, original in arrays.items():
        assert np.array_equal(tree[name], original), name
        assert np.asarray(tree[name]).dtype == np.asarray(original).dtype, name


def test_an_unknown_class_becomes_a_shell_that_keeps_its_name_and_attributes():
    tree = read_bytes(pickle.dumps(Estimator()))

    assert isinstance(tree, Shell)
    assert tree.name == f"{__name__}.Estimator"
    assert np.array_equal(tree.attributes["X_train"], np.arange(12.0).reshape(4, 3))
    assert tree.attributes["params"] == {"n": 3}


def test_slotted_objects_and_container_subclasses_keep_their_content():
    records, registry = Records([np.zeros(2)]), Registry(key=np.ones(3))

    slotted, loaded_records, loaded_registry = read_bytes(
        pickle.dumps((Slotted(), records, registry))
    )

    assert np.array_equal(slotted.attributes["weights"], np.ones((2, 2), dtype=np.float32))
    assert np.array_equal(loaded_records.items[0], np.zeros(2))
    assert np.array_equal(loaded_registry.entries["key"], np.ones(3))


def test_plain_containers_stay_plain():
    tree = read_bytes(pickle.dumps(OrderedDict(a=[1, 2], b=(3, {4}), c=frozenset({5}))))

    assert tree == {"a": [1, 2], "b": (3, {4}), "c": frozenset({5})}
