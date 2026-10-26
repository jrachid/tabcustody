import sys
from pathlib import Path

import numpy as np
import pytest

from tabcustody._erase import CannotErase, erase

FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES))
from customers import customers

X, Y = customers()
ERASED = 42


class Nearest:
    """Keeps its rows and its target as class indices, as scikit-learn's k-NN does, and predicts probabilities from every row."""

    def __init__(self, scale=1.0):
        self.scale = scale

    def get_params(self, deep=True):
        return {"scale": self.scale}

    def fit(self, X, y):
        self.classes_, self._y = np.unique(y, return_inverse=True)
        self._fit_X = np.asarray(X, dtype=float) * self.scale
        self.n_features_in_ = self._fit_X.shape[1]
        return self

    def predict_proba(self, inputs):
        distances = np.linalg.norm(inputs[:, None, :] * self.scale - self._fit_X[None], axis=2)
        weights = np.exp(-distances / 1e4)
        votes = np.stack([weights[:, self._y == c].sum(axis=1) for c in range(len(self.classes_))])
        return (votes / votes.sum(axis=0)).T


def probes():
    return X[:50] + 0.5


def test_an_erased_model_predicts_as_one_fitted_without_the_rows():
    model = Nearest().fit(X, Y)
    keep = np.arange(len(X)) != ERASED

    erasure = erase(model, X[[ERASED]])
    scratch = Nearest().fit(X[keep], Y[keep])

    assert erasure.removed == 1 and erasure.kept == len(X) - 1
    assert erasure.source == "recovered"
    assert (
        erasure.model.predict_proba(probes()).tobytes() == scratch.predict_proba(probes()).tobytes()
    )


def test_given_data_gives_the_same_model_as_recovered_data():
    model = Nearest().fit(X, Y)

    recovered = erase(model, X[[ERASED]])
    given = erase(model, X[[ERASED]], X, Y)

    assert given.source == "given"
    assert np.array_equal(
        recovered.model.predict_proba(probes()), given.model.predict_proba(probes())
    )


def test_text_labels_come_back_through_the_classes_not_as_indices():
    labels = np.where(Y == 1, "risky", "safe")
    model = Nearest().fit(X, labels)

    erasure = erase(model, X[[ERASED]])

    assert list(erasure.model.classes_) == ["risky", "safe"]


def test_every_copy_of_a_duplicated_row_is_removed():
    doubled = np.vstack([X, X[[ERASED]]])
    model = Nearest().fit(doubled, np.append(Y, Y[ERASED]))

    erasure = erase(model, X[[ERASED]])

    assert erasure.removed == 2


def test_an_unknown_row_is_refused_and_nothing_changes():
    model = Nearest().fit(X, Y)
    before = model._fit_X.copy()
    stranger = X[[ERASED]] + 0.01

    with pytest.raises(CannotErase, match=r"row\(s\) \[0\]") as error:
        erase(model, stranger)

    assert np.array_equal(model._fit_X, before)
    assert f"{stranger[0, 1]:.2f}" not in str(error.value)


def test_the_model_passed_in_is_not_changed():
    model = Nearest().fit(X, Y)
    before = model.predict_proba(probes())

    erase(model, X[[ERASED]])

    assert model.predict_proba(probes()).tobytes() == before.tobytes()
    assert len(model._fit_X) == len(X)


def test_an_irreversible_table_needs_the_original_data():
    model = Nearest(scale=2.0).fit(X, Y)

    with pytest.raises(CannotErase, match="pass X and y"):
        erase(model, X[[ERASED]])
    assert erase(model, X[[ERASED]], X, Y).removed == 1


def test_data_that_does_not_refit_to_the_model_is_refused():
    model = Nearest().fit(X, Y)
    other = X.copy()
    other[0, 1] += 1.0

    with pytest.raises(CannotErase, match="do not refit"):
        erase(model, X[[ERASED]], other, Y)


def test_rows_of_another_width_are_refused():
    model = Nearest().fit(X, Y)

    with pytest.raises(CannotErase, match="3"):
        erase(model, X[[ERASED], :2])
