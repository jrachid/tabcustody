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


K1, K2 = b"1" * 32, b"2" * 32


def sealed_v1(tmp_path):
    from tabcustody import seal

    old = tmp_path / "v1.sealed"
    seal(Nearest().fit(X, Y), old, K1)
    return old


def test_erasing_a_file_writes_a_sealed_version_without_the_rows_and_a_receipt(tmp_path):
    from tabcustody import audit_key, check_absent, erase_file, file_id, opens, verify_receipt

    old, new = sealed_v1(tmp_path), tmp_path / "v2.sealed"

    receipt = erase_file(old, new, X[[ERASED]], K2, old_keys=K1)

    report = check_absent(new, X[[ERASED, 7]], keys=K2)
    assert report.present == (1,) and report.verified
    assert receipt["removed"] == 1 and receipt["after"]["id"] == file_id(new)
    assert receipt["before"]["id"] == file_id(old) and receipt["before"]["sealed"]
    assert receipt["check"]["verified"] and not opens(old, K2)
    assert verify_receipt(receipt, audit_key(K2))
    assert (tmp_path / "v2.sealed.receipt.json").exists()


def test_a_changed_receipt_is_refused(tmp_path):
    from tabcustody import audit_key, erase_file, verify_receipt

    receipt = erase_file(sealed_v1(tmp_path), tmp_path / "v2.sealed", X[[ERASED]], K2, K1)

    assert not verify_receipt({**receipt, "removed": 0}, audit_key(K2))
    assert not verify_receipt(receipt, audit_key(K1))


def test_no_written_file_holds_the_erased_rows(tmp_path):
    from tabcustody import erase_file

    erase_file(sealed_v1(tmp_path), tmp_path / "v2.sealed", X[[ERASED]], K2, K1)
    written = [tmp_path / "v2.sealed", tmp_path / "v2.sealed.receipt.json"]
    blobs = b"".join(path.read_bytes() for path in written)

    for value in X[ERASED]:
        assert np.float64(value).tobytes() not in blobs
        assert repr(float(value)).encode() not in blobs
        assert f"{value:.2f}".encode() not in blobs


def test_a_new_key_that_opens_the_old_version_is_refused_and_nothing_is_written(tmp_path):
    from tabcustody import CannotErase, erase_file

    new = tmp_path / "v2.sealed"

    with pytest.raises(CannotErase, match="fresh key"):
        erase_file(sealed_v1(tmp_path), new, X[[ERASED]], K1, K1)
    assert not new.exists()


def test_an_old_version_in_clear_is_named_as_such(tmp_path):
    import pickle

    from tabcustody import erase_file

    old = tmp_path / "v1.pkl"
    old.write_bytes(pickle.dumps(Nearest().fit(X, Y)))

    receipt = erase_file(old, tmp_path / "v2.sealed", X[[ERASED]], K2)

    assert not receipt["before"]["sealed"]
    assert "no key can make them unreadable" in receipt["old_version"]


def test_the_commands_erase_and_check_the_receipt(tmp_path, capsys):
    from tabcustody import audit_key
    from tabcustody._cli import main

    old = sealed_v1(tmp_path)
    (tmp_path / "k1").write_bytes(K1)
    (tmp_path / "k2").write_bytes(K2)
    (tmp_path / "audit").write_text(audit_key(K2).hex())
    np.savetxt(tmp_path / "rows.csv", X[[ERASED]], delimiter=",")
    new = tmp_path / "v2.sealed"
    receipt = tmp_path / "v2.sealed.receipt.json"

    args = ["erase", str(old), str(new), "--rows", str(tmp_path / "rows.csv")]
    keys = ["--new-key-file", str(tmp_path / "k2"), "--key-file", str(tmp_path / "k1")]
    assert main([*args, *keys]) == 0
    assert "erased 1 training row(s)" in capsys.readouterr().out
    assert main(["check-receipt", str(receipt), "--audit-key-file", str(tmp_path / "audit")]) == 0
    receipt.write_text(receipt.read_text().replace('"removed": 1', '"removed": 2'))
    assert main(["check-receipt", str(receipt), "--audit-key-file", str(tmp_path / "audit")]) == 1


class Named(Nearest):
    """Keeps the column names it was fitted with, as scikit-learn does with a DataFrame."""

    def fit(self, X, y):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return super().fit(np.asarray(X), y)

    def predict_proba(self, inputs):
        assert list(inputs.columns) == list(self.feature_names_in_)
        return super().predict_proba(np.asarray(inputs))


def test_column_names_the_model_was_fitted_with_are_kept():
    pandas = pytest.importorskip("pandas")
    model = Named().fit(pandas.DataFrame(X, columns=["age", "income", "debt"]), Y)

    erasure = erase(model, X[[ERASED]])

    assert list(erasure.model.feature_names_in_) == ["age", "income", "debt"]
