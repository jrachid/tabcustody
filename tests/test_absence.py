import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from tabcustody._absence import check_absent, read_rows
from tabcustody._cli import main

FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES))
from customers import customers

X, Y = customers()
ERASED = 42


def saved(tmp_path, model, name="model.pkl"):
    path = tmp_path / name
    path.write_bytes(pickle.dumps(model))
    return path


def without(index):
    keep = np.arange(len(X)) != index
    return {"_fit_X": X[keep], "_y": Y[keep]}


def test_a_kept_row_is_found_and_an_erased_one_is_absent(tmp_path):
    path = saved(tmp_path, without(ERASED))

    report = check_absent(path, X[[ERASED, 7]])

    assert report.present == (1,)
    assert report.rows[1].tables == ("['_fit_X']",)
    assert report.verified and not report.rows[0].present


def test_a_row_one_cent_away_is_not_taken_for_a_kept_row(tmp_path):
    path = saved(tmp_path, without(ERASED))
    near = X[7].copy()
    near[1] += 0.01

    assert check_absent(path, near).present == ()


def test_a_row_behind_a_scaler_is_found_once_the_scaler_is_reversed():
    report = check_absent(FIXTURES / "standard-kneighbors.pkl", X[[5]])

    assert report.present == (0,)
    assert report.searched == ("steps[1][1]._fit_X",)


def test_a_float32_table_still_finds_the_rows_it_holds(tmp_path):
    keep = np.arange(len(X)) != ERASED
    path = saved(tmp_path, {"_fit_X": X[keep].astype(np.float32), "_y": Y[keep]})

    report = check_absent(path, X[[ERASED, 7]])

    assert report.present == (1,)


def test_a_table_narrower_than_the_rows_leaves_absence_unshown(tmp_path):
    path = saved(tmp_path, {"_fit_X": X[:, :2], "_y": Y})

    report = check_absent(path, X[[ERASED]])

    assert not report.verified and report.present == ()
    assert "absence is not shown" in report.to_text()


def test_decoys_drawn_from_the_table_match_nothing():
    report = check_absent(FIXTURES / "kneighbors.pkl", X[[ERASED]])

    assert report.decoys > 150 and report.decoys_matched == 0


def test_a_sealed_file_is_refused_rather_than_called_empty(tmp_path):
    from tabcustody import seal

    path = tmp_path / "model.sealed"
    seal(without(ERASED), path, b"k" * 32)

    report = check_absent(path, X[[7]])

    assert report.error is not None and "sealed" in report.error
    assert not report.verified


def test_the_report_holds_no_value_of_the_given_rows(tmp_path):
    path = saved(tmp_path, without(ERASED))

    report = check_absent(path, X[[ERASED, 7]])
    out = report.to_text() + json.dumps(report.to_dict())

    for value in X[[ERASED, 7]].ravel():
        assert f"{value:.2f}" not in out and repr(float(value)) not in out


def test_a_bad_line_is_named_without_its_content(tmp_path):
    rows = tmp_path / "rows.csv"
    rows.write_text("age,income,debt\n62,38693.02,8319.57\n51,secret-value,9688.95\n")

    with pytest.raises(ValueError) as error:
        read_rows(rows)

    assert "line 3" in str(error.value) and "secret" not in str(error.value)


def test_the_command_exits_by_what_it_could_show(tmp_path, capsys):
    path = saved(tmp_path, without(ERASED))
    absent, present = tmp_path / "absent.csv", tmp_path / "present.npy"
    np.savetxt(absent, X[[ERASED]], delimiter=",")
    np.save(present, X[[7]])
    narrow = saved(tmp_path, {"_fit_X": X[:, :2], "_y": Y}, "narrow.pkl")

    assert main(["check", "--absent", str(absent), str(path)]) == 0
    assert main(["check", "--absent", str(present), str(path)]) == 1
    assert main(["check", "--absent", str(absent), str(narrow)]) == 3
    assert "verdict: all 1 row(s) absent" in capsys.readouterr().out


@pytest.mark.integration
def test_tabpfn_members_hold_the_rows_among_transformed_columns():
    report = check_absent(FIXTURES / "large" / "tabpfn-v2.pkl", X[[ERASED]])

    assert report.present == (0,)
    assert len(report.rows[0].tables) == 8


@pytest.mark.integration
def test_check_never_claims_absence_from_an_unverified_cache():
    report = check_absent(FIXTURES / "large" / "tabicl-kv-cache.pkl", X[[ERASED]])

    assert not report.verified
    assert any("key-value cache" in place for place in report.unverifiable)
    assert "absence is not shown" in report.to_text()


@pytest.mark.integration
def test_a_person_at_a_column_end_may_survive_as_an_extreme_without_the_table():
    richest = int(np.argmax(X[:, 1]))

    report = check_absent(FIXTURES / "large" / "tabicl-kv-cache.pkl", X[[richest]])

    assert report.present == () and report.uncertain == (0,)
    assert report.rows[0].extremes == ("ensemble_generator_.preprocessors_['power']",)
    assert "may be present" in report.to_text()


def test_an_extreme_shared_with_rows_still_in_the_table_is_not_this_persons():
    youngest_not_in_table = np.array([[X[:, 0].min(), 12345.67, 1111.11]])

    report = check_absent(FIXTURES / "tabicl.pkl", youngest_not_in_table)

    assert report.present == () and report.uncertain == ()
    assert report.verified
