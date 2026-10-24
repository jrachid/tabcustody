import json
import re
from pathlib import Path

import pytest

from tabcustody import scan_file
from tabcustody._detect import Finding
from tabcustody._report import Report, _hint

FIXTURES = Path(__file__).parent / "fixtures"
DECIMAL = re.compile(r"(?<![\d.])\d+\.\d+(?![\d.])")


@pytest.mark.parametrize(
    "name",
    ["kneighbors.pkl", "tabicl.pkl", "standard-kneighbors.pkl", "svc.pkl", "tabpfn-v2.tabpfn_fit"],
)
def test_the_report_never_prints_a_value_of_the_table(name):
    report = scan_file(FIXTURES / name)
    rendered = report.to_text() + json.dumps(report.to_dict())

    assert report.findings
    assert DECIMAL.findall(rendered) == []


def test_a_tabicl_finding_warns_that_the_cache_option_still_lets_the_rows_be_rebuilt():
    report = scan_file(FIXTURES / "tabicl.pkl")

    table = next(finding for finding in report.findings if finding.companion is not None)

    assert "save_training_data=False" in table.hint
    assert "rebuilt" in table.hint
    assert "confidential" in table.hint


def test_a_tabpfn_finding_does_not_present_the_cache_mode_as_a_fix():
    report = scan_file(FIXTURES / "tabpfn-v2.tabpfn_fit")

    hint = report.findings[0].hint

    assert "fit_with_cache" in hint
    assert "key-value cache" in hint
    assert "confidential" in hint


@pytest.mark.integration
def test_a_tabicl_file_saved_without_its_training_data_still_carries_it():
    path = FIXTURES / "large" / "tabicl-kv-cache.pkl"
    if not path.exists():
        pytest.skip(f"{path} is built by `tests/fixtures/build.py --large`")

    report = scan_file(path)
    text = report.to_text()

    assert report.carries_training_data
    assert "key-value cache  model_kv_cache_  300 rows x 3 columns" in text
    assert "rebuilt" in report.findings[0].hint
    assert json.loads(json.dumps(report.to_dict()))["findings"][0]["kind"] == "key-value cache"


def cache_finding(owner):
    return Finding("kv_caches", (300, 3), None, (), owner=owner, cache=True)


def test_a_tabpfn_cache_finding_does_not_reuse_tabicl_advice():
    hint = _hint(cache_finding("tabpfn.inference.InferenceEngineExplicitKVCache"))

    assert "TabICL" not in hint
    assert "TabPFN" in hint
    assert "not been measured" in hint
    assert "confidential" in hint


def test_a_tabicl_cache_finding_keeps_its_own_advice():
    hint = _hint(cache_finding("tabicl._sklearn.classifier.TabICLClassifier"))

    assert hint.startswith("TabICL's key-value cache")


def test_a_cache_without_rows_and_columns_reports_its_rows_alone():
    finding = Finding("kv_caches", (300,), None, ("why",), cache=True)
    report = Report("model.pkl", "pickle", (finding,))

    assert "key-value cache  kv_caches  300 rows\n" in report.to_text() + "\n"
    assert report.to_dict()["findings"][0]["columns"] is None


def test_column_extremes_alone_do_not_mean_the_file_carries_training_data():
    extremes = Finding("pipeline", (2, 3), None, ("why",), extremes=True)
    report = Report("model.pkl", "pickle", (extremes,))

    assert not report.carries_training_data
    assert "no training table found" in report.to_text()
    assert "column extremes  pipeline  smallest and largest value of 3 columns" in report.to_text()
    assert report.to_dict()["findings"][0]["kind"] == "column extremes"


def test_column_extremes_say_whose_values_they_are():
    report = scan_file(FIXTURES / "tabicl.pkl")

    extremes = next(finding for finding in report.findings if finding.extremes)

    assert "confidential" in extremes.hint
    assert "save_training_data" not in extremes.hint


@pytest.mark.integration
@pytest.mark.parametrize("name", ["tabpfn-v2-cache.tabpfn_fit", "tabpfn-v3.5-cache.tabpfn_fit"])
def test_a_tabpfn_file_fitted_with_cache_carries_its_training_data(name):
    path = FIXTURES / "large" / name
    if not path.exists():
        pytest.skip(f"{path} is built by `tests/fixtures/build.py --tabpfn-cache`")

    report = scan_file(path)

    assert report.carries_training_data
    assert (
        "key-value cache  executor_state.joblib:kv_caches  300 rows x 3 columns" in report.to_text()
    )
    assert "not been measured" in report.findings[0].hint


@pytest.mark.parametrize(
    ("name", "library"),
    [
        ("tabpfn-v2.tabpfn_fit", "TabPFN"),
        ("kneighbors.pkl", "instance-based"),
        ("svc.pkl", "instance-based"),
    ],
)
def test_models_without_a_fix_say_so(name, library):
    report = scan_file(FIXTURES / name)

    table = next(finding for finding in report.findings if finding.companion is not None)

    assert library in table.hint


def test_a_clean_file_says_it_carries_no_training_table():
    report = scan_file(FIXTURES / "xgboost.pkl")

    assert not report.carries_training_data
    assert "no training table" in report.to_text()


def test_findings_that_differ_only_by_an_index_are_grouped_in_the_text():
    report = scan_file(FIXTURES / "tabpfn-v2.tabpfn_fit")
    text = report.to_text()

    assert len(report.findings) == 8
    assert "executor_state.joblib:ensemble_members[*].X_train" in text
    assert "(8 alike)" in text
    assert text.count("fix:") == 1
    assert len(report.to_dict()["findings"]) == 8
