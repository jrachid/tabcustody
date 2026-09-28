import json
import re
from pathlib import Path

import pytest

from tabcustody import scan_file

FIXTURES = Path(__file__).parent / "fixtures"
DECIMAL = re.compile(r"\d+\.\d+")


@pytest.mark.parametrize(
    "name",
    ["kneighbors.pkl", "tabicl.pkl", "standard-kneighbors.pkl", "svc.pkl", "tabpfn-v2.tabpfn_fit"],
)
def test_the_report_never_prints_a_value_of_the_table(name):
    report = scan_file(FIXTURES / name)
    rendered = report.to_text() + json.dumps(report.to_dict())

    assert report.findings
    assert DECIMAL.findall(rendered) == []


def test_a_tabicl_finding_names_the_save_option_that_drops_the_table():
    report = scan_file(FIXTURES / "tabicl.pkl")

    table = next(finding for finding in report.findings if finding.companion is not None)

    assert "kv_cache=True" in table.hint
    assert "save_training_data=False" in table.hint


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
