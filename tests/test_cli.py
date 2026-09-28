import json
import subprocess
import sys
from pathlib import Path

from tabcustody._cli import main

FIXTURES = Path(__file__).parent / "fixtures"


def test_the_command_exits_non_zero_when_a_table_is_found(capsys):
    assert main(["scan", str(FIXTURES / "kneighbors.pkl")]) == 1
    assert "_fit_X" in capsys.readouterr().out


def test_the_command_exits_zero_on_clean_files():
    assert main(["scan", str(FIXTURES / "xgboost.pkl"), str(FIXTURES / "lightgbm.pkl")]) == 0


def test_an_unreadable_file_exits_with_two(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not a pickle")

    assert main(["scan", str(path), str(FIXTURES / "kneighbors.pkl")]) == 2


def test_json_output_lists_one_report_per_file(capsys):
    main(["scan", "--json", str(FIXTURES / "kneighbors.pkl"), str(FIXTURES / "xgboost.pkl")])

    reports = json.loads(capsys.readouterr().out)

    assert [report["carries_training_data"] for report in reports] == [True, False]


def test_the_package_runs_as_a_module():
    result = subprocess.run(
        [sys.executable, "-m", "tabcustody", "scan", str(FIXTURES / "xgboost.pkl")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
