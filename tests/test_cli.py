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


def test_seal_then_unseal_round_trips_through_the_command(tmp_path, capsys):
    import numpy as np

    model = {"_fit_X": np.arange(60.0).reshape(20, 3), "_y": np.arange(20)}
    source = tmp_path / "model.pkl"
    source.write_bytes(__import__("pickle").dumps(model))
    sealed = tmp_path / "model.sealed"
    rebuilt = tmp_path / "back.pkl"
    key = tmp_path / "key"
    key.write_bytes(b"\x07" * 32)

    assert main(["seal", str(source), str(sealed), "--key-file", str(key)]) == 0
    assert not scan_carries(sealed)
    assert main(["unseal", str(sealed), str(rebuilt), "--key-file", str(key)]) == 0

    back = __import__("pickle").loads(rebuilt.read_bytes())
    import numpy as np

    assert np.array_equal(back["_fit_X"], model["_fit_X"])


def test_sealing_without_a_key_exits_two(tmp_path, monkeypatch):
    monkeypatch.delenv("TABCUSTODY_KEY", raising=False)
    source = tmp_path / "model.pkl"
    source.write_bytes(__import__("pickle").dumps({"x": 1}))

    assert main(["seal", str(source), str(tmp_path / "out.sealed")]) == 2


def test_unsealing_with_a_wrong_key_exits_three(tmp_path):
    import numpy as np

    model = {"_fit_X": np.arange(60.0).reshape(20, 3), "_y": np.arange(20)}
    source = tmp_path / "model.pkl"
    source.write_bytes(__import__("pickle").dumps(model))
    sealed = tmp_path / "model.sealed"
    right, wrong = tmp_path / "right", tmp_path / "wrong"
    right.write_bytes(b"\x01" * 32)
    wrong.write_bytes(b"\x02" * 32)

    main(["seal", str(source), str(sealed), "--key-file", str(right)])

    assert main(["unseal", str(sealed), str(tmp_path / "back.pkl"), "--key-file", str(wrong)]) == 3


def scan_carries(path) -> bool:
    from tabcustody import scan_file

    return scan_file(path).carries_training_data


def test_a_logged_identifier_and_an_audit_key_check_a_proof_from_the_command_line(
    tmp_path, capsys, monkeypatch
):
    import numpy as np

    from tabcustody import seal

    path = tmp_path / "model.sealed"
    rows = np.arange(120, dtype=float).reshape(40, 3)
    seal({"X": rows, "y": np.arange(40)}, path, b"k" * 32)
    monkeypatch.setenv("TABCUSTODY_KEY", "k" * 32)

    assert main(["id", str(path)]) == 0
    logged = capsys.readouterr().out.strip()
    assert main(["prove", str(path), "--position", "3"]) == 0
    (tmp_path / "proof.json").write_text(capsys.readouterr().out)
    assert main(["audit-key"]) == 0
    (tmp_path / "audit").write_text(capsys.readouterr().out)

    check = [
        "check-proof",
        str(tmp_path / "proof.json"),
        "--audit-key-file",
        str(tmp_path / "audit"),
    ]
    assert main([*check, "--id", logged]) == 0
    assert main([*check, "--id", "0" * 64]) == 1


def test_id_on_a_file_sealed_before_format_2_exits_with_three(capsys):
    assert main(["id", str(FIXTURES / "sealed-format-1.zip")]) == 3
    assert "format 2" in capsys.readouterr().err
