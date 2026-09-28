from pathlib import Path

import joblib
import numpy as np
import pytest

from tabcustody import scan_file

FIXTURES = Path(__file__).parent / "fixtures"


class _OpensAFileOnLoad:
    def __init__(self, target: Path) -> None:
        self.target = target

    def __reduce__(self):
        return (open, (str(self.target), "w"))


@pytest.mark.parametrize("name", ["kneighbors.joblib", "kneighbors.zlib.joblib"])
def test_a_joblib_file_is_scanned_like_a_pickle(name):
    from_joblib = scan_file(FIXTURES / name)
    from_pickle = scan_file(FIXTURES / "kneighbors.pkl")

    def tables(report):
        return [(f.path, f.shape, f.companion) for f in report.findings if f.companion]

    assert from_joblib.error is None
    assert tables(from_joblib) == tables(from_pickle)


def test_joblib_reports_the_copy_it_writes_where_pickle_shares_one_array():
    copies = [f.path for f in scan_file(FIXTURES / "kneighbors.joblib").findings if not f.companion]

    assert copies == ["_tree<state>[0]"]


@pytest.mark.parametrize("compress", [0, 3, ("gzip", 3), ("bz2", 3), ("xz", 3)])
def test_compressed_joblib_files_are_read(tmp_path, compress):
    rows = np.random.default_rng(1).normal(size=(50, 4))
    path = tmp_path / "model.joblib"
    joblib.dump({"X_train": rows, "y_train": np.arange(50)}, path, compress=compress)

    report = scan_file(path)

    assert [f.path for f in report.findings] == ["['X_train']"]


def test_an_object_array_inside_a_joblib_file_is_read_without_running_it(tmp_path):
    marker = tmp_path / "created"
    trapped = np.empty(1, dtype=object)
    trapped[0] = _OpensAFileOnLoad(marker)
    path = tmp_path / "trap.joblib"
    joblib.dump({"objects": trapped}, path)

    report = scan_file(path)

    assert report.error is None
    assert not marker.exists()


def test_a_tabpfn_fit_archive_is_scanned_member_by_member():
    report = scan_file(FIXTURES / "tabpfn-v2.tabpfn_fit")

    paths = [finding.path for finding in report.findings]

    assert report.format == "zip"
    assert "executor_state.joblib:ensemble_members[0].X_train" in paths
    assert all(path.split(":")[0].endswith(".joblib") for path in paths)
    assert all(
        "executor_state.joblib:" in reason
        for finding in report.findings
        for reason in finding.evidence
        if "1-D array" in reason
    )


def test_a_file_that_is_not_a_model_is_reported_as_unreadable(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not a pickle")

    report = scan_file(path)

    assert report.error is not None
    assert report.findings == ()
