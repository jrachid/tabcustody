import pickle
import subprocess
import sys
from pathlib import Path

from tabcustody._reader import Shell, read


class _OpensAFileOnLoad:
    def __init__(self, target: Path) -> None:
        self.target = target

    def __reduce__(self):
        return (open, (str(self.target), "w"))


class _RunsCodeOnLoad:
    def __init__(self, target: Path) -> None:
        self.target = target

    def __reduce__(self):
        return (exec, (f"open({str(self.target)!r}, 'w').close()",))


def test_a_pickle_that_runs_a_command_on_load_runs_nothing(tmp_path):
    for trap in (_OpensAFileOnLoad, _RunsCodeOnLoad):
        marker = tmp_path / f"{trap.__name__}.created"
        path = tmp_path / "trap.pkl"
        path.write_bytes(pickle.dumps({"model": trap(marker)}))

        tree = read(path)

        assert not marker.exists()
        assert isinstance(tree["model"], Shell)


def test_scanning_imports_neither_torch_nor_xgboost(tmp_path):
    path = tmp_path / "foreign.pkl"
    path.write_bytes(
        b"\x80\x02}q\x00(X\x05\x00\x00\x00torchq\x01ctorch.nn.modules.linear\nLinear\nq\x02)\x81q\x03"
        b"X\x07\x00\x00\x00xgboostq\x04cxgboost.sklearn\nXGBClassifier\nq\x05)\x81q\x06u."
    )
    probe = (
        "import sys; from tabcustody._reader import read; "
        f"tree = read({str(path)!r}); "
        "loaded = [m for m in ('torch', 'xgboost', 'sklearn', 'tabpfn', 'tabicl', 'tabdpt') if m in sys.modules]; "
        "print(tree['torch'].name, tree['xgboost'].name, loaded)"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert (
        result.stdout.strip() == "torch.nn.modules.linear.Linear xgboost.sklearn.XGBClassifier []"
    )
