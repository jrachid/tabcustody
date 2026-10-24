import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tabcustody._reader import Shell, UnreadableFileError, read, read_bytes

# Keep byte strings below the size past which a torch-storage blob is skipped (see P3).
_scalars = st.none() | st.booleans() | st.integers(-(2**64), 2**64) | st.floats(allow_nan=False)
_leaves = _scalars | st.text(max_size=20) | st.binary(max_size=256)
_data = st.recursive(
    _leaves,
    lambda children: (
        st.lists(children, max_size=6)
        | st.lists(children, max_size=6).map(tuple)
        | st.dictionaries(st.integers() | st.text(max_size=20), children, max_size=6)
        | st.frozensets(_leaves, max_size=6)
    ),
    max_leaves=20,
)


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


@settings(max_examples=400)
@given(_data, st.integers(min_value=2, max_value=pickle.HIGHEST_PROTOCOL))
def test_plain_data_reads_exactly_as_pickle_would(obj, protocol):
    blob = pickle.dumps(obj, protocol=protocol)

    assert read_bytes(blob) == pickle.loads(blob)


@settings(max_examples=300)
@given(st.binary(max_size=2048))
def test_random_bytes_give_a_tree_or_an_unreadable_error_never_a_crash(blob):
    try:
        read_bytes(blob)
    except UnreadableFileError:
        pass


@settings(max_examples=100)
@given(
    st.lists(st.integers(-(2**63), 2**63 - 1), min_size=1, max_size=50),
    st.sampled_from(["float64", "float32", "int64", "int16", "uint8"]),
)
def test_numpy_arrays_survive_whatever_their_shape_and_dtype(values, dtype):
    array = np.array(values, dtype="int64").astype(dtype)

    assert np.array_equal(read_bytes(pickle.dumps(array)), array)


def test_a_global_to_a_dangerous_callable_becomes_a_shell_and_runs_nothing(tmp_path):
    marker = tmp_path / "ran"
    # GLOBAL exec (as a protocol-2 __builtin__ name), its code argument, then REDUCE: an attack the reader must not run.
    blob = (
        b"c__builtin__\nexec\n"
        + pickle.dumps(f"open({str(marker)!r}, 'w').close()")[:-1]
        + b"\x85R."
    )

    tree = read_bytes(blob)

    assert not marker.exists()
    assert isinstance(tree, Shell)


def test_a_self_referential_structure_is_read_without_endless_recursion():
    loop: list = []
    loop.append(loop)

    tree = read_bytes(pickle.dumps(loop))

    assert tree[0] is tree
