import pickle
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tabcustody._reader import UnreadableFileError, read_bytes

pytestmark = pytest.mark.fuzz

# A path no run should ever create: a stream that executed code would touch it, and the reader must not.
_MARKER = Path(tempfile.mkdtemp()) / "ran"
_CODE = f"open({str(_MARKER)!r}, 'w').close()".encode()

# Globals a crafted pickle reaches for to run code; the reader must turn each into an inert shell.
_DANGEROUS = [
    b"cos\nsystem\n",
    b"c__builtin__\neval\n",
    b"c__builtin__\nexec\n",
    b"cposix\nsystem\n",
]
_OPERANDS = [
    b")",  # EMPTY_TUPLE
    b"]",  # EMPTY_LIST
    b"}",  # EMPTY_DICT
    b"N",  # NONE
    b"\x85",  # TUPLE1
    b"\x86",  # TUPLE2
    b"R",  # REDUCE
    b"b",  # BUILD
    b"\x81",  # NEWOBJ
    b"0",  # POP
    b"2",  # DUP
    b"(",  # MARK
    b"e",  # APPENDS
    b"s",  # SETITEM
    b"\x94",  # MEMOIZE
    b"K\x01",  # BININT1 1
    b"\x8c\x03abc",  # SHORT_BINUNICODE
    b"\x8e\x00\x00\x00\x00\x00\x00\x00\x00",  # BINBYTES8 length 0
]
_token = st.sampled_from([*_DANGEROUS, *_OPERANDS])
_stream = st.builds(
    lambda proto, body: proto + b"".join(body) + b".",
    st.sampled_from([b"\x80\x02", b"\x80\x04", b""]),
    st.lists(_token, max_size=60),
)


def _read_without_running(blob: bytes) -> None:
    try:
        read_bytes(blob)
    except UnreadableFileError:
        pass
    assert not _MARKER.exists()


@given(_stream)
def test_a_crafted_opcode_stream_runs_nothing_and_never_hangs(blob):
    _read_without_running(blob)


@given(st.binary(max_size=4096), st.sampled_from([0, 2, 4, 5]))
def test_a_reduce_reaching_a_dangerous_global_stays_inert(blob, protocol):
    header = (b"\x80" + bytes([protocol])) if protocol else b""
    attack = header + b"c__builtin__\nexec\n" + pickle.dumps(_CODE.decode())[-40:]
    _read_without_running(attack + blob)
