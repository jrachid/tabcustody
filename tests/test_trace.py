import hashlib

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from tabcustody._trace import (
    _node,
    encode_row,
    inclusion_path,
    merkle_root,
    table_leaves,
    verify_inclusion,
    verify_row,
)

KEY = b"k" * 32
ROWS = 300

# RFC 6962 test inputs and the roots of their first n leaves, from certificate-transparency's merkle_tree_test.
CT_INPUTS = [
    "",
    "00",
    "10",
    "2021",
    "3031",
    "40414243",
    "5051525354555657",
    "606162636465666768696a6b6c6d6e6f",
]
CT_ROOTS = [
    "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
    "fac54203e7cc696cf0dfcb42c92a1d9dbaf70ad9e621f4bd8d98662f00e3c125",
    "aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77",
    "d37ee418976dd95753c1c73862b9398fa2a2cf9b4ff0fdfe8b30cd95209614b7",
    "4e3bbb1f7b478dcfe71fb631631519a3bca12c9aefca1612bfce4c13a86264d4",
    "76e67dadbcdf1e10e1b74ddc608abd2f98dfb16fbce75277b5232a127f2087ef",
    "ddb89be403809e325750d3d263cd78929c2942b7942a34b77e122c9594a74c8c",
    "5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328",
]


def table():
    rng = np.random.default_rng(0)
    return rng.normal(size=(ROWS, 3)), rng.integers(0, 2, ROWS)


def test_the_tree_matches_the_certificate_transparency_vectors():
    leaves = [hashlib.sha256(b"\x00" + bytes.fromhex(d)).digest() for d in CT_INPUTS]

    roots = [merkle_root(leaves[:n]).hex() for n in range(1, 9)]

    assert roots == CT_ROOTS


@given(st.integers(1, 70), st.data())
def test_every_row_of_any_table_size_has_a_proof_that_verifies(size, data):
    leaves = [hashlib.sha256(bytes([i % 256, i // 256])).digest() for i in range(size)]
    index = data.draw(st.integers(0, size - 1))

    path = inclusion_path(leaves, index)

    assert verify_inclusion(leaves[index], index, size, path, merkle_root(leaves))


def test_a_row_in_the_table_has_a_proof_the_audit_key_accepts():
    X, y = table()
    leaves = table_leaves(KEY, X, y)

    path = inclusion_path(leaves, 123)

    assert verify_row(KEY, 123, X[123], y[123], ROWS, path, merkle_root(leaves))


def test_a_proof_for_a_changed_or_moved_row_is_refused():
    X, y = table()
    leaves = table_leaves(KEY, X, y)
    root, path = merkle_root(leaves), inclusion_path(leaves, 123)
    changed = X[123].copy()
    changed[0] += 1e-12

    assert not verify_row(KEY, 123, changed, y[123], ROWS, path, root)
    assert not verify_row(KEY, 123, X[123], 1 - y[123], ROWS, path, root)
    assert not verify_row(KEY, 124, X[123], y[123], ROWS, path, root)
    assert not verify_row(b"x" * 32, 123, X[123], y[123], ROWS, path, root)


def test_an_inner_node_cannot_pass_for_a_row():
    X, y = table()
    leaves = table_leaves(KEY, X[:4], y[:4])
    root = merkle_root(leaves)
    # The second-preimage forgery: present the two children of a node as a "row" of a two-leaf tree.
    forged = np.frombuffer(leaves[0] + leaves[1], dtype=np.float64)

    assert verify_inclusion(_node(leaves[0], leaves[1]), 0, 2, [_node(leaves[2], leaves[3])], root)
    assert not verify_row(KEY, 0, forged, None, 2, [_node(leaves[2], leaves[3])], root)


def test_the_row_encoding_is_frozen():
    rows = [
        (0, np.array([1.5, -2.0, 3.25]), np.int64(1)),
        (7, np.array([np.nan, -0.0, 0.0]), None),
        (299, np.array([1, 2, 3], dtype=np.int32), 0.5),
    ]

    digests = [hashlib.sha256(encode_row(*r)).hexdigest()[:16] for r in rows]

    assert digests == ["e29ad8935f55ddbd", "f81cabb8a4020eb6", "c864360edc97c9b5"]


def test_every_nan_encodes_alike_and_negative_zero_does_not_encode_as_zero():
    quiet = np.array([np.nan])
    payload = np.array([0x7FF0000000000001], dtype=np.uint64).view(np.float64)

    assert encode_row(0, quiet) == encode_row(0, payload)
    assert encode_row(0, np.array([-0.0])) != encode_row(0, np.array([0.0]))
    assert encode_row(0, quiet.astype(">f8")) == encode_row(0, quiet)
