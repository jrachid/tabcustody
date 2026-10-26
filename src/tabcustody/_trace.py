"""Commits to a table's rows with a Merkle tree in the form of RFC 9162, whose leaves are keyed hashes of each row and its target, so a row's inclusion can be proved without showing the others."""

from __future__ import annotations

import hashlib
import hmac
import struct
from typing import Any

import numpy as np

_LEAF = b"\x00"
_NODE = b"\x01"
_ID = b"tabcustody-id/1\x00"


def _array(values: np.ndarray) -> bytes:
    if values.dtype.kind not in "biuf":
        raise TypeError(f"only numeric rows can be committed, not {values.dtype}")
    values = np.ascontiguousarray(values, dtype=values.dtype.newbyteorder("<"))
    if values.dtype.kind == "f":
        values = values.copy()
        values[np.isnan(values)] = np.array(np.nan, dtype=values.dtype)
    dtype = values.dtype.str.encode()
    return struct.pack(">B", len(dtype)) + dtype + struct.pack(">I", values.size) + values.tobytes()


def encode_row(position: int, row: np.ndarray, target: Any = None) -> bytes:
    """Encodes a row, its position and its target the one way the commitment reads them: every NaN becomes one bit pattern, -0.0 stays distinct from 0.0."""
    head = struct.pack(">Q", position) + _array(np.asarray(row).ravel())
    if target is None:
        return head + b"\x00"
    return head + b"\x01" + _array(np.asarray(target).ravel())


def leaf_hash(key: bytes, encoded: bytes) -> bytes:
    return hmac.new(key, _LEAF + encoded, hashlib.sha256).digest()


def _node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(_NODE + left + right).digest()


def _split(size: int) -> int:
    """Returns the largest power of two below `size`, where RFC 9162 cuts a tree in two."""
    return 1 << ((size - 1).bit_length() - 1)


def table_leaves(key: bytes, table: np.ndarray, target: np.ndarray | None = None) -> list[bytes]:
    return [
        leaf_hash(key, encode_row(i, row, None if target is None else target[i]))
        for i, row in enumerate(table)
    ]


def _root(leaves: list[bytes], lo: int, hi: int) -> bytes:
    if hi - lo == 1:
        return leaves[lo]
    middle = lo + _split(hi - lo)
    return _node(_root(leaves, lo, middle), _root(leaves, middle, hi))


def merkle_root(leaves: list[bytes]) -> bytes:
    if not leaves:
        return hashlib.sha256(b"").digest()
    return _root(leaves, 0, len(leaves))


def inclusion_path(leaves: list[bytes], index: int) -> list[bytes]:
    """Returns the sibling hashes from leaf `index` up to the root, in the order RFC 9162 verifies them."""
    if not 0 <= index < len(leaves):
        raise IndexError(index)
    path: list[bytes] = []
    lo, hi = 0, len(leaves)
    while hi - lo > 1:
        middle = lo + _split(hi - lo)
        if index < middle:
            path.append(_root(leaves, middle, hi))
            hi = middle
        else:
            path.append(_root(leaves, lo, middle))
            lo = middle
    return path[::-1]


def verify_inclusion(leaf: bytes, index: int, size: int, path: list[bytes], root: bytes) -> bool:
    """Tells whether `leaf` sits at `index` of a tree of `size` leaves with this root, by the algorithm of RFC 9162, section 2.1.3.2."""
    if not 0 <= index < size:
        return False
    fn, sn, result = index, size - 1, leaf
    for sibling in path:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            result = _node(sibling, result)
            while not (fn & 1 or fn == 0):
                fn >>= 1
                sn >>= 1
        else:
            result = _node(result, sibling)
        fn >>= 1
        sn >>= 1
    return sn == 0 and hmac.compare_digest(result, root)


def verify_row(
    key: bytes,
    position: int,
    row: np.ndarray,
    target: Any,
    size: int,
    path: list[bytes],
    root: bytes,
) -> bool:
    """Tells whether this row, with this target, was at `position` of the committed table; needs the audit key and none of the other rows."""
    leaf = leaf_hash(key, encode_row(position, row, target))
    return verify_inclusion(leaf, position, size, path, root)


def identifier(skeleton_sha256: bytes, roots: list[bytes]) -> str:
    return hashlib.sha256(_ID + skeleton_sha256 + b"".join(roots)).hexdigest()


def _array_from(exact: dict[str, str]) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(exact["hex"]), dtype=np.dtype(exact["dtype"]))


def verify_proof(proof: dict[str, Any], audit_key: bytes, file_id: str) -> bool:
    """Tells whether the proof shows its row inside the file whose logged identifier is `file_id`; needs the audit key, neither the sealing key nor the file. A malformed proof is refused, never raised on."""
    try:
        skeleton = bytes.fromhex(proof["skeleton_sha256"])
        roots = [bytes.fromhex(root) for root in proof["roots"]]
        if not hmac.compare_digest(identifier(skeleton, roots), file_id):
            return False
        table = int(proof["table"])
        if table < 0:
            return False
        target = None if proof["target"] is None else _array_from(proof["target"])
        return verify_row(
            audit_key,
            int(proof["position"]),
            _array_from(proof["row"]),
            target,
            int(proof["rows"]),
            [bytes.fromhex(sibling) for sibling in proof["path"]],
            roots[table],
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return False
