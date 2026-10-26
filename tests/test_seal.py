import numpy as np

from tabcustody._detect import find_tables
from tabcustody._reader import read_bytes
from tabcustody._seal import lift, restore, sealable_objects

ROWS = 300


class Tree:
    __slots__ = ("data",)


class Estimator:
    def __init__(self):
        self._fit_X = np.arange(ROWS * 3, dtype=float).reshape(ROWS, 3)
        self._y = np.arange(ROWS)
        self.n_features_in_ = 3
        # The fitted tree shares the same array object, as a real kNN's KD tree does.
        self._tree = Tree()
        self._tree.data = self._fit_X


class Pipeline:
    def __init__(self):
        self.steps = [("scale", object()), ("knn", Estimator())]


class Unsupervised:
    def __init__(self):
        self._fit_X = np.arange(ROWS * 3, dtype=float).reshape(ROWS, 3)
        self.n_features_in_ = 3


class WithExtremes:
    def __init__(self):
        self.X_min_ = np.zeros((1, 3))
        self.X_max_ = np.ones((1, 3))
        self.X_transformed_ = np.arange(ROWS * 3, dtype=float).reshape(ROWS, 3)
        self._fit_X = self.X_transformed_
        self._y = np.arange(ROWS)


def test_the_training_table_and_its_target_are_found_in_a_live_model():
    shapes = sorted((a.shape for a in sealable_objects(Estimator())), key=len)

    assert (ROWS,) in shapes and (ROWS, 3) in shapes


def test_a_target_less_model_is_sealed_by_its_declared_width():
    model = Unsupervised()

    model_bytes, lifted = lift(model)

    assert [a.shape for a in lifted] == [(ROWS, 3)]
    assert find_tables(read_bytes(model_bytes)) == []


def test_column_extremes_and_a_transformed_copy_are_lifted_too():
    model = WithExtremes()

    model_bytes, lifted = lift(model)

    assert {a.shape for a in lifted} == {(1, 3), (ROWS, 3), (ROWS,)}
    assert find_tables(read_bytes(model_bytes)) == []


def test_the_sealed_bytes_carry_no_training_table_even_the_shared_copy():
    model_bytes, lifted = lift(Estimator())

    assert [a.shape for a in lifted] == [(ROWS, 3), (ROWS,)]
    assert find_tables(read_bytes(model_bytes)) == []


def test_restoring_gives_the_exact_arrays_back_and_keeps_them_shared():
    model = Estimator()

    model_bytes, lifted = lift(model)
    back = restore(model_bytes, lifted)

    assert np.array_equal(back._fit_X, model._fit_X)
    assert np.array_equal(back._y, model._y)
    assert back._tree.data is back._fit_X


def test_the_live_model_is_not_touched_by_sealing():
    model = Estimator()
    before = model._fit_X.copy()

    lift(model)

    assert np.array_equal(model._fit_X, before)
    assert isinstance(model._fit_X, np.ndarray)


def test_a_table_nested_in_a_pipeline_is_sealed_and_restored():
    model = Pipeline()

    model_bytes, lifted = lift(model)
    back = restore(model_bytes, lifted)

    assert find_tables(read_bytes(model_bytes)) == []
    assert np.array_equal(back.steps[1][1]._fit_X, model.steps[1][1]._fit_X)


def test_a_sealed_file_round_trips_and_a_scan_finds_nothing(tmp_path):
    from tabcustody import scan_file, seal, unseal

    model = Estimator()
    path = tmp_path / "model.sealed"
    key = b"\x01" * 32

    seal(model, path, key)
    report = scan_file(path)
    back = unseal(path, key)

    assert not report.carries_training_data
    assert report.error is None
    assert np.array_equal(back._fit_X, model._fit_X)
    assert back._tree.data is back._fit_X


def test_the_sealed_file_holds_neither_the_key_nor_a_value_of_the_table(tmp_path):
    from tabcustody import seal

    model = Estimator()
    path = tmp_path / "model.sealed"
    key = b"secret-key-of-32-bytes-exactly!!"

    seal(model, path, key)
    blob = path.read_bytes()

    assert key not in blob
    assert model._fit_X[0].tobytes() not in blob


def test_a_wrong_key_is_refused_before_anything_is_rebuilt(tmp_path):
    from tabcustody import CannotUnseal, seal, unseal

    path = tmp_path / "model.sealed"
    seal(Estimator(), path, b"\x01" * 32)

    try:
        unseal(path, b"\x02" * 32)
        raise AssertionError("a wrong key should be refused")
    except CannotUnseal as error:
        assert "key" in str(error)


def test_a_changed_file_is_refused(tmp_path):
    import json
    import zipfile

    from tabcustody import CannotUnseal, seal, unseal

    path = tmp_path / "model.sealed"
    key = b"\x01" * 32
    seal(Estimator(), path, key)

    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        model_bytes = archive.read("model.pkl")
    flipped = bytearray(__import__("base64").b64decode(manifest["objects"][0]["ciphertext"]))
    flipped[0] ^= 1
    manifest["objects"][0]["ciphertext"] = __import__("base64").b64encode(bytes(flipped)).decode()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model.pkl", model_bytes)
        archive.writestr("manifest.json", json.dumps(manifest))

    try:
        unseal(path, key)
        raise AssertionError("a changed file should be refused")
    except CannotUnseal as error:
        assert "changed" in str(error)


def test_the_manifest_does_not_confirm_a_known_table(tmp_path):
    import hashlib
    import pickle
    import zipfile

    from tabcustody import seal

    model = Estimator()
    path = tmp_path / "model.sealed"
    seal(model, path, b"\x01" * 32)

    with zipfile.ZipFile(path) as archive:
        manifest = archive.read("manifest.json").decode()
    plain = hashlib.sha256(pickle.dumps(model._fit_X, protocol=pickle.DEFAULT_PROTOCOL)).hexdigest()

    assert plain not in manifest


def test_scanning_imports_no_cryptography(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "x.pkl"
    path.write_bytes(__import__("pickle").dumps({"a": 1}))
    probe = (
        f"import sys; from tabcustody import scan_file; scan_file({str(path)!r}); "
        "print('cryptography' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

    assert out.stdout.strip() == "False"


def test_a_replaced_model_skeleton_is_refused_and_not_run(tmp_path):
    import json
    import pickle
    import zipfile

    from tabcustody import CannotUnseal, seal, unseal

    path = tmp_path / "model.sealed"
    key = b"\x01" * 32
    seal(Estimator(), path, key)
    ran = tmp_path / "ran"

    class Bomb:
        def __reduce__(self):
            return (open, (str(ran), "w"))

    with zipfile.ZipFile(path) as archive:
        manifest = archive.read("manifest.json")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model.pkl", pickle.dumps({"x": Bomb()}))
        archive.writestr("manifest.json", manifest)

    try:
        unseal(path, key)
        raise AssertionError("a replaced model skeleton should be refused")
    except CannotUnseal as error:
        assert "changed" in str(error)
    assert not ran.exists()
    assert json.loads(manifest)  # the manifest itself was untouched, yet unseal still refused


def fake_tabpfn(monkeypatch):
    """Registers stand-ins for TabPFN's network and cache classes under its module names, so they pickle by reference."""
    import sys
    import types

    names = (
        "tabpfn",
        "tabpfn.architectures",
        "tabpfn.architectures.v3",
        "tabpfn.architectures.kv_cache",
    )
    modules = {name: types.ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    class Network:
        def __init__(self):
            self.weights = np.arange(64.0)

    class KVCacheEntry:
        def __init__(self):
            self.key = np.arange(ROWS * 3, dtype=float).reshape(1, ROWS, 3)

    class Cache:
        def __init__(self):
            self.kv = {0: KVCacheEntry()}
            self.train_shape = (1, ROWS)

    for cls, module in ((Network, "v3"), (Cache, "v3"), (KVCacheEntry, "kv_cache")):
        cls.__module__ = f"tabpfn.architectures.{module}"
        cls.__qualname__ = cls.__name__
        setattr(modules[cls.__module__], cls.__name__, cls)
    return Network, Cache


def test_tabpfn_network_weights_stay_in_the_clear_skeleton(monkeypatch):
    Network, _ = fake_tabpfn(monkeypatch)
    model = Estimator()
    model.models_ = [Network()]
    model.networks = {"main": model.models_[0]}

    model_bytes, lifted = lift(model)

    assert [a.shape for a in lifted] == [(ROWS, 3), (ROWS,)]
    assert np.arange(64.0).tobytes() in model_bytes


def test_a_tabpfn_cache_is_still_lifted_whole(monkeypatch):
    Network, Cache = fake_tabpfn(monkeypatch)
    model = Estimator()
    model.models_ = [Network()]
    model.caches = [Cache()]

    _, lifted = lift(model)

    assert any(obj is model.caches for obj in lifted)
    assert not any(obj is model.models_ for obj in lifted)


def test_a_ring_of_keys_opens_the_file_its_one_right_key_opens(tmp_path):
    from tabcustody import seal, unseal

    path = tmp_path / "model.sealed"
    seal(Estimator(), path, b"\x01" * 32)

    back = unseal(path, [b"\x02" * 32, b"\x01" * 32, b"\x03" * 32])

    assert np.array_equal(back._fit_X, Estimator()._fit_X)


def test_a_ring_without_the_right_key_is_refused(tmp_path):
    import pytest

    from tabcustody import CannotUnseal, seal, unseal
    from tabcustody._seal import opens

    path = tmp_path / "model.sealed"
    seal(Estimator(), path, b"\x01" * 32)

    with pytest.raises(CannotUnseal, match="no key opens"):
        unseal(path, [b"\x02" * 32, b"\x03" * 32])
    assert not opens(path, [b"\x02" * 32]) and opens(path, [b"\x02" * 32, b"\x01" * 32])
