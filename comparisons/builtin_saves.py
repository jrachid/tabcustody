"""Measures what each library's own save function keeps: TabPFN's `.tabpfn_fit` archive in both fit modes and TabICL's `save()` with and without its option.

Run with `uv run python builtin_saves.py`; the output is the README's built-in saves table."""

import pickle
import tempfile
import warnings
import zipfile
from importlib.metadata import version
from pathlib import Path

import numpy as np

from leak import customers, found, target

warnings.filterwarnings("ignore")

workdir = Path(tempfile.mkdtemp())


def tabpfn_fit_archive(v2: bool, fit_mode: str) -> tuple[str, str, str]:
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    options = {"device": "cpu", "fit_mode": fit_mode}
    model = (
        TabPFNClassifier.create_default_for_version(ModelVersion.V2, **options)
        if v2
        else TabPFNClassifier(**options)
    ).fit(customers, target)
    reference = model.predict_proba(customers)
    path = workdir / f"tabpfn-{v2}-{fit_mode}.tabpfn_fit"
    model.save_fit_state(path)
    with zipfile.ZipFile(path) as archive:
        blob = b"".join(archive.read(name) for name in archive.namelist())
    reloaded = TabPFNClassifier.load_from_fit_state(path, device="cpu")
    gap = np.max(np.abs(reloaded.predict_proba(customers) - reference))
    weights = "v2" if v2 else "v3.5, the default weights,"
    label = f"TabPFN {weights} `save_fit_state` (tabpfn {version('tabpfn')}), `fit_mode=\"{fit_mode}\"`"
    return label, found(blob), f"{gap:.0e}"


def tabicl_save(kv_cache: bool, keep_data: bool) -> tuple[str, str, str]:
    from tabicl import TabICLClassifier

    model = TabICLClassifier(device="cpu", kv_cache=kv_cache).fit(customers, target)
    reference = model.predict_proba(customers)
    path = workdir / f"tabicl-{kv_cache}-{keep_data}.pkl"
    model.save(path, save_training_data=keep_data)
    reloaded = pickle.loads(path.read_bytes())
    gap = np.max(np.abs(reloaded.predict_proba(customers) - reference))
    label = f"TabICL {version('tabicl')} `save(save_training_data={keep_data})`, `kv_cache={kv_cache}`"
    return label, found(path.read_bytes()), f"{gap:.0e}"


print("| Save | Incomes found verbatim | Largest prediction gap after reload |")
print("|---|---|---|")
for row in (
    tabpfn_fit_archive(v2=True, fit_mode="fit_preprocessors"),
    tabpfn_fit_archive(v2=True, fit_mode="fit_with_cache"),
    tabpfn_fit_archive(v2=False, fit_mode="fit_preprocessors"),
    tabpfn_fit_archive(v2=False, fit_mode="fit_with_cache"),
    tabicl_save(kv_cache=False, keep_data=True),
    tabicl_save(kv_cache=True, keep_data=False),
):
    print("| " + " | ".join(row) + " |")
