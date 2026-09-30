"""Measures what each library's own save function keeps: TabPFN's `.tabpfn_fit` archive and TabICL's `save()` with and without its option.

Run with `uv run python builtin_saves.py`; the output is the README's built-in saves table."""

import pickle
import tempfile
import warnings
import zipfile
from importlib.metadata import version
from pathlib import Path

import numpy as np

from leak import customers, found, tabpfn_v2_model, target

warnings.filterwarnings("ignore")

workdir = Path(tempfile.mkdtemp())


def tabpfn_fit_archive() -> tuple[str, str, str]:
    path = workdir / "model.tabpfn_fit"
    tabpfn_v2_model().save_fit_state(path)
    with zipfile.ZipFile(path) as archive:
        blob = b"".join(archive.read(name) for name in archive.namelist())
    return f"TabPFN v2 `save_fit_state` (tabpfn {version('tabpfn')})", found(blob), "n/a"


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
for row in (tabpfn_fit_archive(), tabicl_save(kv_cache=False, keep_data=True), tabicl_save(kv_cache=True, keep_data=False)):
    print("| " + " | ".join(row) + " |")
