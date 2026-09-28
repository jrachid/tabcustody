"""Measures, from a pickled model alone, how many of its 300 training incomes are readable or rebuildable.

Run with `uv run python leak.py [xgboost tabicl tabpfn-v2 tabpfn tabdpt]`; the output is the README table."""

import pickle
import struct
import sys
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROWS = 300
rng = np.random.default_rng(0)
customers = pd.DataFrame(
    {
        "age": rng.integers(20, 70, ROWS),
        "income": rng.normal(40000, 12000, ROWS).round(2),
        "debt": rng.normal(8000, 3000, ROWS).round(2),
    }
)
target = (customers["debt"] / customers["income"] > 0.2).astype(int)
incomes = customers["income"].to_numpy()


def verbatim(blob: bytes) -> int:
    return sum(blob.count(struct.pack("<d", v)) + blob.count(struct.pack("<f", v)) > 0 for v in incomes)


def rebuilt(restored) -> int | None:
    """Rebuilds incomes from the unpickled object only, never from the original frame."""
    for attr in ("X_train", "X_", "_X_train"):
        stored = getattr(restored, attr, None)
        if stored is None:
            continue
        stored = np.asarray(stored, dtype=float)
        scaler = getattr(restored, "scaler", None)
        if scaler is not None and hasattr(scaler, "inverse_transform"):
            stored = scaler.inverse_transform(stored)
        return int(np.isclose(stored[:, 1], incomes, atol=0.005).sum())
    return None


def report(name: str, model) -> None:
    blob = pickle.dumps(model)
    restored = pickle.loads(blob)
    found = verbatim(blob)
    recovered = rebuilt(restored)
    recovered = found if recovered is None else max(found, recovered)
    print(f"| {name} | {len(blob) / 1e6:.1f} MB | {found} / {ROWS} | {recovered} / {ROWS} |")


def xgboost_model():
    import xgboost

    return xgboost.XGBClassifier(n_estimators=50, max_depth=3).fit(customers, target)


def tabicl_model():
    from tabicl import TabICLClassifier

    return TabICLClassifier(device="cpu").fit(customers, target)


def tabpfn_v2_model():
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    return TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu").fit(customers, target)


def tabdpt_model():
    from tabdpt import TabDPTClassifier

    model = TabDPTClassifier(device="cpu")
    model.fit(customers.to_numpy(float), target.to_numpy())
    return model


def tabpfn_latest_model():
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(device="cpu").fit(customers, target)


MODELS = {
    "xgboost": (f"XGBoost {version('xgboost')} (control)", xgboost_model),
    "tabicl": (f"TabICL {version('tabicl')}", tabicl_model),
    "tabpfn-v2": (f"TabPFN v2 (tabpfn {version('tabpfn')})", tabpfn_v2_model),
    "tabpfn": (f"TabPFN v3.5, the default weights (tabpfn {version('tabpfn')})", tabpfn_latest_model),
    "tabdpt": (f"TabDPT {version('tabdpt')}", tabdpt_model),
}

if __name__ == "__main__":
    chosen = sys.argv[1:] or list(MODELS)
    print("versions:", ", ".join(f"{p} {version(p)}" for p in ("xgboost", "tabicl", "tabpfn", "tabdpt", "torch")))
    print("| Model | Saved file | Incomes found verbatim | Incomes rebuilt from the file alone |")
    print("|---|---|---|---|")
    for key in chosen:
        label, build = MODELS[key]
        report(label, build())
