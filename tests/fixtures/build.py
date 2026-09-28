"""Writes the reference model files the test suite scans, from the 300 synthetic customers of comparisons/leak.py.

Run with the comparisons environment: `~/.venvs/tabcustody-comparisons/bin/python tests/fixtures/build.py [--large]`."""

import json
import pickle
import subprocess
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
LARGE = HERE / "large"
ROWS = 300


def customers() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "age": rng.integers(20, 70, ROWS),
            "income": rng.normal(40000, 12000, ROWS).round(2),
            "debt": rng.normal(8000, 3000, ROWS).round(2),
        }
    )
    return frame, (frame["debt"] / frame["income"] > 0.2).astype(int)


def build(name: str):
    X, y = customers()
    if name == "tabicl":
        from tabicl import TabICLClassifier

        return TabICLClassifier(device="cpu").fit(X, y)
    if name == "tabpfn-v2":
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        return TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu").fit(X, y)
    if name == "tabpfn-v3.5":
        from tabpfn import TabPFNClassifier

        return TabPFNClassifier(device="cpu").fit(X, y)
    if name == "tabdpt":
        from tabdpt import TabDPTClassifier

        model = TabDPTClassifier(device="cpu")
        model.fit(X.to_numpy(float), y.to_numpy())
        return model
    if name == "kneighbors":
        from sklearn.neighbors import KNeighborsClassifier

        return KNeighborsClassifier().fit(X, y)
    if name == "svc":
        from sklearn.svm import SVC

        return SVC().fit(X, y)
    if name == "xgboost":
        import xgboost

        return xgboost.XGBClassifier(n_estimators=50, max_depth=3).fit(X, y)
    if name == "lightgbm":
        import lightgbm

        return lightgbm.LGBMClassifier(n_estimators=50, verbose=-1).fit(X, y)
    if name == "logistic-regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=1000).fit(X, y)
    if name == "random-forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    if name == "gradient-boosting":
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(random_state=0).fit(X, y)
    if name == "quantile-pipeline":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import QuantileTransformer

        return make_pipeline(QuantileTransformer(n_quantiles=100), LogisticRegression()).fit(X, y)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=300, random_state=0).fit(X, y)
    raise KeyError(name)


SMALL = {
    "tabicl": "tabicl",
    "kneighbors": "scikit-learn",
    "svc": "scikit-learn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "logistic-regression": "scikit-learn",
    "random-forest": "scikit-learn",
    "gradient-boosting": "scikit-learn",
    "mlp": "scikit-learn",
    "quantile-pipeline": "scikit-learn",
}
BIG = {"tabpfn-v2": "tabpfn", "tabpfn-v3.5": "tabpfn", "tabdpt": "tabdpt"}


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        name = sys.argv[2]
        target = (LARGE if name in BIG else HERE) / f"{name}.pkl"
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(pickle.dumps(build(name)))
        sys.exit()
    wanted = {**SMALL, **(BIG if "--large" in sys.argv else {})}
    # One process per model: on macOS, fitting XGBoost and a PyTorch model in the same process segfaults.
    for name in wanted:
        subprocess.run([sys.executable, __file__, "--one", name], check=True)
        print(f"wrote {name}")
    manifest = {
        name: {"library": lib, "version": version(lib), "numpy": version("numpy")}
        for name, lib in SMALL.items()
    }
    (HERE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
