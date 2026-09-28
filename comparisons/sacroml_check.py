"""Runs SACRO-ML's instance-based attack, the nearest existing check, on a kNN control and on TabICL.

Run with `uv run python sacroml_check.py`; the output is the README's neighbours table."""

import logging
import tempfile
import warnings
from importlib.metadata import version

from sacroml.attacks.instance_based_attack import InstanceBasedAttack
from sacroml.attacks.target import Target
from sklearn.neighbors import KNeighborsClassifier

from leak import customers, tabicl_model, target

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

X = customers.to_numpy(float)
y = target.to_numpy()

print(f"sacroml {version('sacroml')}")
print("| Model | SACRO-ML verdict | Rows it matched |")
print("|---|---|---|")
for label, model in (
    (f"KNeighborsClassifier (scikit-learn {version('scikit-learn')})", KNeighborsClassifier().fit(X, y)),
    (f"TabICL {version('tabicl')}", tabicl_model()),
):
    attack = InstanceBasedAttack(output_dir=tempfile.mkdtemp(), write_report=False)
    attack.attack(Target(model=model, X_train=X, y_train=y, X_test=X[:50], y_test=y[:50]))
    results = vars(attack.results)
    confirmed = results.get("data_leakage_confirmed")
    verdict = "leakage confirmed" if confirmed else "not instance-based, no data leakage risk"
    print(f"| {label} | {verdict} | {results.get('n_matched', 0)} / {len(X)} |")
