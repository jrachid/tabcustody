import sys
from pathlib import Path

import numpy as np
import pytest

from tabcustody._detect import find_tables
from tabcustody._reader import read
from tabcustody._reverse import reverse

FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES))
from customers import customers

ORIGINAL, _ = customers()


def node(tree, path):
    for part in path.replace("[", ".[").split("."):
        if not part:
            continue
        if part.startswith("["):
            key = part[1:-1]
            if key.isdigit():
                tree = tree[int(key)]
            else:
                tree = (tree if isinstance(tree, dict) else tree.entries)[key.strip("'")]
        else:
            tree = tree.attributes[part]
    return tree


def rebuilt(tree, finding):
    return reverse(np.asarray(node(tree, finding.path)), node(tree, finding.reversible_by))


@pytest.mark.parametrize("scaler", ["standard", "minmax", "robust"])
def test_a_scaled_pipeline_table_is_rebuilt_from_the_scaler_step(scaler):
    tree = read(FIXTURES / f"{scaler}-kneighbors.pkl")
    found = {finding.path: finding for finding in find_tables(tree)}

    table = found["steps[1][1]._fit_X"]

    assert table.reversible_by == "steps[0][1]"
    np.testing.assert_allclose(rebuilt(tree, table), ORIGINAL, rtol=0, atol=1e-6)


@pytest.mark.integration
def test_a_standardised_table_is_rebuilt_from_the_scaler_saved_beside_it():
    path = FIXTURES / "large" / "tabdpt.pkl"
    if not path.exists():
        pytest.skip(f"{path} is built by `tests/fixtures/build.py --large`")
    tree = read(path)
    table = next(finding for finding in find_tables(tree) if finding.path == "X_train")

    assert table.reversible_by == "scaler"
    np.testing.assert_allclose(rebuilt(tree, table), ORIGINAL, rtol=0, atol=1e-6)


@pytest.mark.parametrize("name", ["kneighbors", "tabicl"])
def test_a_table_stored_as_is_is_not_said_to_be_reversible(name):
    tree = read(FIXTURES / f"{name}.pkl")

    primary = [finding for finding in find_tables(tree) if finding.companion is not None]

    assert primary
    assert all(finding.reversible_by is None for finding in primary)


def test_a_scaler_nested_inside_another_transformer_is_not_credited():
    tree = read(FIXTURES / "tabicl.pkl")
    power = node(tree, "ensemble_generator_.preprocessors_['power']")

    assert power.attributes["normalizer_"].attributes["_scaler"].name.endswith("StandardScaler")
    assert all(finding.reversible_by is None for finding in find_tables(tree))
