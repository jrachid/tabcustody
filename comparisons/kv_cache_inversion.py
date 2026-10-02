"""Rebuilds the training rows of a TabICL file saved with `kv_cache=True` and `save(save_training_data=False)`, from the file alone.

Run with `uv run python kv_cache_inversion.py [--wide]`; the output is the README's line on TabICL's cache. `--wide` adds a 20-column table and takes about an hour."""

import pickle
import sys
import tempfile
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tabicl import TabICLClassifier
from tabicl._model.learning import ICLearning

from leak import ROWS, customers, target

warnings.filterwarnings("ignore")

BATCH = 8000


def wide_customers() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(ROWS, 4))
    columns = {"age": rng.integers(20, 70, ROWS), "income": rng.normal(40000, 12000, ROWS).round(2), "debt": rng.normal(8000, 3000, ROWS).round(2)}
    columns |= {f"count_{i}": rng.poisson(3 + 2 * i, ROWS) for i in range(4)}
    columns |= {f"amount_{i}": np.exp(rng.normal(6 + i, 0.8, ROWS)).round(2) for i in range(4)}
    columns |= {f"score_{i}": (latent @ rng.normal(size=4) * 10 + 50).round(3) for i in range(5)}
    columns |= {f"rate_{i}": rng.uniform(0.01, 0.3, ROWS).round(4) for i in range(4)}
    frame = pd.DataFrame(columns)
    return frame, ((frame["debt"] / frame["income"] > 0.2) ^ (frame["score_0"] > 55)).astype(int)


def standardise(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(-1, keepdim=True)) / x.std(-1, unbiased=False, keepdim=True)


class CachedRows:
    """What the file reveals about each training row: the standardised input of the first in-context block, read back from its key-value cache."""

    def __init__(self, path: Path):
        self.model = pickle.loads(path.read_bytes())
        generator = self.model.ensemble_generator_
        self.branches = list(self.model.model_kv_cache_)
        self.columns = list(self.model.feature_names_in_)
        icl = self.model.model_.icl_predictor
        block = icl.tf_icl.blocks[0]
        weight, bias = block.attn.in_proj_weight.detach().double(), block.attn.in_proj_bias.detach().double()
        d = weight.shape[1]
        projections = torch.cat([weight[d : 2 * d], weight[2 * d :]])
        gamma, beta = block.norm1.weight.detach().double(), block.norm1.bias.detach().double()
        targets = []
        for branch in self.branches:
            kv = self.model.model_kv_cache_[branch].icl_cache.kv[0]
            members, _, rows, _ = kv.key.shape
            keys = kv.key.permute(0, 2, 1, 3).reshape(members, rows, d).double() - bias[d : 2 * d]
            values = kv.value.permute(0, 2, 1, 3).reshape(members, rows, d).double() - bias[2 * d :]
            # The first block applies no positional rotation, so its keys and values are linear in the layer-normed row.
            normed = torch.linalg.lstsq(projections, torch.cat([keys, values], -1).reshape(-1, 2 * d).T).solution.T
            targets.append(standardise((normed.reshape(members, rows, d) - beta) / gamma))
        self.targets = torch.cat(targets)
        with torch.no_grad():
            self.label_vectors = icl.y_encoder(torch.tensor([[0.0, 1.0]])).double()[0]
        self.slot_of_one = torch.tensor([shuffle[1] for branch in self.branches for shuffle in generator.class_shuffles_[branch]])
        scaler = generator.preprocessors_["none"].standard_scaler_
        self.mean, self.scale = scaler.mean_, scaler.scale_

    def represent(self, z: np.ndarray, labels: np.ndarray) -> torch.Tensor:
        """The same quantity for candidate rows given in units of the file's own scaler, each passed through the file's model as a test row."""
        captured = []
        original = ICLearning._icl_predictions_with_cache

        def capture(icl, R, icl_cache, y_train=None, use_cache=False, store_cache=True):
            captured.append(R.detach().double().clone())
            return original(icl, R, icl_cache, y_train, use_cache, store_cache)

        ICLearning._icl_predictions_with_cache = capture
        try:
            for start in range(0, len(z), BATCH):
                self.model.predict_proba(pd.DataFrame(self.mean + z[start : start + BATCH] * self.scale, columns=self.columns))
        finally:
            ICLearning._icl_predictions_with_cache = original
        per_branch = len(self.branches)
        R = torch.cat([torch.cat(captured[i : i + per_branch]) for i in range(0, len(captured), per_branch)], dim=1)
        labels = torch.tensor(labels)
        slots = torch.where(labels[None, :] == 1, self.slot_of_one[:, None], 1 - self.slot_of_one[:, None])
        return standardise(R + self.label_vectors[slots])

    def gap(self, z: np.ndarray, labels: np.ndarray, rows: np.ndarray) -> np.ndarray:
        x, t = self.represent(z, labels), self.targets[:, rows]
        return ((x - t).norm(dim=-1) / t.norm(dim=-1)).mean(0).numpy()


def grid_start(cache: CachedRows) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-4, 4, 21)
    grid = np.stack(np.meshgrid(*[axis] * len(cache.columns), indexing="ij"), -1).reshape(-1, len(cache.columns))
    rows = cache.targets.shape[1]
    best = (np.zeros((rows, grid.shape[1])), np.zeros(rows, int), np.full(rows, np.inf))
    for label in (0, 1):
        x = cache.represent(grid, np.full(len(grid), label))
        distance = sum(torch.cdist(cache.targets[m], x[m]) / cache.targets[m].norm(dim=-1, keepdim=True) for m in range(len(x))) / len(x)
        value, index = distance.min(dim=1)
        better = value.numpy() < best[2]
        best[0][better], best[1][better], best[2][better] = grid[index.numpy()][better], label, value.numpy()[better]
    return best


def sweep_start(cache: CachedRows) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis, rows, width = np.linspace(-4, 4, 17), cache.targets.shape[1], len(cache.columns)

    def sweep(z, labels, current):
        z, current = z.copy(), current.copy()
        for j in range(width):
            candidates = np.repeat(z, len(axis), axis=0)
            candidates[:, j] = np.tile(axis, rows)
            value = cache.gap(candidates, np.repeat(labels, len(axis)), np.repeat(np.arange(rows), len(axis))).reshape(rows, len(axis))
            choice = value.argmin(1)
            better = value[np.arange(rows), choice] < current
            z[better, j], current[better] = axis[choice[better]], value[np.arange(rows), choice][better]
        return z, current

    tries = [sweep(np.zeros((rows, width)), np.full(rows, label), np.full(rows, np.inf)) for label in (0, 1)]
    labels = (tries[1][1] < tries[0][1]).astype(int)
    z, current = np.where(labels[:, None] == 1, tries[1][0], tries[0][0]), np.minimum(tries[0][1], tries[1][1])
    for _ in range(3):
        z, current = sweep(z, labels, current)
    return z, labels, current


def refine(cache: CachedRows, z: np.ndarray, labels: np.ndarray, current: np.ndarray, step: float, offsets: np.ndarray, rounds: int) -> np.ndarray:
    rows, steps = len(z), np.full(len(z), step)
    for _ in range(rounds):
        candidates = (z[:, None, :] + steps[:, None, None] * offsets[None]).reshape(-1, z.shape[1])
        value = cache.gap(candidates, np.repeat(labels, len(offsets)), np.repeat(np.arange(rows), len(offsets))).reshape(rows, len(offsets))
        choice = value.argmin(1)
        moved = value[np.arange(rows), choice] < current
        z[moved] = candidates.reshape(rows, len(offsets), -1)[moved, choice[moved]]
        current[moved] = value[moved, choice[moved]]
        steps[~moved] /= 2
        if steps.max() < 1e-7:
            break
    return z


def rebuild(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Every training row and label, rebuilt from the saved file, the TabICL library and its public weights only."""
    cache = CachedRows(path)
    width = len(cache.columns)
    if width <= 3:
        z, labels, current = grid_start(cache)
        offsets = np.stack(np.meshgrid(*[[-1, 0, 1]] * width, indexing="ij"), -1).reshape(-1, width)
        z = refine(cache, z, labels, current, 0.4, offsets, rounds=60)
    else:
        z, labels, current = sweep_start(cache)
        z = refine(cache, z, labels, current, 0.5, np.vstack([np.eye(width), -np.eye(width)]), rounds=300)
    return cache.mean + z * cache.scale, labels


def measure(frame: pd.DataFrame, labels: pd.Series) -> str:
    path = Path(tempfile.mkdtemp()) / "tabicl.pkl"
    TabICLClassifier(device="cpu", kv_cache=True).fit(frame, labels).save(path, save_training_data=False)
    rows, rebuilt_labels = rebuild(path)
    truth = frame.to_numpy(float)
    error = np.abs(rows - truth) / np.where(truth == 0, 1, np.abs(truth))
    return (
        f"| TabICL {version('tabicl')}, {frame.shape[1]} columns | {(error < 0.01).all(1).sum()} / {ROWS} "
        f"| {(error < 0.01).mean():.1%} | {np.median(error):.2%} | {(rebuilt_labels == labels.to_numpy()).sum()} / {ROWS} |"
    )


print("| Model fitted with `kv_cache=True`, saved with `save_training_data=False` | Rows rebuilt within 1% | Cells within 1% | Median cell error | Labels rebuilt |")
print("|---|---|---|---|---|")
print(measure(customers, target))
if "--wide" in sys.argv:
    print(measure(*wide_customers()))
