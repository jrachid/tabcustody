"""Writes a model object that keeps its training rows as a pandas DataFrame and its target as a Series.

Run once per pandas major version, since pandas lays out pickled frames differently across versions:
`uv run --isolated --with "pandas<3" python tests/fixtures/build_frames.py` and the same with `pandas>=3`."""

import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from customers import COLUMNS, customers


class FrameModel:
    def __init__(self) -> None:
        rows, target = customers()
        self.X_train = pd.DataFrame(rows, columns=list(COLUMNS)).astype({"age": int})
        self.y_train = pd.Series(target, name="late")


major = pd.__version__.split(".")[0]
(Path(__file__).parent / f"frame-pandas{major}.pkl").write_bytes(pickle.dumps(FrameModel()))
print(f"wrote frame-pandas{major}.pkl with pandas {pd.__version__}")
