"""The 300 synthetic customers every reference file is fitted on, rebuilt with NumPy alone so tests can compare against them."""

import numpy as np

ROWS = 300
COLUMNS = ("age", "income", "debt")


def customers() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    age = rng.integers(20, 70, ROWS)
    income = rng.normal(40000, 12000, ROWS).round(2)
    debt = rng.normal(8000, 3000, ROWS).round(2)
    return np.column_stack([age, income, debt]).astype(float), (debt / income > 0.2).astype(int)
