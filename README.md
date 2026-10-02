# tabcustody

> With in-context tabular models, the model file **is** your training data. tabcustody proves it on your files, then lets you deploy, erase and trace without exposing that data.

**Status: pre-alpha.** Version 0.1 ships the scanner: it tells whether a model file carries its training data, without running the file. Separating, erasing and tracing come next.

## Quick start

```bash
pip install tabcustody
tabcustody scan model.pkl other.joblib fitted.tabpfn_fit
```

The command reads pickle files, joblib files (plain or compressed with zlib, gzip, bz2 or xz) and zip archives of them such as TabPFN's `.tabpfn_fit`. It exits with 1 when a file carries training data, 2 when a file cannot be read, 0 otherwise, so it can gate a CI job; `--json` prints one report per file.

From Python, with scikit-learn installed:

```python
import pickle

from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier

from tabcustody import scan_file

X, y = make_classification(n_samples=200, n_features=4, random_state=0)
with open("model.pkl", "wb") as file:
    pickle.dump(KNeighborsClassifier().fit(X, y), file)

report = scan_file("model.pkl")
print(report.to_text())
assert report.carries_training_data
```

```
model.pkl (pickle)
  training table  _fit_X  200 rows x 4 columns
    why: the 1-D array _y has one entry per row
    why: _fit_X is a name libraries use for training rows
    fix: An instance-based model keeps training rows by design: keep this file as confidential as the data it was fitted on.
```

On a TabPFN archive the report names the rows of every ensemble member, and on a TabICL file saved with `save_training_data=False` it reports the key-value cache the rows can be rebuilt from:

```
fitted.tabpfn_fit (zip)
  training table  executor_state.joblib:ensemble_members[*].X_train  300 rows x 10 or 4 columns  (8 alike)
    why: the 1-D array executor_state.joblib:ensemble_members[*].y_train has one entry per row
    why: X_train is a name libraries use for training rows
    fix: TabPFN keeps the rows in its default fit mode. Releases that include PriorLabs/TabPFN pull request 1323 drop them in fit_mode='fit_with_cache', but keep a key-value cache derived from them and the smallest and largest value of each column: keep this file as confidential as the data it was fitted on.

tabicl-cache.pkl (pickle)
  key-value cache  model_kv_cache_  300 rows x 3 columns
    why: TabICL's in-context cache keeps one key and one value per training row
    fix: TabICL's key-value cache lets the training rows be rebuilt with the library and its public weights: keep this file as confidential as the data it was fitted on.
```

The scanner never unpickles the file the usual way: only NumPy arrays and plain containers are rebuilt, every other class becomes an inert placeholder, and no library the file names is imported. The report gives paths, shapes and evidence, never a value of the table.

## The problem, measured

A classic model such as XGBoost learns rules from its training data, then forgets the data: the saved file holds split thresholds, nothing else. The new tabular foundation models — TabPFN, TabICL, TabDPT — work differently. They predict by reading the training table again at every prediction, the way a language model reads its prompt. So a fitted model keeps the table, and saving the model saves the table.

The same 300 synthetic customers, each model fitted then saved with `pickle`, and only the saved file inspected afterwards:

| Model | Saved file | Incomes found verbatim | Incomes rebuilt from the file alone |
|---|---|---|---|
| XGBoost 3.4.1 (control) | 0.1 MB | 0 / 300 (chance: 0) | 0 / 300 |
| TabICL 2.2.0 | < 0.1 MB | 300 / 300 (chance: 0) | 300 / 300 |
| TabPFN v2 (tabpfn 9.0.0) | 29.2 MB | 300 / 300 (chance: 2) | 300 / 300 |
| TabPFN v3.5, the default weights (tabpfn 9.0.0) | 876.3 MB | 300 / 300 (chance: 21) | 300 / 300 |
| TabDPT 1.3.1 | 252.5 MB | 11 / 300 (chance: 11) | 300 / 300 |

A value is found verbatim when its 8-byte or 4-byte encoding appears in the file. A large file of model weights contains some of those byte patterns by coincidence, so each count comes with the number that 300 values absent from the table reach in the same file.

Measured on 30 September 2026 by [`comparisons/leak.py`](comparisons/leak.py), which anyone can rerun with the pinned versions of [`comparisons/pyproject.toml`](comparisons/pyproject.toml).

TabDPT is the instructive row. It stores the table standardised — an income of 38,693.02 becomes -0.058 — so searching the file for known values finds no more than chance does, and a manual check concludes the file is clean. But the scaler that standardised the table is saved in the same file, and calling it in reverse rebuilds every income to the cent.

Nothing here is a bug in these libraries: keeping the table is how in-context learning works. The risk is in the habits around model files. Teams have treated them as harmless build artefacts for a decade — committed to Git, copied to buckets, sent to vendors — and with these models each of those copies is a copy of the customer table.

## What the libraries' own saves keep

The table above uses `pickle`. Each library's own save function tells the same story by default, and one of them offers a way out:

| Save | Incomes found verbatim | Largest prediction gap after reload |
|---|---|---|
| TabPFN v2 `save_fit_state` (tabpfn 9.0.0) | 300 / 300 (chance: 0) | n/a |
| TabICL 2.2.0 `save(save_training_data=True)`, `kv_cache=False` | 300 / 300 (chance: 0) | 0 |
| TabICL 2.2.0 `save(save_training_data=False)`, `kv_cache=True` | 4 / 300 (chance: 6) | 0 |

Measured by [`comparisons/builtin_saves.py`](comparisons/builtin_saves.py).

TabICL's documentation presents `save_training_data=False` as giving "better data privacy": the file keeps the model's cached key-value projections of the table instead of the table, and predictions after reload are unchanged. It works only if the model was fitted with `kv_cache=True` and it is off by default. It leaves no income verbatim: the 4 matches are chance, as many as values absent from the table reach, they change from one dataset to the next, and none sits inside an array of the reloaded model.

The cache still gives the rows back. Its first block stores, for every training row, keys and values that are a linear function of that row, and the weights that produced them are public, so each row's representation can be read back. The model in the file computes the same representation for any row passed to it for prediction, so a search over candidate rows finds the one that matches:

| Model fitted with `kv_cache=True`, saved with `save_training_data=False` | Rows rebuilt within 1% | Cells within 1% | Median cell error | Labels rebuilt |
|---|---|---|---|---|
| TabICL 2.2.0, 3 columns | 300 / 300 | 100.0% | 0.02% | 300 / 300 |

Measured by [`comparisons/kv_cache_inversion.py`](comparisons/kv_cache_inversion.py), which uses the saved file, the TabICL library and its public weights, and nothing else; the original table only scores the result. No value comes back to the cent, but every one comes back close. The search is bounded by the mean and the standard deviation of every column, which the file also keeps in clear. Removing the table from the file is not enough while a cache derived from its rows stays there.

TabPFN's `.tabpfn_fit` archive leaves out the foundation model's weights but keeps the table, and TabDPT offers no option to drop it. TabPFN is about to follow TabICL: [pull request 1323](https://github.com/PriorLabs/TabPFN/pull/1323), merged on 28 September 2026 for memory reasons and not yet released, drops each ensemble member's table once its cache is built in `fit_mode="fit_with_cache"`. Measured on TabPFN's main branch at commit `dbba40314b`, with TabPFN v2 weights: the default mode keeps 300 of 300 incomes in the `.tabpfn_fit` archive, and `fit_with_cache` keeps the smallest and the largest value of every column — here the incomes of the poorest and the richest customer — and no other income beyond chance. Predictions after reload are unchanged in both.

## What existing tools see

The closest prior work is [SACRO-ML](https://github.com/AI-SDC/SACRO-ML), which checks models leaving trusted research environments. Its instance-based attack detects models that store their training rows — SVMs and k-nearest-neighbours, the classic cases — by comparing the stored instances with the training data you give it. On its own ground it works, and on anything else it reports no risk:

| Model | SACRO-ML 2.0.1 verdict | Rows it matched |
|---|---|---|
| KNeighborsClassifier (scikit-learn 1.9.1) | leakage confirmed | 300 / 300 |
| TabICL 2.2.0 | not instance-based, no data leakage risk | 0 / 300 |

Same 300 customers as above, measured by [`comparisons/sacroml_check.py`](comparisons/sacroml_check.py). The TabICL file it clears is one from which every income is rebuilt.

Other tools look at model files for a different reason, or skip them:

- **Pickle security scanners** — [fickling](https://github.com/trailofbits/fickling), [picklescan](https://github.com/mmaitre314/picklescan), [modelscan](https://github.com/protectai/modelscan), the [Hugging Face pickle scan](https://huggingface.co/docs/hub/security-pickle) — look for code that runs when the file loads. Arrays are data, so they are not their concern.
- **Model auditors** — [ModelAudit](https://github.com/promptfoo/modelaudit) also searches weights for embedded credentials such as API keys, not for rows of a dataset.
- **Cloud data-loss prevention** — [Amazon Macie](https://docs.aws.amazon.com/macie/latest/user/discovery-supported-storage.html) classifies pickle and NumPy files as objects it cannot analyse, so personal data inside them goes unreported.

tabcustody discovers the table without being handed the training data, covers the in-context tabular models, and tells when values stored behind a scaler can be rebuilt, as with TabDPT.

## Guarantees and the tests that prove them

| Guarantee | Test |
|---|---|
| Reading a file runs none of its code, joblib object arrays included | `test_safety.py::test_a_pickle_that_runs_a_command_on_load_runs_nothing`, `test_formats.py::test_an_object_array_inside_a_joblib_file_is_read_without_running_it` |
| Scanning imports no model library | `test_safety.py::test_scanning_imports_neither_torch_nor_xgboost` |
| The training table of TabICL, TabPFN v2, TabPFN v3.5 and TabDPT is found | `test_detection.py::test_the_training_table_of_each_in_context_model_is_found` (TabPFN and TabDPT files are too large to version: marked `integration`), `test_formats.py::test_a_tabpfn_fit_archive_is_scanned_member_by_member` |
| A TabICL file saved with `save_training_data=False` is reported for its key-value cache | `test_detection.py::test_a_tabicl_key_value_cache_is_reported_as_the_training_rows_it_was_built_from`, `::test_a_tabicl_file_saved_without_its_training_data_still_reports_its_cache` (marked `integration`) |
| Instance-based models (k-nearest-neighbours, SVM) are reported | `test_detection.py::test_instance_based_models_are_reported_like_sacroml_does` |
| No finding on XGBoost, LightGBM, logistic regression, random forest, gradient boosting, a 100×100 MLP or a QuantileTransformer pipeline | `test_detection.py::test_classic_models_report_no_table`, `::test_a_square_weight_matrix_next_to_its_bias_is_not_a_table`, `::test_a_quantile_grid_beside_its_levels_is_not_a_table` |
| A table kept as a pandas DataFrame, pickled by pandas 2 or 3, is found and rebuilt in its column order | `test_detection.py::test_a_table_kept_as_a_pandas_dataframe_is_found`, `::test_a_dataframe_is_rebuilt_column_by_column_in_its_own_order` |
| A table stored standardised is declared reversible, and reversing gives the exact rows | `test_reversal.py::test_a_standardised_table_is_rebuilt_from_the_scaler_saved_beside_it`, `::test_a_scaled_pipeline_table_is_rebuilt_from_the_scaler_step` |
| A scaler that does not feed the table is never credited | `test_reversal.py::test_a_scaler_nested_inside_another_transformer_is_not_credited`, `::test_a_table_stored_as_is_is_not_said_to_be_reversible` |
| The report never prints a value of the table | `test_report.py::test_the_report_never_prints_a_value_of_the_table` |
| joblib files, compressed or not, read like pickles | `test_formats.py::test_a_joblib_file_is_scanned_like_a_pickle`, `::test_compressed_joblib_files_are_read` |
| The command exits 1 on a file that carries training data | `test_cli.py::test_the_command_exits_non_zero_when_a_table_is_found` |

## Limits

- **A table is recognised by its target.** The scanner looks for a 2-D numeric array beside a 1-D array with one entry per row. Rows kept without a target, or tables under 20 rows (`--min-rows`), are not reported.
- **A one-column table stored in sorted order** is taken for a lookup grid, such as QuantileTransformer's quantiles, and not reported.
- **Only numeric columns are read.** Text and category columns of a DataFrame stay in the file but are not part of the reported shape.
- **Reversal covers scikit-learn's StandardScaler, MinMaxScaler and RobustScaler**, when the scaler sits beside the table or is the pipeline step right before it. Tables behind other transforms are still reported; they are not called reversible.
- **Caches.** TabICL's key-value cache is reported. TabPFN's, kept in `fit_mode="fit_with_cache"` by releases that include pull request 1323, is not: whether its rows can be rebuilt has not been measured.
- **Formats.** Not read: lz4-compressed joblib, safetensors, ONNX, `torch.save` archives. PyTorch tensors inside a pickle stay opaque placeholders.
- **Memory.** The whole object tree is loaded: scanning the 836 MB TabPFN v3.5 pickle peaks at about the size of the file.

## What comes next

1. **Detect** — shipped in 0.1.
2. **Separate.** Save the model without its table, and supply the table at prediction time from a place you control.
3. **Erase.** Find a person's rows, remove them, refit, and keep a record that proves it — an erasure request under GDPR article 17 becomes a routine operation with these models, where classic models would need retraining.
4. **Trace and verify.** Know which exact table was in place for any past prediction, and check that production predicts like development.

Each step ships on its own, with its guarantees backed by named tests.

## What tabcustody does not claim

- **Leakage through predictions.** Whether an attacker can infer training rows from a model's answers is a research question, studied in [arXiv 2606.26021](https://arxiv.org/abs/2606.26021) and [arXiv 2606.31474](https://arxiv.org/abs/2606.31474). tabcustody is about the far plainer leak above: the rows are in the file.
- **That users are already reporting this.** As of 28 September 2026 no public issue describes it. These models are young; tabcustody is built ahead of the moment teams put them in production at scale.
- **Models not in the table.** TabPFN 2.5, 2.6 and 3, and AutoGluon's Mitra, are not measured yet. They join the table when `comparisons/leak.py` has run on them.

## License

MIT
