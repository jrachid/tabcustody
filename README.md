# tabcustody

> With in-context tabular models, the model file **is** your training data. tabcustody proves it on your files, then lets you deploy, erase and trace without exposing that data.

**Status: pre-alpha.** Version 0.2 ships the scanner: it tells whether a model file carries its training data, without running the file. Separating, erasing and tracing come next.

## Quick start

```bash
pip install tabcustody
tabcustody scan model.pkl other.joblib fitted.tabpfn_fit
```

The command reads pickle files, joblib files (plain or compressed with zlib, gzip, bz2 or xz) and zip archives of them such as TabPFN's `.tabpfn_fit`. It exits with 1 when a file carries training data, 2 when a file cannot be read, 0 otherwise, so it can gate a CI job; the extremes of each column are reported but do not, alone, make a file carry training data, since a scaler such as MinMaxScaler keeps them in many classic pipelines; `--json` prints one report per file.

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

On a TabPFN archive the report names the rows of every ensemble member, and on a TabICL file saved with `save_training_data=False` it reports the key-value cache the rows can be rebuilt from, and the extremes of each column kept beside it:

```
fitted.tabpfn_fit (zip)
  training table  executor_state.joblib:ensemble_members[*].X_train  300 rows x 10 or 4 columns  (8 alike)
    why: the 1-D array executor_state.joblib:ensemble_members[*].y_train has one entry per row
    why: X_train is a name libraries use for training rows
    fix: TabPFN keeps the rows in its default fit mode. From tabpfn 9.1.0, fit_mode='fit_with_cache' drops them but keeps a key-value cache derived from them, and every value of each column, sorted column by column, with the default weights; the smallest and largest value of each column with the v2 weights: keep this file as confidential as the data it was fitted on.

tabicl-cache.pkl (pickle)
  key-value cache  model_kv_cache_  300 rows x 3 columns
    why: TabICL's in-context cache keeps one key and one value per training row
    fix: TabICL's key-value cache lets the training rows be rebuilt with the library and its public weights: keep this file as confidential as the data it was fitted on.
  column extremes  ensemble_generator_.preprocessors_['power']  smallest and largest value of 3 columns
    why: X_min_ and X_max_ keep the smallest and largest value of each column, which the scaler at ensemble_generator_.preprocessors_['power'].standard_scaler_ brings back to their scale
    fix: The smallest and largest value of each column are the values of the people at either end of it, kept in clear even without the table: keep this file as confidential as the data it was fitted on.
```

The scanner never unpickles the file the usual way: only NumPy arrays and plain containers are rebuilt, every other class becomes an inert placeholder, and no library the file names is imported. The report gives paths, shapes and evidence, never a value of the table.

## The problem, measured

A classic model such as XGBoost learns rules from its training data, then forgets the data: the saved file holds split thresholds, nothing else. The new tabular foundation models — TabPFN, TabICL, TabDPT — work differently. They predict by reading the training table again at every prediction, the way a language model reads its prompt. So a fitted model keeps the table, and saving the model saves the table.

The same 300 synthetic customers, each model fitted then saved with `pickle`, and only the saved file inspected afterwards:

| Model | Saved file | Incomes found verbatim | Incomes rebuilt from the file alone |
|---|---|---|---|
| XGBoost 3.4.1 (control) | 0.1 MB | 0 / 300 (chance: 0) | 0 / 300 |
| TabICL 2.2.0 | < 0.1 MB | 300 / 300 (chance: 0) | 300 / 300 |
| TabPFN v2 (tabpfn 9.2.0) | 29.2 MB | 300 / 300 (chance: 2) | 300 / 300 |
| TabPFN v3.5, the default weights (tabpfn 9.2.0) | 876.3 MB | 300 / 300 (chance: 21) | 300 / 300 |
| TabDPT 1.3.1 | 252.5 MB | 11 / 300 (chance: 11) | 300 / 300 |

A value is found verbatim when its 8-byte or 4-byte encoding appears in the file. A large file of model weights contains some of those byte patterns by coincidence, so each count comes with the number that 300 values absent from the table reach in the same file.

Measured on 30 September 2026, the TabPFN rows again on 24 October 2026 with tabpfn 9.2.0, by [`comparisons/leak.py`](comparisons/leak.py), which anyone can rerun with the pinned versions of [`comparisons/pyproject.toml`](comparisons/pyproject.toml).

TabDPT is the instructive row. It stores the table standardised — an income of 38,693.02 becomes -0.058 — so searching the file for known values finds no more than chance does, and a manual check concludes the file is clean. But the scaler that standardised the table is saved in the same file, and calling it in reverse rebuilds every income to the cent.

Nothing here is a bug in these libraries: keeping the table is how in-context learning works. The risk is in the habits around model files. Teams have treated them as harmless build artefacts for a decade — committed to Git, copied to buckets, sent to vendors — and with these models each of those copies is a copy of the customer table.

## What the libraries' own saves keep

The table above uses `pickle`. Each library's own save function tells the same story by default, and two of them offer an option that drops the table:

| Save | Incomes found verbatim | Largest prediction gap after reload |
|---|---|---|
| TabPFN v2 `save_fit_state` (tabpfn 9.2.0), `fit_mode="fit_preprocessors"` | 300 / 300 (chance: 0) | 0 |
| TabPFN v2 `save_fit_state` (tabpfn 9.2.0), `fit_mode="fit_with_cache"` | 4 / 300 (chance: 1) | 0 |
| TabPFN v3.5, the default weights, `save_fit_state` (tabpfn 9.2.0), `fit_mode="fit_preprocessors"` | 300 / 300 (chance: 0) | 0 |
| TabPFN v3.5, the default weights, `save_fit_state` (tabpfn 9.2.0), `fit_mode="fit_with_cache"` | 300 / 300 (chance: 0) | 0 |
| TabICL 2.2.0 `save(save_training_data=True)`, `kv_cache=False` | 300 / 300 (chance: 0) | 0 |
| TabICL 2.2.0 `save(save_training_data=False)`, `kv_cache=True` | 4 / 300 (chance: 6) | 0 |

Measured by [`comparisons/builtin_saves.py`](comparisons/builtin_saves.py).

TabICL's documentation presents `save_training_data=False` as giving "better data privacy": the file keeps the model's cached key-value projections of the table instead of the table, and predictions after reload are unchanged. It works only if the model was fitted with `kv_cache=True` and it is off by default. It leaves no income verbatim: the 4 matches are chance, as many as values absent from the table reach, they change from one dataset to the next, and none sits inside an array of the reloaded model.

The cache still gives the rows back. Its first block stores, for every training row, keys and values that are a linear function of that row, and the weights that produced them are public, so each row's representation can be read back. The model in the file computes the same representation for any row passed to it for prediction, so a search over candidate rows finds the one that matches, the collision attack that [Shadow in the Cache](https://arxiv.org/abs/2508.09442) (NDSS 2026) describes for the key-value caches of language models:

| Model fitted with `kv_cache=True`, saved with `save_training_data=False` | Rows rebuilt within 1% | Cells within 1% | Median cell error | Labels rebuilt |
|---|---|---|---|---|
| TabICL 2.2.0, 3 columns | 300 / 300 | 100.0% | 0.02% | 300 / 300 |
| TabICL 2.2.0, 20 columns | 168 / 300 | 93.6% | 0.08% | 300 / 300 |

Measured by [`comparisons/kv_cache_inversion.py`](comparisons/kv_cache_inversion.py), which uses the saved file, the TabICL library and its public weights, and nothing else; the original table only scores the result. The search gives no value back to the cent. With 20 columns it is harder and stops short on 132 rows, though most of their cells still come back within 1%. The file also keeps statistics of every column in clear: the mean and the standard deviation, which bound the search, and the smallest and the largest value, stored scaled beside the scaler that brings them back to the cent — here the ages of the youngest and the oldest customer, and the lowest and the highest income and debt. Removing the table from the file is not enough while a cache derived from its rows stays there.

TabDPT offers no option to drop the table. TabPFN's `.tabpfn_fit` archive leaves out the foundation model's weights but keeps the table in the default fit mode, `fit_mode="fit_preprocessors"`. Since tabpfn 9.1.0, released on 2 October 2026, `fit_mode="fit_with_cache"` drops each ensemble member's table once its key-value cache is built ([pull request 1323](https://github.com/PriorLabs/TabPFN/pull/1323), merged for memory reasons), and predictions after reload are unchanged. What the archive keeps then depends on the weights. With the v2 weights it keeps the smallest and the largest value of every column — here the incomes of the poorest and the richest customer — and no other income beyond chance. With the default weights it keeps every income: the cache stores the values of every column, sorted column by column, each to the cent. The cache is reported by `tabcustody scan` in both; whether the rows can be rebuilt from it, as with TabICL, has not been measured.

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
| Reading skips the weights, keeping each table in full | `test_reader.py::test_a_pytorch_storage_is_dropped_for_a_placeholder_that_keeps_its_length`, `::test_a_large_numpy_array_is_read_in_full` |
| The training table of TabICL, TabPFN v2, TabPFN v3.5 and TabDPT is found | `test_detection.py::test_the_training_table_of_each_in_context_model_is_found` (TabPFN and TabDPT files are too large to version: marked `integration`), `test_formats.py::test_a_tabpfn_fit_archive_is_scanned_member_by_member` |
| A TabICL file saved with `save_training_data=False` is reported for its key-value cache | `test_detection.py::test_a_tabicl_key_value_cache_is_reported_as_the_training_rows_it_was_built_from`, `::test_a_tabicl_file_saved_without_its_training_data_still_reports_its_cache` (marked `integration`) |
| A TabPFN file fitted with `fit_mode="fit_with_cache"` (tabpfn 9.1.0 and later) is reported for its key-value cache, quantized to 8 bits or not | `test_detection.py::test_a_tabpfn_key_value_cache_is_reported_as_the_training_rows_it_was_built_from`, `::test_a_tabpfn_file_fitted_with_cache_reports_its_cache` (marked `integration`) |
| A cache whose classes are not a known one is still reported when its tensors have a per-row axis | `test_detection.py::test_a_cache_of_an_unknown_class_is_found_by_its_shape` |
| The smallest and largest value of each column that TabICL keeps behind its scaler are reported, with or without the table, and alone do not make the command exit 1 | `test_detection.py::test_tabicl_column_extremes_are_reported_beside_the_table`, `::test_tabicl_column_extremes_are_reported_even_without_the_table` (marked `integration`), `test_report.py::test_column_extremes_alone_do_not_mean_the_file_carries_training_data` |
| Instance-based models are reported: k-nearest-neighbours, SVM, and without a target NearestNeighbors, KernelDensity, KernelPCA and Isomap | `test_detection.py::test_instance_based_models_are_reported_like_sacroml_does`, `::test_models_without_a_target_are_reported_by_the_width_they_were_fitted_on` |
| No finding on XGBoost, LightGBM, logistic regression, random forest, gradient boosting, a 100×100 MLP or a QuantileTransformer pipeline | `test_detection.py::test_classic_models_report_no_table`, `::test_a_square_weight_matrix_next_to_its_bias_is_not_a_table`, `::test_a_quantile_grid_beside_its_levels_is_not_a_table` |
| Arrays computed from the rows are not reported as rows: a Gaussian process's Cholesky factor, LocalOutlierFactor's neighbour distances, out-of-bag predictions beside a k-nearest-neighbours table | `test_detection.py::test_a_cholesky_factor_is_not_called_a_training_table`, `::test_neighbour_distances_are_not_called_a_training_table`, `::test_out_of_bag_probabilities_are_not_a_copy` |
| A table kept as a pandas DataFrame, pickled by pandas 2 or 3, is found and rebuilt in its column order | `test_detection.py::test_a_table_kept_as_a_pandas_dataframe_is_found`, `::test_a_dataframe_is_rebuilt_column_by_column_in_its_own_order` |
| A table stored standardised is declared reversible, and reversing gives the exact rows | `test_reversal.py::test_a_standardised_table_is_rebuilt_from_the_scaler_saved_beside_it`, `::test_a_scaled_pipeline_table_is_rebuilt_from_the_scaler_step` |
| A table behind a PowerTransformer is rebuilt to the cent, one behind a QuantileTransformer closely | `test_reversal.py::test_a_power_transformed_table_is_rebuilt_to_the_cent`, `::test_a_quantile_transformed_table_is_rebuilt_closely` |
| A scaler that does not feed the table is never credited | `test_reversal.py::test_a_scaler_nested_inside_another_transformer_is_not_credited`, `::test_a_table_stored_as_is_is_not_said_to_be_reversible` |
| The report never prints a value of the table | `test_report.py::test_the_report_never_prints_a_value_of_the_table` |
| The report names the columns when the model kept their names, never a value | `test_report.py::test_the_report_names_the_columns_when_the_model_kept_their_names` |
| joblib files, compressed or not, read like pickles | `test_formats.py::test_a_joblib_file_is_scanned_like_a_pickle`, `::test_compressed_joblib_files_are_read` |
| The command exits 1 on a file that carries training data | `test_cli.py::test_the_command_exits_non_zero_when_a_table_is_found` |
| A sealed model reloads and predicts as before, and a scan finds nothing in it | `test_seal.py::test_a_sealed_file_round_trips_and_a_scan_finds_nothing` |
| A sealed file holds neither the key nor a value of the table | `test_seal.py::test_the_sealed_file_holds_neither_the_key_nor_a_value_of_the_table` |
| A wrong key is refused before anything is rebuilt, and a changed file is refused | `test_seal.py::test_a_wrong_key_is_refused_before_anything_is_rebuilt`, `::test_a_changed_file_is_refused` |
| The manifest does not let a held dataset be confirmed without the key | `test_seal.py::test_the_manifest_does_not_confirm_a_known_table` |

## Limits

- **A table is recognised by its target or by its width.** The scanner looks for a 2-D numeric array beside a 1-D array with one entry per row, or one with as many columns as the estimator above it declares in `n_features_in_`. Rows kept without a target by a class that does not declare `n_features_in_`, or tables under 20 rows (`--min-rows`), are not reported.
- **A one-column table stored in sorted order** is taken for a lookup grid, such as QuantileTransformer's quantiles, and not reported.
- **Only numeric columns are read.** Text and category columns of a DataFrame stay in the file but are not part of the reported shape.
- **Column names are shown when the model kept them** in `feature_names_in_` on the estimator that holds the table, such as a k-nearest-neighbours model fitted on a named DataFrame. A table stored preprocessed (TabPFN, TabICL), behind a scaler, or without names is reported by its shape alone. The report gives the names, never a value.
- **Reversal covers scikit-learn's StandardScaler, MinMaxScaler and RobustScaler**, when the scaler sits beside the table or is the pipeline step right before it, and **PowerTransformer and QuantileTransformer** when they are the pipeline step right before it. The quantile inverse is close but not exact: a value outside the learned quantiles is clipped and does not come back to the cent. Tables behind other transforms are still reported; they are not called reversible.
- **Caches.** The key-value caches of TabICL and of TabPFN 9.1.0 and later are reported, recognised by the names of their classes. A cache whose classes are not known is still reported when it holds PyTorch tensors with an axis equal to a training-row count the file records; one that keeps no such tensor, or whose row count the file does not record, is not. Whether TabPFN's rows can be rebuilt from its cache has not been measured.
- **Column extremes** are recognised as `X_min_` and `X_max_` beside a scaler of the same width, as TabICL stores them. Others are not reported: those of MinMaxScaler (`data_min_`, `data_max_`), and those TabPFN v2 keeps in the first and last rows of a quantile grid.
- **Formats.** Not read: lz4-compressed joblib, safetensors, ONNX, `torch.save` archives. PyTorch tensors inside a pickle stay opaque placeholders.
- **Memory.** The object tree is read without the weights: the bytes of a PyTorch storage are skipped and replaced by a placeholder that keeps their length, so scanning the 836 MB TabPFN v3.5 pickle peaks at about 40 MB. A large table stored as a NumPy array is still read in full, since it is what the scanner looks for.

## What comes next

1. **Detect** — shipped in 0.1.
2. **Separate** — in development on the main branch. `pip install tabcustody[seal]`, then `tabcustody.seal(model, path, key)` writes the model with its training table, copies, target, column extremes and key-value cache encrypted (AES-256-GCM, with a commitment to the key so a file cannot be made to open under two keys); `tabcustody.unseal(path, key)` reloads it to predict exactly as before. Without the key a scan finds neither a table nor a cache. A model that keeps its rows only inside a C object's state, such as KernelDensity's KD tree, is refused rather than half-sealed. The `tabcustody seal` command and a release come next.
3. **Erase.** Find a person's rows, remove them, refit, and keep a record that proves it — an erasure request under GDPR article 17 becomes a routine operation with these models, where classic models would need retraining.
4. **Trace and verify.** Know which exact table was in place for any past prediction, and check that production predicts like development.

Each step ships on its own, with its guarantees backed by named tests.

## What tabcustody does not claim

- **Leakage through predictions.** Whether an attacker can infer training rows from a model's answers is a research question, studied in [arXiv 2606.26021](https://arxiv.org/abs/2606.26021) and [arXiv 2606.31474](https://arxiv.org/abs/2606.31474). tabcustody is about the far plainer leak above: the rows are in the file.
- **That users are already reporting this.** As of 28 September 2026 no public issue describes it. These models are young; tabcustody is built ahead of the moment teams put them in production at scale.
- **Models not in the table.** TabPFN 2.5, 2.6 and 3, and AutoGluon's Mitra, are not measured yet. They join the table when `comparisons/leak.py` has run on them.

## License

MIT
