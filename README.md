# tabcustody

> With in-context tabular models, the model file **is** your training data. tabcustody proves it on your files, then lets you deploy, erase and trace without exposing that data.

**Status: pre-alpha — there is nothing to install yet.** This README states the problem, measured. The library comes next, starting with the scanner.

## The problem, measured

A classic model such as XGBoost learns rules from its training data, then forgets the data: the saved file holds split thresholds, nothing else. The new tabular foundation models — TabPFN, TabICL, TabDPT — work differently. They predict by reading the training table again at every prediction, the way a language model reads its prompt. So a fitted model keeps the table, and saving the model saves the table.

The same 300 synthetic customers, each model fitted then saved with `pickle`, and only the saved file inspected afterwards:

| Model | Saved file | Incomes found verbatim | Incomes rebuilt from the file alone |
|---|---|---|---|
| XGBoost 3.4.1 (control) | 0.1 MB | 0 / 300 | 0 / 300 |
| TabICL 2.2.0 | < 0.1 MB | 300 / 300 | 300 / 300 |
| TabPFN v2 (tabpfn 9.0.0) | 29.2 MB | 300 / 300 | 300 / 300 |
| TabPFN v3.5, the default weights (tabpfn 9.0.0) | 876.3 MB | 300 / 300 | 300 / 300 |
| TabDPT 1.3.1 | 252.5 MB | 11 / 300 | 300 / 300 |

Measured on 28 September 2026 by [`comparisons/leak.py`](comparisons/leak.py), which anyone can rerun with the pinned versions of [`comparisons/pyproject.toml`](comparisons/pyproject.toml).

TabDPT is the instructive row. It stores the table standardised — an income of 38,693.02 becomes -0.058 — so searching the file for known values finds almost nothing, and a manual check concludes the file is clean. But the scaler that standardised the table is saved in the same file, and calling it in reverse rebuilds every income to the cent.

Nothing here is a bug in these libraries: keeping the table is how in-context learning works. The risk is in the habits around model files. Teams have treated them as harmless build artefacts for a decade — committed to Git, copied to buckets, sent to vendors — and with these models each of those copies is a copy of the customer table.

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

tabcustody is meant to discover the table without being handed the training data, to cover the in-context tabular models, and to rebuild values stored behind a reversible transform, as with TabDPT.

## What tabcustody will do

1. **Detect.** Tell whether a model file carries training data, including data stored transformed next to the object that can reverse it.
2. **Separate.** Save the model without its table, and supply the table at prediction time from a place you control.
3. **Erase.** Find a person's rows, remove them, refit, and keep a record that proves it — an erasure request under GDPR article 17 becomes a routine operation with these models, where classic models would need retraining.
4. **Trace and verify.** Know which exact table was in place for any past prediction, and check that production predicts like development.

Only the first is under way. Each step ships on its own, with its guarantees backed by named tests.

## What tabcustody does not claim

- **Leakage through predictions.** Whether an attacker can infer training rows from a model's answers is a research question, studied in [arXiv 2606.26021](https://arxiv.org/abs/2606.26021) and [arXiv 2606.31474](https://arxiv.org/abs/2606.31474). tabcustody is about the far plainer leak above: the rows are in the file.
- **That users are already reporting this.** As of 28 September 2026 no public issue describes it. These models are young; tabcustody is built ahead of the moment teams put them in production at scale.
- **Models not in the table.** TabPFN 2.5, 2.6 and 3, and AutoGluon's Mitra, are not measured yet. They join the table when `comparisons/leak.py` has run on them.

## License

MIT
