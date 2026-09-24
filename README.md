# INP-X Edit Forensics

Code and analysis artifacts for **What Does Pixel Supervision Buy? Evidence Allocation in Frozen-Feature Edit Forensics**.

**Authors:** Boxuan Li, Peng Gao, and Canjie Wang.

**Affiliation:** College of Artificial Intelligence and Low-Altitude Technology, South China Agricultural University.

**Corresponding author:** Peng Gao (gaopeng.peng@scau.edu.cn).

The study examines how mask supervision affects image-edit detection and localization using frozen visual features, lightweight patch readouts, and source-disjoint evaluation. This repository accompanies a manuscript; it does not imply acceptance by a journal.

## Main findings

In the matched full-budget experiment (seed 42), adding pixel losses increases localization micro IoU from 0.0490 to 0.5467 and patch AUROC from 0.5752 to 0.9477, while detection AUC changes from 0.9490 to 0.9222. The main budget sweep contains seven mask budgets and three runs per budget.

The manuscript distinguishes the matched loss intervention from the budget sweep, whose sampling support and historical initialization conventions vary. External evaluation measures transfer with source-validation thresholds and reports target-test oracle thresholds separately.

## Quick start: CPU analysis

From the repository root:

```bash
python -m pip install -r requirements-analysis.txt
python tools/refit_final.py
```

The required evaluation summaries are included in `outputs/`. The command recomputes `outputs/refit_final.json`; no images, feature arrays, checkpoints, or GPU are needed. The packaged inputs reproduce the recorded fit in the checked environment. Numerical-library versions may affect low-order digits.

Training and inference require datasets and model artifacts. See [Reproduction guide](TIFS_REPRODUCTION.md) for paths, requirements, and evaluation conventions.

## Repository contents

| Directory | Contents |
| --- | --- |
| `src/baselines/` | Main v4 trainer and historical readout variants |
| `src/data/` | Canonical source-identity parsing |
| `src/evaluation/` | Evaluation scripts and exploratory diagnostics |
| `tools/` | Budget fits, source-cluster bootstrap, external evaluation, and figure generation |
| `outputs/` | Recorded evaluation summaries and analysis results |
| `figures/` | Current English manuscript figures |

The main training entry point is `src/baselines/weakly_supervised_v4.py` with the master split and annotation manifest. The current analysis entry points are `refit_final.py`, `paired_boot_canonical_v2.py`, and `external_final.py`. Other variants and legacy diagnostic scripts document exploratory experiments and are not interchangeable with the main protocol.

## Data and artifact availability

Raw datasets, masks, feature caches, trained checkpoints, and third-party repositories are not distributed here. Inference and training scripts retain experiment-specific paths that require adaptation. The public release supports CPU recomputation of the budget fits; reproducing all training and inference requires the separately obtained datasets and model artifacts.

MagicBrush results use the corrected alpha-channel mask convention (`alpha != 255`). Current baseline results are in `outputs/baseline_table_completed_20260920.json`.

## License

Project code is released under the MIT License. Dataset, pretrained-model, and third-party software licenses remain governed by their respective providers.
