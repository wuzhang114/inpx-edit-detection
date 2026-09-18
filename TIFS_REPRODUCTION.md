# Reproduction guide

The commands below assume Python 3.10, PyTorch 2.5.1, and a CUDA-capable
GPU. Install the project dependencies required by the selected training or
evaluation script, then place the licensed INP-X data and feature caches in
local paths configured by the script arguments.

## Final analysis artifacts

The final analysis JSON files are included in `outputs/`:

- `paired_boot_final.json`: paired source-cluster bootstrap for the matched intervention;
- `refit_final.json`: seven-budget grouped response fits and held-out errors;
- `external_final.json`: final cross-editor evaluation;
- `provenance_audit.json`: input, checkpoint, and evaluation provenance.

The main analysis can be regenerated with:

```bash
python tools/paired_boot_canonical_v2.py
python tools/refit_final.py
python tools/external_final.py
```

The scripts require the local dataset and checkpoint paths used by the
experiment. They do not download or redistribute INP-X data.

## Figures

The final English figures are in `figures/`. The manuscript source is
maintained separately from this code release.

## Reproducibility conventions

Canonical source groups are formed with `src/data/canonical_source.py`.
Validation thresholds are selected before test evaluation. Bootstrap resamples
retain all members of a source cluster. The budget fit uses seven budget means;
three runs at a budget are used to estimate run variation and are not treated
as independent budget locations.
