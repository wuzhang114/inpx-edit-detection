# INP-X Edit Forensics: Reproducible TIFS Release

This repository contains the code and analysis artifacts for the manuscript
**What Does Pixel Supervision Buy? Evidence Allocation in Frozen-Feature Edit Forensics**.
The release studies image-edit detection and localization with frozen DINOv2
features, a lightweight patch readout, nested mask budgets, and a matched
pixel-loss intervention.

## Repository contents

- `src/`: feature extraction, readout training, evaluation, and canonical-source utilities.
- `tools/`: final bootstrap, budget-fit, transfer-evaluation, audit, and figure scripts.
- `outputs/`: final result JSON files used by the manuscript.
- `figures/`: English publication figures used by the manuscript.
- `TIFS_REPRODUCTION.md`: environment, data, and reproduction instructions.

The repository intentionally excludes raw datasets, feature caches, training
checkpoints, third-party repositories, and local build directories. INP-X and
the external datasets remain subject to their original licenses.

## Main results

The matched full-budget intervention increases localization micro IoU from
0.049 to 0.547 and pixel AUROC from 0.575 to 0.948, while detection AUC
changes from 0.949 to 0.922. The final budget fit and source-cluster bootstrap
outputs are stored under `outputs/`.

## License

Code is released under the MIT License. Dataset and pretrained-model licenses
remain governed by their respective providers.
