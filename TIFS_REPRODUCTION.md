# Reproduction guide

## Environment and scope

The recorded experiments used Python 3.10, PyTorch 2.5.1, and an NVIDIA GeForce RTX 4060 Laptop GPU with approximately 8 GB of memory. CPU budget fitting requires NumPy and SciPy. Training and inference additionally use PyTorch, Pillow, scikit-learn, and tqdm; MagicBrush source-mask extraction uses PyArrow. This is an experiment environment description, not a fully locked environment.

## Budget-response analysis

```bash
python -m pip install -r requirements-analysis.txt
python tools/refit_final.py
```

Inputs are the 21 main per-run evaluation JSON files and `phase_law_validation.json` in `outputs/`. The script overwrites `outputs/refit_final.json`. Preserve that file for comparison if using a different environment. The fit uses seven budget means; leave-one-budget-out evaluation excludes all runs at the held-out budget. Intermediate-budget observations are retrospective checks.

## Training and inference

The main trainer is `src/baselines/weakly_supervised_v4.py`. Configure its `CACHE`, `IMG_ROOT`, and `OUT` paths, obtain the licensed datasets, and supply source-disjoint split and annotation manifests matching the feature-cache row order. Source identities are implemented in `src/data/canonical_source.py`.

The main inputs are frozen DINOv2-S/14 tokens at resolution 518 and a fixed high-pass channel. The readout has 29,699 trainable parameters. The main schedule is 20 epochs with batch size 128 per stream. Positive-budget mask batches match the image batch size, including the last partial batch.

Example, after preparing data and manifests:

```bash
python src/baselines/weakly_supervised_v4.py --master_split outputs/split_budget_master.json --mask_manifest outputs/mask_budget_manifest.json --mask_budget ALL --epochs 20 --batch_size 128 --hidden 64 --seed 42 --tag reproduction_on_s42
```

For the matched control, add `--pixel_off` and use a distinct tag. The example requires artifacts not supplied in this repository. It is not a data-download command.

Final full-budget runs seed model construction and are identified by `g2u_s42/s7/s2024`; the matched control is `po_bALL_s42`. Earlier lower-budget runs seeded after construction, so recorded seeds do not fully determine their initial weights. The earlier `g2` auxiliary configuration uses 10 epochs and batch size 256.

## Current analysis entry points

| Script | Required inputs |
| --- | --- |
| `tools/refit_final.py` | Included evaluation summaries |
| `tools/paired_boot_canonical_v2.py` | Aligned test data, masks, feature caches, and named checkpoints |
| `tools/external_final.py` | External datasets, DINOv2 weights, readout checkpoints, and reference results |
| `tools/complete_baseline_table.py` | Saved baseline maps and the external FLAME implementation |
| `tools/plot_tifs_matlab_v22.m` | MATLAB and included budget/layer summary JSONs |

Historical absolute paths and local Torch Hub paths must be adapted before inference. Main feature arrays use raw memory maps despite their `.npy` suffixes; use the recorded shape, dtype, and row mapping.

## Evaluation conventions

Detection uses the maximum patch probability. Localization micro IoU pools intersections and unions; mean per-image IoU averages images equally. Source-validation thresholds are frozen before testing. Target-test oracle thresholds are reported as diagnostics.

The paired bootstrap uses 2,000 canonical-source cluster resamples with fixed checkpoints. Three-run standard deviations measure run variation separately. The matched loss intervention uses one seed.

MagicBrush masks use `alpha != 255` from the official RGBA annotations. `tools/repair_magicbrush_alpha_masks_v20.py` documents the conversion and requires original source files and saved predictions. Current results on 528 images are mean IoU 0.1022 at threshold 0.27, oracle IoU 0.1674, and patch AUROC 0.6500. The historical `src/evaluation/extract_magicbrush.py` luminance conversion must not be used for these results.

The baseline table uses one threshold per method selected to maximize dataset micro IoU; both IoU columns use the same predictions. The proposed readout is trained on INP-X masks; comparison baselines retain their original training configurations.

## Figures and historical scripts

The `figures/` directory contains the current method diagram, budget and layer curves, and qualitative examples. The MATLAB script writes regenerated curves to `paper/figures/matlab_v22/`; the public copies use stable filenames under `figures/`.

Legacy v1--v6 variants, EFD probes, manuscript assembly utilities, and older plotting scripts are retained for experiment history. Some require local manuscript assets. They are not the main reproduction entry points. In particular, `efd_zero_detector.py` uses mask-conditioned diagnostics and must not be treated as a mask-free detector.

## Provenance

Recorded checkpoint hashes and result provenance are included in the analysis JSON files. Local path strings in those records identify the original environment; they do not indicate that data or checkpoints are bundled. The release does not distribute raw images, masks, feature arrays, or checkpoints.
