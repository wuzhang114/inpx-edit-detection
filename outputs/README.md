# Recorded analysis inputs and results

Current manuscript results are in `paired_boot_final.json`, `refit_final.json`,
`external_final.json`, and `baseline_table_completed_20260920.json`.

The `eval500_eval_b*_s*.json` and `eval500_eval_g2u_s*.json` summaries supply the
main budget fit. The `sw` tags identify exploratory mask-swap runs and are not
part of the seven-budget fit. `phase_law_validation.json` contains retrospective
intermediate-budget checks. Layer summaries support the MATLAB curve script.

`external_uniform_g2.json`, `cocoglide_budget_curve.json`, and
`eval500_eval_g2_s42.json` preserve earlier configuration references needed by
external-evaluation parity checks. In particular, **MagicBrush values in
`external_uniform_g2.json` use obsolete luminance masks and must not be used as
current results**. The current evaluator uses
`magicbrush_corrected_reference_v20.json` and the corrected alpha masks.

Paths embedded in historical provenance records identify the original local
environment. They must be mapped to the user's data locations for inference.
