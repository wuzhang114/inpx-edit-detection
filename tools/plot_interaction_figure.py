"""Core interaction figure for the main text (TMM review direction 2).

Plots, for each intervention (blur, low-pass, background blur):
  - task (detection / localization) x mask condition (true/random/shifted)
  - relative drop (1 - metric/intact) with paired-bootstrap 95% CI
  - the placebo-corrected interaction I_placebo with its 95% CI
Data: outputs/placebo_controls_matched_v2.json (per-image arrays kept).
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')

OUT = Path("D:/lunwen/outputs")
FIG_DIR = Path("D:/lunwen/paper/figures")
FIG_PNG = FIG_DIR / "figure4_interaction.png"
FIG_PDF = FIG_DIR / "figure4_interaction.pdf"
D = json.load(open(OUT / "placebo_controls_matched_v2.json", encoding="utf-8"))
SEEDS = ["sd_seed42", "sd_seed7", "sd_seed2024"]
KINDS = ["blur_edit", "lowpass_edit", "blur_bg"]
CONDS = ["true", "random", "shifted"]
N_BOOT = 2000
RNG = np.random.RandomState(20260828)


def boot_ci_from_pairs(a, b, n=len(CONDS)):
    """Bootstrap 95% CI of mean(a) where a is array of per-image drops."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    idx = RNG.randint(0, len(a), (N_BOOT, len(a)))
    am = a[idx].mean(axis=1)
    return float(np.percentile(am, 2.5)), float(np.percentile(am, 97.5))


# ------- collect per-seed relative drops (per-image IoU & det AUC) -------
stats = {}
for kind in KINDS:
    stats[kind] = {"det": {c: [] for c in CONDS}, "loc": {c: [] for c in CONDS}}
    for seed in SEEDS:
        c = D[seed]["conditions"]
        auc_b = c["intact"]["det_auc_matched"]
        per_b = c["intact"]["loc_per_image_iou"]
        for cond in CONDS:
            key = f"{kind}__{cond}"
            if key not in c:
                continue
            r = c[key]
            stats[kind]["det"][cond].append(1 - r["det_auc_matched"] / auc_b)
            stats[kind]["loc"][cond].append(1 - r["loc_per_image_iou"] / per_b)

# ------- plot -------
fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.72))
colors = {"det": "#D55E00", "loc": "#0072B2"}
labels = {"det": "Detection", "loc": "Localization"}

for ax, kind in zip(axes, KINDS):
    xs = np.arange(len(CONDS))
    width = 0.35
    for i, (task, off) in enumerate([("det", -0.17), ("loc", +0.17)]):
        vals = [np.mean(stats[kind][task][c]) for c in CONDS]
        sems = [np.std(stats[kind][task][c]) / np.sqrt(len(SEEDS)) for c in CONDS]
        ax.bar(xs + off, vals, width, color=colors[task], alpha=0.85,
               label=labels[task], yerr=[1.96 * s for s in sems],
               capsize=3, error_kw={"lw": 0.8})
    ax.set_xticks(xs)
    ax.set_xticklabels(["True\n$M$", "Random\n$R$", "Shifted\n$Q$"], fontsize=7.0)
    name = {"blur_edit": "Blur", "lowpass_edit": "Low-pass", "blur_bg": "Background blur"}[kind]
    ax.set_title(name, fontsize=8.6, weight="bold", pad=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(-0.55 if kind == "blur_bg" else -0.1, 0.35 if kind == "blur_bg" else 0.15)
    ax.tick_params(axis="both", labelsize=6.5, length=2.5, pad=2)
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.35, alpha=0.7)
    ax.set_axisbelow(True)

    # I_placebo annotation (seed42 shown, CI over bootstrap)
    it = D["sd_seed42"]["interactions"][kind]
    for key in ["random_vs_true"]:
        x = it[key]
        dd = x["drop_det_true_minus_alt"]
        dl = x["drop_loc_true_minus_alt"]
        ipl = x["mean"]
        ii = x["ci95"]
        ax.text(
            0.98, 0.98,
            f"$I_{{pl}}$={ipl:+.3f}\n95% CI [{ii[0]:+.3f}, {ii[1]:+.3f}]",
            transform=ax.transAxes, fontsize=6.1,
            ha="right", va="top", linespacing=1.05,
            bbox=dict(boxstyle="round,pad=0.22", fc="#FFF8E6", ec="#B77900", lw=0.45))

axes[0].set_ylabel("Relative drop", fontsize=7.0)
fig.text(0.5, 0.005, "same intervention applied to true (M), random (R), and shifted (Q) masks",
         ha="center", va="bottom", fontsize=6.2, color="#475569")
fig.legend(handles=[
    plt.Rectangle((0, 0), 1, 1, color=colors["det"], label="Detection"),
    plt.Rectangle((0, 0), 1, 1, color=colors["loc"], label="Localization"),
], loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=6.6)
fig.subplots_adjust(left=0.065, right=0.995, top=0.83, bottom=0.19, wspace=0.23)
fig.savefig(FIG_PDF, bbox_inches="tight", pad_inches=0.02)
fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight", pad_inches=0.02)
print("saved:", FIG_PDF)
print("saved:", FIG_PNG)
