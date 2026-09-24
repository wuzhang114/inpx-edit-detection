"""生成方法×数据集矩阵图 (检测/定位), 输出 paper/figures/"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATASETS = ["INP-X 500", "CelebAHQ", "CocoGlide", "MagicBrush", "SDXL-500"]

# ---- 检测矩阵 ----
det_rows = [
    ("v4 (ours)",   [0.924, 0.808, 0.732, 0.644, 0.658], True),
    ("FLAME",       [0.601, 0.004, 0.035, 0.743, 0.160], False),
    ("MVSS",        [0.464, 0.486, 0.537, 0.534, 0.302], False),
]
det_note = ("AUC where score ranking is available; FLAME CelebAHQ/CocoGlide/SDXL are fixed-threshold recall (AP<0.001, R-marked in the paper); "
            "MVSS: our 128px protocol.")

# ---- 定位矩阵 ----
loc_rows = [
    ("v4 (ours)",   [0.529, 0.293, 0.527, 0.064, 0.076], True),
    ("FLAME-LAD",   [0.128, None,  0.455, 0.259, None],  False),
    ("FLAME full",  [0.140, None,  0.486, None,  None],  False),
    ("ObjectFormer",[0.232, 0.214, 0.402, 0.382, 0.125], False),
    ("MVSS",        [0.113, 0.166, 0.482, 0.384, 0.134], False),
    ("IML-ViT",     [0.145, 0.103, 0.272, 0.379, 0.190], False),
    ("TRAIL",       [0.003, None,  None,  None,  None],  False),
    ("DinoLizer",   [0.260, None,  None,  None,  None],  False),
]
loc_note = ("v4: global micro mIoU at the fixed INP-X validation threshold (37x37 grid); IMDL baselines: INP-X column is the 512px official per-image best-thr "
            "mean (0.232/0.113/0.145, matching the paper main table); other dataset columns are 128px approximations; "
            "FLAME: official protocol; TRAIL: INP-X mIoU 0.003 (ViT-B/14, its CocoGlide patch AUROC 0.736 is reported in the CocoGlide table); "
            "DinoLizer: fixed 0.5 threshold.")


def plot_matrix(rows, fname, title, note, fmt=".3f", cmap="RdYlGn"):
    n_methods, n_ds = len(rows), len(DATASETS)
    fig, ax = plt.subplots(figsize=(11, 0.9 * n_methods + 1.6))
    data = np.full((n_methods, n_ds), np.nan)
    for i, (name, vals, _) in enumerate(rows):
        for j, v in enumerate(vals):
            if v is not None:
                data[i, j] = v
    vmin, vmax = 0.0, 1.0
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n_ds))
    ax.set_xticklabels(DATASETS, fontsize=11)
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels([r[0] for r in rows], fontsize=11)
    for i, (name, vals, is_ours) in enumerate(rows):
        for j, v in enumerate(vals):
            txt = "—" if v is None else f"{v:{fmt}}"
            color = "white" if v is not None and (v < 0.25 or v > 0.85) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10.5, color=color,
                    fontweight="bold" if is_ours else "normal")
    # Highlight the proposed readout in the comparison table.
    for i, (_, _, is_ours) in enumerate(rows):
        if is_ours:
            ax.add_patch(plt.Rectangle((-.5, i-.5), n_ds, 1, fill=False,
                                       edgecolor="blue", linewidth=3))
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xticks(np.arange(-.5, n_ds, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_methods, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="x", rotation=15)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("value (higher = better)", fontsize=10)
    fig.text(0.02, 0.01, note, fontsize=8, color="gray", wrap=True)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", fname)


plot_matrix(det_rows, "D:/lunwen/paper/figures/det_matrix.png", "Detection Matrix (AUC / recall)", det_note)
plot_matrix(loc_rows, "D:/lunwen/paper/figures/loc_matrix.png", "Localization Matrix (mIoU)", loc_note)
