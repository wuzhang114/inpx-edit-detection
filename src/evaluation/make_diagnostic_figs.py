"""Historical diagnostic plots for exploratory readout configurations.

Plots recorded mask-size summaries, directional-feature ablations, and image
format distributions. These plots are not the current manuscript figures.
Unavailable linear-probe values are omitted rather than plotted as zeros."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 注册中文字体 (Windows 微软雅黑)
for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc",
           r"C:\Windows\Fonts\simhei.ttf"]:
    try:
        fm.fontManager.addfont(fp)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("D:/lunwen/figures_paper")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ============ 图1: mask 分档失效图 ============
def fig1_mask_failure():
    # 检测 (跨域 CelebAHQ, F.Acc by mask size)
    det = {
        "弱监督v2 (检测)": {"小": 71.2, "中": 82.0, "大": 86.3},
    }
    # 定位 (mIoU by mask size, 跨域 CelebAHQ)
    loc = {
        "方向VAE": {"小": 0.054, "中": 0.127, "大": 0.335},
        "弱监督v2": {"小": 0.153, "中": 0.218, "大": 0.338},
        "弱监督v3": {"小": 0.114, "中": 0.173, "大": 0.418},
    }
    sizes = ["小(2-5%)", "中(5-15%)", "大(>15%)"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    x = np.arange(len(sizes))
    w = 0.25
    for i, (k, v) in enumerate(det.items()):
        ax.bar(x + i * w, [v[s] for s in ["小", "中", "大"]], w, label=k)
    ax.set_xticks(x + w)
    ax.set_xticklabels(sizes)
    ax.set_ylabel("检测 F.Acc (%)")
    ax.set_title("图像级检测: 跨域 CelebAHQ, 按 mask 面积")
    ax.legend()

    ax = axes[1]
    x = np.arange(len(sizes))
    w = 0.25
    for i, (k, v) in enumerate(loc.items()):
        ax.bar(x + i * w, [v[s] for s in ["小", "中", "大"]], w, label=k)
    ax.set_xticks(x + w)
    ax.set_xticklabels(sizes)
    ax.set_ylabel("定位 mIoU")
    ax.set_title("像素级定位: 按 mask 面积")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig1_mask_failure.png", bbox_inches="tight")
    print("fig1 已保存")


# ============ 图2: 方向性 VAE 消融 ============
def fig2_directional_ablation():
    # (变体, 全局AUC, real-vs-exchanged AUC)
    data = [
        ("低尾 5桶", 0.487, 0.498),
        ("高尾 5桶", 0.517, 0.542),
        ("双侧 5桶", 0.523, 0.538),
        ("高尾 DINO桶", 0.492, 0.497),
        ("高尾 20桶", 0.538, 0.557),
        ("高尾 48桶", 0.544, 0.558),
    ]
    names = [d[0] for d in data]
    auc_global = [d[1] for d in data]
    auc_exc = [d[2] for d in data]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names))
    ax.bar(x - 0.18, auc_global, 0.36, label="全局 AUC")
    ax.bar(x + 0.18, auc_exc, 0.36, label="real vs exchanged AUC")
    ax.axhline(0.5, color="red", ls="--", lw=1, label="随机")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.45, 0.6)
    ax.set_title("方向性 VAE 高尾信号: 条件化方式与桶数消融")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig2_directional_ablation.png", bbox_inches="tight")
    print("fig2 已保存")


# ============ 图3: 格式偏置 ============
def fig3_format_bias():
    # 从对齐日志/已知统计: real 混合分辨率, 编辑统一 512
    sizes_kb = {
        "real": np.array([63] * 50 + [492] * 5 + [120] * 20),
        "standard": np.array([74] * 70 + [101] * 10),
        "exchanged": np.array([69] * 70 + [94] * 10),
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot([sizes_kb["real"], sizes_kb["standard"], sizes_kb["exchanged"]],
                    labels=["real", "standard", "exchanged"])
    ax.set_ylabel("文件大小 (KB, 示意分布)")
    ax.set_title("格式偏置: 真实图分辨率多样 vs 编辑图统一 512×512")
    ax.annotate("编辑图全部 512×512\n(统一重采样)",
                xy=(2, 90), xytext=(1.1, 250),
                arrowprops=dict(arrowstyle="->"), fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_format_bias.png", bbox_inches="tight")
    print("fig3 已保存")


if __name__ == "__main__":
    fig1_mask_failure()
    fig2_directional_ablation()
    fig3_format_bias()
    print(f"全部保存到 {OUT}")
