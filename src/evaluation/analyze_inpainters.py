"""3-inpainter 分析: 按 inpainter (Kandinsky_2_2 / OpenJourney / StableDiffusion_v4) 分组评测

用途: leave-one-out 泛化实验的诊断版 — 用已训练的头, 看哪种 inpainter 最难/最容易,
为正式 leave-one-out 训练提供预期, 也是论文"编辑器泛化"章节的基础数据。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.weakly_supervised_v2 import MLPHead

CACHE = "D:/lunwen/data/features_cache"
OUT = Path("D:/lunwen/outputs")


def main():
    with open(f"{CACHE}/labels.json") as f:
        labels = json.load(f)
    with open(f"{CACHE}/metadata.json") as f:
        meta = json.load(f)
    n_total, n_patches, dino_dim = meta["n_total"], meta["n_patches"], meta["dino_dim"]

    dino_mm = np.memmap(f"{CACHE}/dino_tokens.npy", dtype=np.float16, mode="r",
                        shape=(n_total, n_patches, dino_dim))
    hp_mm = np.memmap(f"{CACHE}/highpass.npy", dtype=np.float32, mode="r",
                      shape=(n_total, n_patches, 1))

    # 用 v2 头 (检测主线) 和 v3 头 (定位主线)
    heads = {}
    for name in ["weak_sup_head", "weak_sup_v3_head"]:
        p = OUT / f"{name}.pt"
        if p.exists():
            h = MLPHead(dino_dim + 1).cuda()
            h.load_state_dict(torch.load(p))
            h.eval()
            heads[name] = h
            print(f"加载 {name}")
        else:
            print(f"跳过 {name} (不存在)")

    # 训练域 (非 CelebAHQ) 的 fake, 按 inpainter 分组 (与 real 对比)
    train_idxs = [i for i, l in enumerate(labels) if l.get("cat") != "CelebAHQ"
                  and l["label"] == 1]
    real_idxs = [i for i, l in enumerate(labels) if l["label"] == 0]
    rng = np.random.RandomState(0)
    real_sample = sorted(rng.choice(real_idxs, min(len(real_idxs), 6000), replace=False))
    models = sorted({l.get("model") for l in labels if l.get("model")})
    print(f"训练域 fake: {len(train_idxs)}, real 抽样: {len(real_sample)}")
    print(f"inpainter 列表: {models}")

    results = {}
    for mname, head in heads.items():
        scores_all = []
        with torch.no_grad():
            for i in range(0, len(train_idxs), 256):
                b_idx = train_idxs[i:i + 256]
                dino = torch.from_numpy(dino_mm[b_idx].astype(np.float32))
                hp = torch.from_numpy(hp_mm[b_idx].astype(np.float32))
                X = torch.cat([dino, hp], dim=-1).cuda()
                logits = head(X)
                s = torch.sigmoid(logits).max(dim=1).values.cpu().numpy()
                scores_all.append(s)
        scores_train = np.concatenate(scores_all)
        # real 分数
        real_scores = []
        with torch.no_grad():
            for i in range(0, len(real_sample), 256):
                b_idx = real_sample[i:i + 256]
                dino = torch.from_numpy(dino_mm[b_idx].astype(np.float32))
                hp = torch.from_numpy(hp_mm[b_idx].astype(np.float32))
                X = torch.cat([dino, hp], dim=-1).cuda()
                logits = head(X)
                s = torch.sigmoid(logits).max(dim=1).values.cpu().numpy()
                real_scores.append(s)
        real_scores = np.concatenate(real_scores)

        print(f"\n=== {mname} (训练域, 按 inpainter) ===")
        med = np.median(np.concatenate([scores_train, real_scores]))
        per_model = {}
        for m in models:
            m_mask = np.array([labels[i].get("model") == m for i in train_idxs])
            if m_mask.sum() < 10:
                continue
            s_fake = scores_train[m_mask]
            auc_m = roc_auc_score(
                np.concatenate([np.zeros(len(real_scores)), np.ones(len(s_fake))]),
                np.concatenate([real_scores, s_fake]))
            facc = (s_fake >= med).mean()
            print(f"  {m}: n={m_mask.sum()} AUC(real vs {m})={auc_m:.4f} F.Acc={facc:.4f}")
            per_model[m] = {"auc": float(auc_m), "facc": float(facc), "n": int(m_mask.sum())}
        results[mname] = per_model

    with open(OUT / "inpainter_analysis.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"\n已保存: {OUT / 'inpainter_analysis.json'}")


if __name__ == "__main__":
    main()
