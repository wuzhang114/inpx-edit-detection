"""结果溯源审计 (零训练): 对每个评测单元列出 checkpoint / 输入维度 / 参数 / 阈值 / 聚合 / seed
输出: outputs/provenance_audit.md + provenance_audit.json
"""
import json, glob, os, hashlib, re
from pathlib import Path
import torch

OUT = Path("D:/lunwen/outputs")
rows = []

# head 元数据: 输入维度(384/385) + 参数量
head_meta = {}
for hp in sorted(glob.glob(str(OUT / "weak_sup_v4_*_head.pt"))) + sorted(glob.glob(str(OUT / "*_head.pt"))):
    name = os.path.basename(hp)
    if name in head_meta:
        continue
    try:
        sd = torch.load(hp, map_location="cpu")
        if isinstance(sd, dict):
            w0 = None
            for k, v in sd.items():
                if k.endswith("0.weight") and hasattr(v, "shape") and v.dim() == 1:
                    w0 = int(v.shape[0]); break
            if w0 is None and "weight" in sd and sd["weight"].ndim == 2:
                w0 = int(sd["weight"].shape[1])
            # 只统计可训练参数语义: 排除 BN 的 running_mean/var/num_batches_tracked buffer
            nparam = sum(int(v.numel()) for k, v in sd.items()
                         if hasattr(v, "numel") and not any(t in k for t in
                         ("running_mean", "running_var", "num_batches_tracked")))
            head_meta[name] = {"in_dim": w0, "params": nparam, "params_note": "trainable-only (excl BN buffers)",
                               "sha256": hashlib.sha256(Path(hp).read_bytes()).hexdigest()}
    except Exception as e:
        head_meta[name] = {"error": str(e)[:60]}

for f in sorted(glob.glob(str(OUT / "eval500_eval_*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    tag = os.path.basename(f).replace("eval500_eval_", "").replace(".json", "")
    head = d.get("head", "")
    hm = head_meta.get(head, {})
    pi = d.get("per_image_fixed_miou")
    if isinstance(pi, list):
        pi = sum(pi) / len(pi) if pi else None
    seed_match = re.search(r"_s(\d+)$", tag)
    rows.append({
        "tag": tag,
        "head": head,
        "in_dim": hm.get("in_dim"),
        "params": hm.get("params"),
        "split": d.get("split"),
        "seed": d.get("seed"),
        "seed_from_tag": int(seed_match.group(1)) if seed_match else None,
        "seed_note": "tag is not proof of initialization reproducibility",
        "evaluation_sha256": hashlib.sha256(Path(f).read_bytes()).hexdigest(),
        "val_thr": d.get("val_thr"),
        "fixed_miou(micro)": d.get("fixed_miou"),
        "per_image_fixed_miou": pi,
        "pixel_auroc": d.get("pixel_auroc"),
        "det_auc_800": d.get("det_auc_800"),
    })

# 分类标记
def cls(tag):
    if tag.endswith("sw_s42") or "sw" in tag.split("_")[-1]:
        return "SWAP(对照)"
    if tag.startswith("clip"):
        return "CLIP"
    if tag.startswith("l1_384") or "linear" in tag:
        return "LINEAR"
    if tag.startswith("l4k") or "4k" in tag.lower():
        return "4K"
    if tag.startswith("g"):
        return "G-series(g0/g1/g2)"
    if tag.startswith("b") or "_b" in tag:
        return "BUDGET(b*)"
    return "OTHER"

lines = ["# 结果溯源审计 (零训练, 自动生成)", "",
         "| tag | 类别 | head | 输入维度 | 参数 | split | seed | val_thr | micro mIoU | per-image | pixel AUROC | det AUC(800) |",
         "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
for r in rows:
    lines.append("| {tag} | {c} | {head} | {in_dim} | {params} | {split} | {seed} | {val_thr} | {m} | {p} | {pa} | {da} |".format(
        tag=r["tag"], c=cls(r["tag"]), head=r["head"], in_dim=r["in_dim"], params=r["params"],
        split=str(r["split"])[:26], seed=r["seed"],
        val_thr=(round(r["val_thr"], 3) if isinstance(r["val_thr"], (int, float)) else r["val_thr"]),
        m=(round(r["fixed_miou(micro)"], 4) if isinstance(r["fixed_miou(micro)"], (int, float)) else None),
        p=(round(r["per_image_fixed_miou"], 4) if isinstance(r["per_image_fixed_miou"], (int, float)) else None),
        pa=(round(r["pixel_auroc"], 4) if isinstance(r["pixel_auroc"], (int, float)) else None),
        da=(round(r["det_auc_800"], 4) if isinstance(r["det_auc_800"], (int, float)) else None)))

# 混杂汇总
mixed = [r for r in rows if r["in_dim"] == 384]
lines += ["", f"## 输入维度混杂检查: 共 {len(rows)} 个评测单元, 其中 384 维(no_hp) {len(mixed)} 个",
          "384 维单元 tag: " + ", ".join(r["tag"] for r in mixed[:30]), "",
          "## head 元数据", ""]
for k, v in sorted(head_meta.items()):
    lines.append(f"- {k}: in_dim={v.get('in_dim')} params={v.get('params')}")

(OUT / "provenance_audit.md").write_text("\n".join(lines), encoding="utf-8")
json.dump({"rows": rows, "heads": head_meta}, open(OUT / "provenance_audit.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"评测单元: {len(rows)}; head 文件: {len(head_meta)}")
print(f"384维(no_hp)单元: {len(mixed)}")
for r in mixed[:12]:
    print("  ", r["tag"], "|", r["head"], "| in_dim=", r["in_dim"], "| params=", r["params"])
print("saved outputs/provenance_audit.md/.json")
