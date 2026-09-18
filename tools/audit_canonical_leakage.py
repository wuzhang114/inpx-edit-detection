"""canonical 泄漏审计:旧 split 复现 + master split 复核(五项)。

输出: outputs/audit_canonical_leakage.json
  - old_split  : split_sd_seed42/7/2024 的(train∩val, test_edit_leak, test_real_leak)组级与样本级计数
  - master      : split_budget_master.json 五项验证复核(组级+样本级)
  - sun_coverage: SUN canonical 与 real 命名空间的匹配覆盖率
  - excluded    : imdl 测试 800 映射与组内规模
用法: python tools/audit_canonical_leakage.py
"""
import json
import os
import sys
import io
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data.canonical_source import canonical_source_id  # noqa: E402

CACHE = "D:/lunwen/data/features_cache"
DATA_JSON = "D:/lunwen/data/imdl_inpx_test.json"
OUT = "D:/lunwen/outputs"
OLD_SPLITS = ["split_sd_seed42.json", "split_sd_seed7.json", "split_sd_seed2024.json"]
MASTER = f"{OUT}/split_budget_master.json"


def load_all():
    labels = json.load(open(f"{CACHE}/labels.json", encoding="utf-8"))
    cids = [canonical_source_id(l["path"], l.get("cat")) for l in labels]
    return labels, cids


def group_sets(cids, idxs):
    return set(cids[i] for i in idxs)


def audit_old_split(labels, cids):
    labels_arr = [l["label"] for l in labels]
    res = {}
    for sf in OLD_SPLITS:
        p = os.path.join(OUT, sf)
        if not os.path.exists(p):
            continue
        sp = json.load(open(p, encoding="utf-8"))
        tr, va = group_sets(cids, sp["train_idx"]), group_sets(cids, sp["val_idx"])
        trv = tr | va
        # 测试集 imdl
        recs = json.load(open(DATA_JSON, encoding="utf-8"))
        img_root = "D:/lunwen/data/INP-X/inpainting_exchange"
        path2idx = {l["path"].replace("\\", "/"): i for i, l in enumerate(labels)}
        edit_ids, real_ids = [], []
        for img_path, mask_path in recs:
            key = str(Path(img_path).relative_to(img_root)).replace("\\", "/")
            idx = path2idx.get(key)
            if idx is None:
                continue
            (real_ids if mask_path == "Negative" else edit_ids).append(idx)
        edit_leak = [i for i in edit_ids if cids[i] in trv]
        real_leak = [i for i in real_ids if cids[i] in trv]
        res[sf] = {
            "train_n": len(sp["train_idx"]), "val_n": len(sp["val_idx"]),
            "train_val_group_overlap": len(tr & va),
            "test_edit_n": len(edit_ids),
            "test_edit_leak_group": len(set(cids[i] for i in edit_leak)),
            "test_edit_leak_sample": len(edit_leak),
            "test_real_n": len(real_ids),
            "test_real_leak_group": len(set(cids[i] for i in real_leak)),
            "test_real_leak_sample": len(real_leak),
        }
        # 分域统计
        by = defaultdict(lambda: [0, 0])
        for i in edit_leak:
            d = labels[i]["cat"]
            by[d][0] += 1
        for i in real_leak:
            d = labels[i]["cat"]
            by[d][1] += 1
        res[sf]["leak_by_domain"] = {d: {"edit": a, "real": b} for d, (a, b) in by.items()}
    return res


def audit_master(labels, cids):
    sp = json.load(open(MASTER, encoding="utf-8"))
    sets = {
        "train": (set(sp["train_groups"]), set(sp["train_idx"])),
        "val": (set(sp["val_groups"]), set(sp["val_idx"])),
        "test": (set(sp["test_groups"]), set(sp["test_idx"])),
        "excluded": (set(sp["excluded_groups"]), set(sp["excluded_idx"])),
    }
    checks = {}
    for a, b in [("train", "val"), ("train", "test"), ("train", "excluded"),
                 ("val", "excluded"), ("test", "excluded")]:
        ga, ia = sets[a]
        gb, ib = sets[b]
        checks[f"{a}_cap_{b}"] = {
            "group_overlap": len(ga & gb),
            "sample_overlap": len(ia & ib),
            "ok": len(ga & gb) == 0 and len(ia & ib) == 0,
        }
    # 全部 idx 与组一致性(每个 idx 的 canonical ∈ 所属 split 组集)
    per_split_consistent = {}
    for name, (gs, idxs) in sets.items():
        bad = sum(1 for i in idxs if cids[i] not in gs)
        per_split_consistent[name] = {"n_idx": len(idxs), "inconsistent": bad}
    return {"checks": checks, "per_split_consistent": per_split_consistent,
            "all_ok": all(v["ok"] for v in checks.values())
            and all(v["inconsistent"] == 0 for v in per_split_consistent.values())}


def sun_coverage(labels):
    """SUN 编辑 canonical 与 SUN real 名称空间的匹配覆盖率。"""
    real_names = {os.path.basename(l["path"]).replace("\\", "/"):
                  canonical_source_id(l["path"], l["cat"]) for l in labels
                  if l.get("cat") == "SUN_RGBD" and l.get("label") == 0}
    # 编辑 canonical 集合
    edit_cids = set()
    for l in labels:
        if l.get("cat") == "SUN_RGBD" and l.get("label") == 1:
            edit_cids.add(canonical_source_id(l["path"], l["cat"]))
    # 编辑 canonical 可归因(在 real 名称空间出现,或 5 位数字 real 不属于编辑命名空间)
    # 判定:编辑 canonical 去掉前缀后,是否等于任一长名 real 的 canonical
    long_real = set(c for c in real_names.values() if not c.endswith("__real_num__") and not c.endswith(':__real_num__' + '00000') if '__real_num__' not in c)
    long_real = {c for c in long_real if '__real_num__' not in c}
    matched = sum(1 for c in edit_cids if c in long_real)
    return {"sun_edit_groups": len(edit_cids), "sun_long_real_groups": len(long_real),
            "match_ratio": round(matched / max(len(edit_cids), 1), 4),
            "matched": matched, "unmatched": len(edit_cids) - matched,
            "notes": "5 位数字 SUN real(独立命名空间)不计入匹配分母;匹配 = 编辑 canonical 与长名 real canonical 一致"}


def main():
    labels, cids = load_all()
    doc = {
        "old_split": audit_old_split(labels, cids),
        "master": audit_master(labels, cids),
        "sun_coverage": sun_coverage(labels),
    }
    outp = os.path.join(OUT, "audit_canonical_leakage.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(json.dumps(doc, ensure_ascii=False, indent=1)[:4000])
    print("saved:", outp)


if __name__ == "__main__":
    main()
