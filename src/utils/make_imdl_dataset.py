"""生成 IMDLBenCo 推理用的 INP-X 子集数据集 JSON (JsonDataset 格式)

格式: [[image_path, mask_path], ...]  mask_path="Negative" 表示真实图
输出: D:/lunwen/data/imdl_inpx_test.json (500 编辑对 + 300 real, 来自 train-data)
"""
import json
import random
from pathlib import Path

IMG_ROOT = Path("D:/lunwen/data/INP-X/inpainting_exchange")
OUT = Path("D:/lunwen/data/imdl_inpx_test.json")

idx = json.load(open("D:/lunwen/data/inpx_index.json"))
rng = random.Random(42)

# 编辑图: 取 train-data 的 standard (带 mask)
edits = [s for s in idx if s.get("split") == "train-data"
         and s.get("standard_path") and s.get("mask_path")]
rng.shuffle(edits)
edits = edits[:500]

records = []
missing = 0
for s in edits:
    img = IMG_ROOT / s["standard_path"]
    msk = IMG_ROOT / s["mask_path"]
    if img.exists() and msk.exists():
        records.append([str(img), str(msk)])
    else:
        missing += 1

# 真实图
reals = []
for cat in ["CelebAHQ", "CityScapes", "OpenImages", "SUN_RGBD"]:
    d = IMG_ROOT / "train-data" / "data" / "originals" / cat
    reals += sorted(d.glob("*.jpg"))
rng.shuffle(reals)
for p in reals[:300]:
    records.append([str(p), "Negative"])

rng.shuffle(records)
with open(OUT, "w") as f:
    json.dump(records, f, indent=1)
print(f"IMDLBenCo 数据集: {OUT}")
print(f"编辑对: {500 - missing}, real: {min(300, len(reals))}, 总: {len(records)}")
