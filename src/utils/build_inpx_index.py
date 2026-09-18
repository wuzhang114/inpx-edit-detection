"""INP-X 数据集索引构建器(适配 Kaggle 实际结构)

实际目录结构:
  INP-X/inpainting_exchange/{train,test}-data/
    data/
      originals/{cat}/*.jpg                # 真实图池 (label=0)
      standard_inpainting/{cat}/{id}_{part}_{cat}_{model}.jpg   # 标准编辑 (label=1)
      inpainting_exchange/{cat}/{id}_{part}_{cat}_{model}_simple.jpg  # 背景保真 (label=1)
    masks/{cat}_masks/{id}_{part}.jpg      # 编辑 mask (与编辑图对应)
    masks/mask_sizes.csv                   # mask 面积元数据

编辑图命名: {id}_{part}_{cat}_{model}.jpg
  - standard 与 exchange 的差异仅在 _simple 后缀
  - mask 名 = {id}_{part}.jpg

生成 JSON 索引(每条 = 一个编辑样本):
  {
    split, cat, model,
    orig_id, part,
    standard_path, exchanged_path, mask_path,
    mask_ratio, size_class
  }
"""
import json
import argparse
import csv
from pathlib import Path

CATS = ["CelebAHQ", "CityScapes", "OpenImages", "SUN_RGBD"]
MODEL_SUFFIXES = ["Kandinsky_2_2", "OpenJourney", "StableDiffusion_v4"]


def parse_edit_name(name: str, cat: str = None):
    """解析编辑图文件名 -> (rest, model, is_simple)

    standard:  {rest}_{cat}_{model}.jpg
    exchange:  {rest}_{cat}_{model}_simple.jpg
    mask:      {rest}.jpg
    """
    stem = name[:-4]
    is_simple = stem.endswith("_simple")
    if is_simple:
        stem = stem[:-7]  # 去掉 _simple
    for m in MODEL_SUFFIXES:
        if stem.endswith("_" + m):
            rest = stem[: -(len(m) + 1)]
            break
    else:
        return None, None, is_simple
    if cat is not None and rest.endswith("_" + cat):
        rest = rest[: -(len(cat) + 1)]
    return rest, m, is_simple


def scan_inpx_dataset(root: str, output_index: str):
    root = Path(root)
    samples = []
    stats = {"real": 0, "standard": 0, "exchanged": 0, "mask": 0, "missing_mask": 0}

    for split_dir in sorted(root.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        data_dir = split_dir / "data"
        masks_dir = split_dir / "masks"
        if not data_dir.is_dir():
            continue

        # mask_sizes.csv -> {filename: {ratio, size_class}}
        mask_meta = {}
        csv_path = masks_dir / "mask_sizes.csv" if masks_dir.is_dir() else None
        if csv_path and csv_path.exists():
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    mask_meta[row["filename"]] = {
                        "ratio": float(row["mask_ratio"]),
                        "size_class": row["size_class"],
                    }

        std_dir = data_dir / "standard_inpainting"
        for cat in CATS:
            cat_std = std_dir / cat
            if not cat_std.is_dir():
                continue
            for f in sorted(cat_std.iterdir()):
                if not f.is_file():
                    continue
                rest, model, _ = parse_edit_name(f.name, cat)
                if rest is None:
                    continue
                mask_name = rest + ".jpg"
                ex_name = f"{rest}_{cat}_{model}_simple.jpg"
                ex_path = data_dir / "inpainting_exchange" / cat / ex_name
                mask_path = masks_dir / f"{cat}_masks" / mask_name if masks_dir.is_dir() else None

                sample = {
                    "split": split,
                    "cat": cat,
                    "model": model,
                    "orig_id_part": rest,
                    "standard_path": str(f.relative_to(root)),
                    "exchanged_path": str(ex_path.relative_to(root)) if ex_path.exists() else None,
                    "mask_path": str(mask_path.relative_to(root)) if (mask_path and mask_path.exists()) else None,
                    "mask_ratio": mask_meta.get(mask_name, {}).get("ratio"),
                    "size_class": mask_meta.get(mask_name, {}).get("size_class"),
                }
                samples.append(sample)
                stats["standard"] += 1
                if sample["exchanged_path"]:
                    stats["exchanged"] += 1
                if sample["mask_path"]:
                    stats["mask"] += 1
                else:
                    stats["missing_mask"] += 1

        # 真实图计数
        orig_dir = data_dir / "originals"
        for cat in CATS:
            if (orig_dir / cat).is_dir():
                stats["real"] += len(list((orig_dir / cat).iterdir()))

    with open(output_index, "w") as f:
        json.dump(samples, f, ensure_ascii=False, indent=1)

    print(f"索引构建完成: {len(samples)} 个编辑样本 -> {output_index}")
    print(f"  统计: 真实图 {stats['real']}, 标准编辑 {stats['standard']}, "
          f"交换后 {stats['exchanged']}, mask {stats['mask']} (缺失 {stats['missing_mask']})")
    by_split = {}
    for s in samples:
        by_split.setdefault(s["split"], {"standard": 0, "mask": 0})
        by_split[s["split"]]["standard"] += 1
        if s["mask_path"]:
            by_split[s["split"]]["mask"] += 1
    for k, v in by_split.items():
        print(f"  {k}: {v}")
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="D:/lunwen/data/INP-X/inpainting_exchange")
    parser.add_argument("--output", type=str, default="D:/lunwen/data/inpx_index.json")
    args = parser.parse_args()
    scan_inpx_dataset(args.root, args.output)
