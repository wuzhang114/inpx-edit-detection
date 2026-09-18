"""D5: 格式对齐规程

问题: 真图与编辑图可能来源于不同压缩格式/质量
  → 模型学到的是格式差异，而非AI生成痕迹
  → Fake or JPEG? (arXiv:2403.17608)

做法:
  1. 检查数据集中的格式分布
  2. 统一转换为 JPEG Q95 (或Q90)
  3. 统一分辨率
  4. 重抽特征对比前后差异
"""
import json
import argparse
from pathlib import Path
from collections import Counter

from PIL import Image
import numpy as np
from tqdm import tqdm


def analyze_format_distribution(index_file: str):
    """分析数据集的格式/分辨率分布,定位偏置"""
    with open(index_file) as f:
        samples = json.load(f)

    stats = {
        "real": {"ext": Counter(), "sizes": []},
        "standard": {"ext": Counter(), "sizes": []},
        "exchanged": {"ext": Counter(), "sizes": []},
    }

    for s in tqdm(samples, desc="分析格式"):
        kind = s.get("kind", "real")
        if kind not in stats:
            continue
        for img_key in ["real", "inpainted", "exchanged"]:
            if img_key not in s or s[img_key] is None:
                continue
            ext = Path(s[img_key]).suffix.lower()
            stats[kind]["ext"][ext] += 1

    print("\n=== 格式分布 ===")
    for kind, info in stats.items():
        print(f"\n{kind}:")
        print(f"  扩展名: {dict(info['ext'])}")

    return stats


def format_align_image(
    img_path: str,
    output_path: str,
    target_format: str = "JPEG",
    quality: int = 95,
    target_size: int = None,
):
    """单张图像格式对齐: 统一格式+质量+分辨率 (损坏文件返回 False)"""
    try:
        img = Image.open(img_path).convert("RGB")
        if target_size is not None:
            img = img.resize((target_size, target_size), Image.LANCZOS)
        img.save(output_path, format=target_format, quality=quality)
        return True
    except OSError as e:
        print(f"  [跳过损坏文件] {img_path}: {e}")
        return False


def format_align_dataset(
    root: str,
    index_file: str,
    output_root: str,
    target_format: str = "JPEG",
    quality: int = 95,
    target_size: int = 512,
    limit: int = 0,
    seed: int = 42,
):
    """对 INP-X 全量数据做格式对齐 (适配 inpx_index.json 真实 schema)

    索引字段: standard_path / exchanged_path / mask_path (相对路径)
    真实图: 在 root/<split>/data/originals/<cat>/ 下扫描
    保留原目录结构,统一输出为 target_format @ quality @ target_size。
    同时生成对齐后的新索引 (与 extract_features.py 兼容)。
    """
    import random
    root = Path(root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with open(index_file) as f:
        samples = json.load(f)

    if limit and limit < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, limit)

    aligned = []
    n_ok = 0
    for s in tqdm(samples, desc="格式对齐"):
        new_s = {**s}
        ok = True
        for img_key, src_key in [("standard_path", "standard_path"),
                                 ("exchanged_path", "exchanged_path"),
                                 ("mask_path", "mask_path")]:
            src_rel = s.get(src_key)
            if not src_rel:
                continue
            src_path = root / src_rel
            if not src_path.exists():
                ok = False
                continue
            # 统一为 JPEG
            new_rel = Path(src_rel).with_suffix(".jpg")
            dst_path = output_root / new_rel
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            ok_img = format_align_image(str(src_path), str(dst_path),
                                        target_format, quality, target_size)
            if not ok_img:
                ok = False
            new_s[img_key] = str(new_rel)
        if ok:
            aligned.append(new_s)
            n_ok += 1

    # 真实图: 扫描 originals 目录 (与 InpxDataset 一致)
    n_real = 0
    for split_name in sorted({s["split"] for s in samples}):
        for cat in ["CelebAHQ", "CityScapes", "OpenImages", "SUN_RGBD"]:
            orig_dir = root / split_name / "data" / "originals" / cat
            if not orig_dir.is_dir():
                continue
            files = sorted(orig_dir.iterdir())
            if limit:
                rng = random.Random(seed + n_real)
                files = rng.sample(files, min(len(files), limit))
            for p in tqdm(files, desc=f"对齐真实图 {split_name}/{cat}", leave=False):
                if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                rel = Path(split_name) / "data" / "originals" / cat / p.name
                dst = output_root / rel.with_suffix(".jpg")
                dst.parent.mkdir(parents=True, exist_ok=True)
                ok_real = format_align_image(str(p), str(dst), target_format, quality, target_size)
                if ok_real:
                    n_real += 1

    # 保存新 index
    new_index = output_root / "index_aligned.json"
    with open(new_index, "w") as f:
        json.dump(aligned, f, ensure_ascii=False, indent=1)

    print(f"\n格式对齐完成! 编辑图 {n_ok}/{len(samples)} 样本, 真实图 {n_real} 张 -> {output_root}")
    print(f"新索引: {new_index}")
    return str(new_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    p_analyze = sub.add_parser("analyze")
    p_analyze.add_argument("--index", type=str, required=True,
                           help="inpx_index.json 路径")

    p_align = sub.add_parser("align")
    p_align.add_argument("--root", type=str, required=True,
                         help="原始图像根目录")
    p_align.add_argument("--index", type=str, required=True,
                         help="原始 index JSON")
    p_align.add_argument("--output", type=str, required=True,
                         help="对齐后输出目录")
    p_align.add_argument("--quality", type=int, default=95)
    p_align.add_argument("--size", type=int, default=512)
    p_align.add_argument("--limit", type=int, default=0,
                         help="子集大小 (0=全部)")

    args = parser.parse_args()

    if args.command == "analyze":
        analyze_format_distribution(args.index)
    elif args.command == "align":
        format_align_dataset(args.root, args.index, args.output,
                             quality=args.quality, target_size=args.size,
                             limit=args.limit)
    else:
        parser.print_help()
