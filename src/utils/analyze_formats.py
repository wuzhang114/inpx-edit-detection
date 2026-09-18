"""D5 格式分布诊断: 统计 INP-X 数据集真实图 vs 编辑图的格式/分辨率/质量

论文素材: Fake or JPEG? 偏置审计。对 inpx_index.json 的编辑图 + originals 真实图
统计: 扩展名、分辨率、JPEG 质量估计、文件大小。
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

IMG_ROOT = Path("D:/lunwen/data/INP-X/inpainting_exchange")


def img_stats(p: Path):
    try:
        im = Image.open(p)
        fmt = im.format
        size = im.size
        q = None
        if fmt == "JPEG" and "quality" in im.info:
            q = im.info["quality"]
        return fmt, size, q, p.stat().st_size
    except OSError:
        return "BROKEN", None, None, 0


def main():
    idx = json.load(open("D:/lunwen/data/inpx_index.json"))
    rng = __import__("random").Random(0)
    sample = rng.sample(idx, 2000)

    stats = {
        "real": Counter(), "standard": Counter(), "exchanged": Counter(),
    }
    sizes = {"real": [], "standard": [], "exchanged": []}
    quals = {"real": [], "standard": [], "exchanged": []}
    broken = Counter()

    # 编辑图
    for s in sample:
        for key, kind in [("standard_path", "standard"), ("exchanged_path", "exchanged")]:
            rel = s.get(key)
            if not rel:
                continue
            p = IMG_ROOT / rel
            fmt, size, q, _ = img_stats(p)
            if fmt == "BROKEN":
                broken[kind] += 1
                continue
            stats[kind][fmt] += 1
            sizes[kind].append(size)
            if q:
                quals[kind].append(q)

    # 真实图 (从 originals 抽样)
    real_sample = []
    for split in ["train-data", "test-data"]:
        for cat in ["CelebAHQ", "CityScapes", "OpenImages", "SUN_RGBD"]:
            d = IMG_ROOT / split / "data" / "originals" / cat
            files = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
            real_sample += files
    rng.shuffle(real_sample)
    for p in real_sample[:2000]:
        fmt, size, q, _ = img_stats(p)
        if fmt == "BROKEN":
            broken["real"] += 1
            continue
        stats["real"][fmt] += 1
        sizes["real"].append(size)
        if q:
            quals["real"].append(q)

    print("=== 格式分布 (抽样) ===")
    for kind in ["real", "standard", "exchanged"]:
        print(f"{kind}: {dict(stats[kind])} 损坏={broken[kind]}")

    print("\n=== 分辨率分布 ===")
    for kind in ["real", "standard", "exchanged"]:
        cnt = Counter(sizes[kind])
        top = cnt.most_common(5)
        print(f"{kind}: {top}")

    print("\n=== JPEG 质量 (EXIF quality 字段) ===")
    for kind in ["real", "standard", "exchanged"]:
        qs = quals[kind]
        if qs:
            import numpy as np
            qa = np.array(qs)
            print(f"{kind}: n={len(qa)} mean={qa.mean():.1f} "
                  f"p25={np.percentile(qa,25):.0f} p50={np.percentile(qa,50):.0f} "
                  f"p75={np.percentile(qa,75):.0f}")
        else:
            print(f"{kind}: 无质量字段 (PNG?)")

    print("\n=== 文件大小 (KB) ===")
    for kind in ["real", "standard", "exchanged"]:
        pass
    # 直接对比三类文件大小
    sizes_kb = {"real": [], "standard": [], "exchanged": []}
    for s in sample:
        for key, kind in [("standard_path", "standard"), ("exchanged_path", "exchanged")]:
            rel = s.get(key)
            if not rel:
                continue
            p = IMG_ROOT / rel
            if p.exists():
                sizes_kb[kind].append(p.stat().st_size / 1024)
    for p in real_sample[:2000]:
        if p.exists():
            sizes_kb["real"].append(p.stat().st_size / 1024)
    import numpy as np
    for kind, arr in sizes_kb.items():
        a = np.array(arr)
        print(f"{kind}: n={len(a)} median={np.median(a):.0f} KB p90={np.percentile(a,90):.0f} KB")


if __name__ == "__main__":
    main()
