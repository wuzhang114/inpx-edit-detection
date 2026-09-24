"""Historical MagicBrush extraction using luminance-derived masks.

For the manuscript results, use tools/repair_magicbrush_alpha_masks_v20.py
to derive masks from the original RGBA alpha channel (alpha != 255).
"""
import io
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

OUT = Path("D:/lunwen/data/magicbrush_eval")
(OUT / "images").mkdir(parents=True, exist_ok=True)
(OUT / "masks").mkdir(parents=True, exist_ok=True)
(OUT / "sources").mkdir(parents=True, exist_ok=True)

n = 0
for i in range(1, 5):
    t = pq.read_table(f"D:/lunwen/data/magicbrush_dev_{i}.parquet")
    for row in t.to_pylist():
        sid = f"{i:02d}_{row['img_id']}_{row['turn_index']}"
        def dec(v):
            if isinstance(v, dict) and v.get("bytes"):
                return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
            return Image.open(v["path"]).convert("RGB") if isinstance(v, dict) else None
        target = dec(row["target_img"])
        mask = dec(row["mask_img"]).convert("L")
        source = dec(row["source_img"])
        if target is None:
            continue
        target.save(OUT / "images" / f"{sid}.png")
        mask.save(OUT / "masks" / f"{sid}.png")
        if source is not None and n < 300:
            source.save(OUT / "sources" / f"{sid}.png")
        n += 1
print(f"extracted: {n} edits (sources: {len(list((OUT/'sources').glob('*.png')))})")
