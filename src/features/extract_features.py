"""特征提取与缓存流水线 -- 适配 INP-X Kaggle 索引

对 INP-X 数据(真实图 + 标准编辑 + 背景保真)预先抽取:
  (a) DINOv2 patch tokens (float16 存储,节省磁盘)
  (b) 交叉差分高通残差
  (c) SD VAE 重建残差 (可选)
  (d) 局部纹理能量 (内容条件归一化用)

之后所有实验从缓存读取。
"""
import os
import sys
import json
import argparse
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.highpass import cross_diff_highpass, pool_to_patch_grid, local_texture_energy


def load_dinov2(model_name: str = "dinov2_vits14"):
    model = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True)
    model = model.cuda().eval()
    return model


def load_sd_vae():
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float16
    )
    vae = vae.cuda().eval()
    return vae


class InpxDataset(Dataset):
    """从 inpx_index.json 构建:real + standard + exchange 三路样本"""

    def __init__(self, root: str, index_file: str, splits=("train-data",), image_size: int = 518,
                 include_real: bool = True, include_standard: bool = True,
                 include_exchanged: bool = True):
        self.root = Path(root)
        self.image_size = image_size
        with open(index_file) as f:
            idx = json.load(f)
        self.samples = []
        for s in idx:
            if s["split"] not in splits:
                continue
            if include_standard and s["standard_path"]:
                self.samples.append({
                    "path": s["standard_path"], "label": 1, "kind": "standard",
                    "mask_path": s["mask_path"], "mask_ratio": s["mask_ratio"],
                    "size_class": s["size_class"], "cat": s["cat"], "model": s["model"],
                })
            if include_exchanged and s["exchanged_path"]:
                self.samples.append({
                    "path": s["exchanged_path"], "label": 1, "kind": "exchanged",
                    "mask_path": s["mask_path"], "mask_ratio": s["mask_ratio"],
                    "size_class": s["size_class"], "cat": s["cat"], "model": s["model"],
                })
        if include_real:
            for split_name in splits:
                for cat in ["CelebAHQ", "CityScapes", "OpenImages", "SUN_RGBD"]:
                    orig_dir = self.root / split_name / "data" / "originals" / cat
                    if not orig_dir.is_dir():
                        continue
                    for p in sorted(orig_dir.iterdir()):
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                            self.samples.append({
                                "path": str(p.relative_to(self.root)), "label": 0, "kind": "real",
                                "mask_path": None, "mask_ratio": None, "size_class": None,
                                "cat": cat, "model": None,
                            })

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(self.root / sample["path"]).convert("RGB")
        return {
            "img": self.transform(img),
            "path": sample["path"],
            "label": sample["label"],
            "kind": sample["kind"],
            "mask_path": sample["mask_path"] if sample["mask_path"] else "",
            "mask_ratio": sample["mask_ratio"] if sample["mask_ratio"] is not None else -1.0,
            "size_class": sample["size_class"] if sample["size_class"] is not None else "unknown",
            "cat": sample["cat"],
            "model": sample["model"] if sample["model"] is not None else "none",
        }


@torch.no_grad()
def extract_features(data_root: str, index_file: str, output_dir: str,
                     splits, dinov2_model: str = "dinov2_vits14", use_vae: bool = False,
                     image_size: int = 518, batch_size: int = 8, num_workers: int = 2):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载 DINOv2: {dinov2_model}")
    backbone = load_dinov2(dinov2_model)
    patch_size = 14 if "14" in dinov2_model else 16
    num_patches_per_side = image_size // patch_size
    n_patches = num_patches_per_side ** 2

    vae = None
    if use_vae:
        print("加载 SD VAE...")
        vae = load_sd_vae()

    dataset = InpxDataset(data_root, index_file, splits=splits, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    n_total = len(dataset)
    print(f"共 {n_total} 张图像, split={splits}, image_size={image_size}, patches={n_patches}")

    dino_dim = backbone.embed_dim
    dino_file = output_dir / "dino_tokens.npy"
    hp_file = output_dir / "highpass.npy"
    tex_file = output_dir / "texture_energy.npy"
    labels_file = output_dir / "labels.json"

    dino_memmap = np.memmap(str(dino_file), dtype=np.float16, mode="w+",
                            shape=(n_total, n_patches, dino_dim))
    hp_memmap = np.memmap(str(hp_file), dtype=np.float32, mode="w+",
                          shape=(n_total, n_patches, 1))
    tex_memmap = np.memmap(str(tex_file), dtype=np.float32, mode="w+",
                           shape=(n_total, n_patches))

    vae_memmap = None
    vae_file = None
    if use_vae:
        vae_file = output_dir / "vae_residual.npy"
        vae_memmap = np.memmap(str(vae_file), dtype=np.float32, mode="w+",
                               shape=(n_total, n_patches, 1))

    offset = 0
    meta = []
    pbar = tqdm(total=n_total, desc="抽取特征")

    for batch in loader:
        imgs = batch["img"].cuda()
        b = imgs.shape[0]

        tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]
        dino_memmap[offset:offset + b] = tokens.cpu().to(torch.float16).numpy()

        hp = cross_diff_highpass(imgs)
        hp = torch.nn.functional.pad(hp, (0, 1, 0, 1))
        hp_pooled = pool_to_patch_grid(hp, n_patches)
        hp_memmap[offset:offset + b] = hp_pooled.cpu().numpy()

        tex = local_texture_energy(imgs, patch_size=patch_size)
        tex = tex.reshape(b, -1)
        tex_memmap[offset:offset + b] = tex.cpu().numpy()

        if vae is not None and vae_memmap is not None:
            with torch.autocast("cuda", dtype=torch.float16):
                vae_in = torch.nn.functional.interpolate(
                    imgs.half(), size=(512, 512), mode="bilinear", align_corners=False
                )
                posterior = vae.encode(vae_in).latent_dist
                rec = vae.decode(posterior.mean).sample
            rec = torch.nn.functional.interpolate(
                rec.float(), size=(imgs.shape[-2], imgs.shape[-1]),
                mode="bilinear", align_corners=False
            )
            vres = (imgs.float() - rec).abs().mean(dim=1, keepdim=True)
            vres_pooled = pool_to_patch_grid(vres, n_patches)
            vae_memmap[offset:offset + b] = vres_pooled.cpu().numpy()

        for i in range(b):
            mask_ratio = batch["mask_ratio"][i]
            size_class = batch["size_class"][i]
            model = batch["model"][i]
            mask_path = batch["mask_path"][i]
            meta.append({
                "idx": offset + i,
                "path": batch["path"][i],
                "label": int(batch["label"][i]),
                "kind": batch["kind"][i],
                "mask_path": None if not mask_path else mask_path,
                "mask_ratio": None if float(mask_ratio) == -1 else float(mask_ratio),
                "size_class": None if size_class == "unknown" else size_class,
                "cat": batch["cat"][i],
                "model": None if model == "none" else model,
            })
        offset += b
        pbar.update(b)

    pbar.close()
    for mm in (dino_memmap, hp_memmap, tex_memmap, vae_memmap):
        if mm is not None:
            mm.flush()

    metadata = {
        "n_total": n_total, "n_patches": n_patches, "dino_dim": dino_dim,
        "dino_model": dinov2_model, "image_size": image_size, "use_vae": use_vae,
        "splits": list(splits), "dtype_dino": "float16",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=1)
    with open(labels_file, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    n_real = sum(1 for m in meta if m["label"] == 0)
    n_fake = sum(1 for m in meta if m["label"] == 1)
    print(f"\n特征缓存完成! {n_total} 张 (real={n_real}, fake={n_fake}) -> {output_dir}")
    print(f"  dino_tokens:   {os.path.getsize(dino_file)/1e9:.2f} GB (float16)")
    print(f"  highpass:      {os.path.getsize(hp_file)/1e9:.2f} GB")
    print(f"  texture_energy:{os.path.getsize(tex_file)/1e9:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="D:/lunwen/data/INP-X/inpainting_exchange")
    parser.add_argument("--index_file", type=str, default="D:/lunwen/data/inpx_index.json")
    parser.add_argument("--output_dir", type=str, default="D:/lunwen/data/features_cache")
    parser.add_argument("--splits", type=str, nargs="+", default=["train-data"])
    parser.add_argument("--dinov2_model", type=str, default="dinov2_vits14")
    parser.add_argument("--use_vae", action="store_true")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    extract_features(
        data_root=args.data_root, index_file=args.index_file, output_dir=args.output_dir,
        splits=args.splits, dinov2_model=args.dinov2_model, use_vae=args.use_vae,
        image_size=args.image_size, batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
