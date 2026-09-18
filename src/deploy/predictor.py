"""v4 检测器推理封装: DINOv2 (冻结) + highpass + 30K 头

输入: RGB 图像 (HWC uint8 numpy 或 PIL)
输出: detection score + 37x37 定位热图
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.weakly_supervised_v2 import MLPHead
from utils.highpass import cross_diff_highpass, pool_to_patch_grid

GRID = 37
SIZE = 518
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class EditDetector:
    def __init__(self, head_path="D:/lunwen/outputs/weak_sup_v4_head.pt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        import torch.hub as hub
        self.dino = hub.load("facebookresearch/dinov2", "dinov2_vits14",
                             trust_repo=True).to(self.device).eval()
        self.head = MLPHead(385).to(self.device)
        self.head.load_state_dict(torch.load(head_path, map_location=self.device))
        self.head.eval()

    @torch.no_grad()
    def predict(self, img: np.ndarray) -> dict:
        """img: HWC uint8 RGB -> {score, heatmap(37x37), mask_bool(37x37)}"""
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        t = F.interpolate(t, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        t = (t - MEAN.to(self.device)) / STD.to(self.device)
        tok = self.dino.forward_features(t)["x_norm_patchtokens"]
        hp = F.pad(cross_diff_highpass(t), (0, 1, 0, 1))
        hp_p = pool_to_patch_grid(hp, GRID * GRID)
        X = torch.cat([tok, hp_p], dim=-1).to(self.device)
        logits = self.head(X).squeeze(0)
        scores = torch.sigmoid(logits).cpu().numpy().reshape(GRID, GRID)
        return {
            "score": float(scores.max()),
            "heatmap": scores,          # 37x37 float
            "mask": (scores >= 0.5),    # 37x37 bool
        }


if __name__ == "__main__":
    from PIL import Image
    d = EditDetector()
    img = np.asarray(Image.open("D:/lunwen/data/sdxl_edits_500/sdxl_0000/edit.jpg").convert("RGB"))
    r = d.predict(img)
    print("score:", round(r["score"], 4), "heatmap shape:", r["heatmap"].shape,
          "mask sum:", int(r["mask"].sum()))
