"""v4 编辑检测 API 服务 (FastAPI)

启动: uvicorn api_server:app --host 0.0.0.0 --port 8000
接口:
  POST /predict   multipart file=image  -> {score, heatmap_b64, mask_b64, grid}
  GET  /health                          -> {status}
  GET  /                             -> demo 前端页面
"""
import base64
import io
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from predictor import EditDetector

app = FastAPI(title="Local Edit Detection API", version="1.0")
detector = EditDetector()

STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def heatmap_png(scores: np.ndarray) -> str:
    """37x37 分数 -> 放大的 jet 热图 PNG (base64)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.imshow(scores, cmap="jet", vmin=0, vmax=1)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(detector.device)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    img = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    r = detector.predict(img)
    return JSONResponse({
        "score": r["score"],
        "grid": GRID if (GRID := 37) else 37,
        "heatmap_b64": heatmap_png(r["heatmap"]),
        "mask_b64": heatmap_png(r["mask"].astype(np.float32)),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
