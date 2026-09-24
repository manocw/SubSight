"""SubSight web viewer (v1, local-first).

Run:
    pip install -r web/requirements.txt
    python -m web.app            # serves http://127.0.0.1:8000

Upload a clip, the worker samples frames, scores them with the ONNX
heads (or a stub when PREDICTOR=stub), and the viewer shows flags on
a timeline. See web/CONTRACT.md for the model scores format.
"""

import json
import os
import sqlite3
import threading
import uuid
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
VIDEOS = DATA / "videos"
SCORES = DATA / "scores"
DB = DATA / "jobs.db"
SAMPLE_FPS = 2.0
THRESHOLDS = {"pipe": 0.02, "macro": 0.05}

CLASSES = ["ship_hull", "anode", "marine_growth", "paint_peel", "corrosion",
           "defect", "propeller", "sea_chest_grating", "over_board_valves",
           "bilge_keel"]

# Per-pixel sigmoid thresholds from the Kaggle val sweep. Production
# classes only: hull 0.5 (prec 0.92), propeller 0.7 (prec 0.74).
# Rest use 0.5, logged but never flagged in the operator view.
SIGMOID_THRESH = {"ship_hull": 0.5, "propeller": 0.7}
PROD_CLASSES = ["ship_hull", "propeller"]

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

app = FastAPI(title="SubSight viewer")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


def db() -> sqlite3.Connection:
    DATA.mkdir(parents=True, exist_ok=True)
    VIDEOS.mkdir(parents=True, exist_ok=True)
    SCORES.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS jobs "
                "(id TEXT PRIMARY KEY, filename TEXT, status TEXT)")
    return con


def set_status(vid: str, status: str) -> None:
    con = db()
    con.execute("UPDATE jobs SET status=? WHERE id=?", (status, vid))
    con.commit()
    con.close()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


class StubPredictor:
    """Fake heads for UI testing (PREDICTOR=stub). Returns a blob."""

    def score(self, frame: np.ndarray) -> dict:
        h, w, _ = frame.shape
        yy, xx = np.mgrid[:h, :w]
        blob = ((xx - w // 2) ** 2 + (yy - h // 2) ** 2) < (h // 4) ** 2
        cov = float(blob.mean())
        return {"pipe": cov, "macro": cov / 10,
                "classes": {c: (cov / 10 if i else 0.0)
                            for i, c in enumerate(CLASSES)}}


class OnnxPredictor:
    """Real heads from web/models (needs onnxruntime + .onnx files)."""

    def __init__(self) -> None:
        import onnxruntime as ort

        self.pipe = ort.InferenceSession(str(HERE / "models" / "pipe.onnx"))
        self.hull = ort.InferenceSession(str(HERE / "models" / "hull.onnx"))

    def score(self, frame: np.ndarray) -> dict:
        from PIL import Image  # local import keeps startup light
        small = np.array(Image.fromarray(frame).resize((256, 256)))
        x = (small.astype(np.float32) / 255 - _MEAN) / _STD
        x = np.moveaxis(x, -1, 0)[None].astype(np.float32)
        pipe = float((sigmoid(self.pipe.run(None, {"input": x})[0])
                      > 0.5).mean())
        probs = sigmoid(self.hull.run(None, {"input": x})[0][0])
        cov = {c: float((probs[i] > SIGMOID_THRESH.get(c, 0.5)).mean())
               for i, c in enumerate(CLASSES)}
        macro = float(np.mean([cov[c] for c in PROD_CLASSES]))
        return {"pipe": pipe, "macro": macro, "classes": cov}


def get_predictor():
    if os.environ.get("PREDICTOR", "stub") == "onnx":
        return OnnxPredictor()
    return StubPredictor()


def process_video(vid: str, src: Path) -> None:
    """Sample frames at SAMPLE_FPS, score, write scores json."""
    set_status(vid, "working")
    try:
        reader = imageio.get_reader(str(src))
        meta = reader.get_meta_data()
        src_fps = float(meta.get("fps", 5.0))
        step = max(1, round(src_fps / SAMPLE_FPS))
        predictor = get_predictor()
        frames = []
        for i, frame in enumerate(reader):
            if i % step:
                continue
            s = predictor.score(frame)
            s["t"] = round(i / src_fps, 2)
            frames.append(s)
        out = {"fps": SAMPLE_FPS, "frames": frames}
        (SCORES / f"{vid}.json").write_text(json.dumps(out))
        set_status(vid, "done")
    except Exception as exc:  # keep the job row honest
        (SCORES / f"{vid}.error.txt").write_text(str(exc))
        set_status(vid, "error")


@app.post("/api/videos")
async def upload(file: UploadFile):
    vid = uuid.uuid4().hex[:12]
    dest = VIDEOS / f"{vid}.mp4"
    dest.write_bytes(await file.read())
    con = db()
    con.execute("INSERT INTO jobs VALUES (?, ?, ?)",
                (vid, file.filename, "queued"))
    con.commit()
    con.close()
    threading.Thread(target=process_video, args=(vid, dest),
                     daemon=True).start()
    return {"id": vid}


@app.get("/api/jobs")
def jobs():
    con = db()
    rows = con.execute("SELECT id, filename, status FROM jobs "
                       "ORDER BY rowid DESC").fetchall()
    con.close()
    return [{"id": r[0], "filename": r[1], "status": r[2]} for r in rows]


@app.get("/api/videos/{vid}/scores")
def scores(vid: str):
    path = SCORES / f"{vid}.json"
    if not path.exists():
        return JSONResponse({"error": "not ready"}, status_code=404)
    return json.loads(path.read_text())


@app.get("/api/videos/{vid}/file")
def videofile(vid: str):
    return FileResponse(VIDEOS / f"{vid}.mp4", media_type="video/mp4")


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.app:app", host="127.0.0.1", port=8000)
