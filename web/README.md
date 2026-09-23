# SubSight web viewer (v1)

Upload a subsea inspection clip, get per-frame pipe plus hull-defect
scores with a flag timeline. Local-first: SQLite, local disk, stub
predictor until the GPU session exports real ONNX weights.

## Run

```bash
pip install -r web/requirements.txt
python -m web.app
```

Open http://127.0.0.1:8000, upload an mp4, watch flags appear.

## Real models

On a GPU box with the repo and checkpoints:

```bash
pip install onnxruntime
python web/export_onnx.py --out web/models
PREDICTOR=onnx python -m web.app
```

See web/CONTRACT.md for the interface both sides honor.

## Later (not now)

Videos move to Cloudflare R2, frontend to Cloudflare Pages, API to
Fly.io or a small VPS, Supabase only when accounts are needed. No
infra until the local version earns it.
