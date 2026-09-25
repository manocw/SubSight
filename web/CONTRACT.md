# Web/model contract (both sessions honor this, change by agreement)

## Model files (gitignored, live in web/models/)

| File | Input | Output |
|---|---|---|
| pipe.onnx | (1,3,256,256) float32, RGB, ImageNet-normalized | (1,1,256,256) float32 logits |
| hull.onnx | (1,3,256,256) float32, RGB, ImageNet-normalized | (1,3,256,256) float32 logits |

Hull channel order (fixed, equals prod3 config data.classes):
ship_hull, propeller, sea_chest_grating.

Produced by web/export_onnx.py on a GPU box from checkpoints/best.pth
(pipe) and tasks/hull-defect/checkpoints_prod3/best.pth (hull). The
10ch file in tasks/hull-defect/checkpoints_bce_dice_40/best.pth stays
on disk as fallback and is never overwritten. The web app never
trains and never imports torch. A 10ch hull.onnx is rejected with a
clear error, re-export from prod3.

## Scores file (one per uploaded video)

`web/data/scores/<video_id>.json`:

```json
{
  "fps": 2.0,
  "frames": [
    {"t": 0.0, "pipe": 0.11, "macro": 0.21,
     "classes": {"ship_hull": 0.40, "propeller": 0.02,
                 "sea_chest_grating": 0.03}},
    "... one entry per sampled frame, t in video seconds ..."
  ]
}
```

Values are predicted-positive pixel fractions per class (plus pipe
and macro). Macro is the mean over the operator pair
(ship_hull, propeller). Grating is logged in the scores file, never
flagged. A frame is flagged when an operator value tops its
threshold: pipe 0.02, ship_hull 0.05, propeller 0.01.

## Endpoints (web/app.py)

- POST /api/videos (multipart file plus head=both/pipe/hull) -> {"id": ...}
  Head gates the job: pipe surveys never fire hull flags and reverse.
  Scores file carries the head used.
- GET /api/jobs -> list with status (queued/working/done/error)
- GET /api/videos/{id}/scores -> scores file above
- GET /api/videos/{id}/overlay -> side-by-side mp4 (raw left,
  overlay right, pipe red, hull green, propeller blue, stamp per frame)
- GET /api/videos/{id}/transcript -> plain text, one line per sampled
  frame (`12.5s: pipe 0.31, propeller 0.08` or `clear`)
- GET / serves the viewer (upload, raw plus overlay players,
  flag timeline, transcript view plus download)
