"""Train YOLOv8 nano on MiniSSS HF waterfall (Phase 3).

Run from repo root:
    python tasks/sonar-detector/src/train.py

Pairs .pbm frames with YOLO txt boxes by filename stem, splits chrono
by timestamp, trains yolov8n at 256px. Reports mAP50 plus per-class
AP on the locked test split, scored once.
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.utils import set_seed  # noqa: E402


def collect_pairs(root: Path) -> list:
    """Match Image stems to YOLO_Annotation stems, sorted by time."""
    imgs = {p.stem: p for p in (root / "Image").glob("*.*")}
    anns = {p.stem: p for p in (root / "YOLO_Annotation").glob("*.txt")}
    anns.pop("classes", None)
    pairs = [(imgs[s], anns[s]) for s in sorted(set(imgs) & set(anns))]
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Train sonar YOLOv8n.")
    ap.add_argument("--config", default="tasks/sonar-detector/config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["data"].get("seed", 42))
    from ultralytics import YOLO

    root = Path(cfg["data"]["root"])
    pairs = collect_pairs(root)
    n = len(pairs)
    tr = int(n * cfg["data"].get("train_ratio", 0.7))
    va = int(n * (cfg["data"].get("train_ratio", 0.7)
                  + cfg["data"].get("val_ratio", 0.15)))
    splits = {"train": pairs[:tr], "val": pairs[tr:va], "test": pairs[va:]}
    print(f"SSS HF: {n} paired / "
          f"{len(splits['train'])} train / {len(splits['val'])} val / "
          f"{len(splits['test'])} test (chrono).")

    work = Path(cfg["train"]["checkpoint_dir"]).parent / "yolo_data"
    from PIL import Image  # .pbm waterfall frames -> .png, YOLO reads png
    for split, items in splits.items():
        (work / "images" / split).mkdir(parents=True, exist_ok=True)
        (work / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, ann in items:
            Image.open(img).save(work / "images" / split /
                                 (img.stem + ".png"))
            shutil.copy(ann, work / "labels" / split / (img.stem + ".txt"))
    data_yaml = work / "sss.yaml"
    yaml.safe_dump({"path": str(work.resolve()), "train": "images/train",
                    "val": "images/val", "test": "images/test",
                    "names": {0: "Pipeline"}}, open(data_yaml, "w"))

    model = YOLO(cfg["model"].get("weights", "yolov8n.pt"))
    model.train(data=str(data_yaml),
                epochs=cfg["train"].get("epochs", 50),
                imgsz=cfg["data"].get("image_size", 256),
                batch=cfg["train"].get("batch", 16),
                lr0=cfg["train"].get("lr", 0.001),
                seed=cfg["data"].get("seed", 42),
                project=str(Path(cfg["train"]["checkpoint_dir"]).resolve()),
                name="yolov8n", exist_ok=True)
    print("Test (locked, scored once):")
    print(model.val(data=str(data_yaml), split="test", imgsz=cfg["data"].get(
        "image_size", 256)))


if __name__ == "__main__":
    main()
