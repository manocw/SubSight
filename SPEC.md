# SubSight spec: phases 2 to 5

Phase 1 stays open until the README gate table is full. No new task folders till then. Each phase below runs at 256px on free Colab T4 first, fixed seeds, no test leakage, never train on the eval split.

## Phase 2: hull defect segmentation

Goal: multi-class mask on ship hull stills: hull, anode, growth, peel, corrosion. Operator toggles this head by job.

Data: LIACI, 1,893 frames, 10 classes, via data.sintef.no. Check licence on the page before publishing. Proxy for rare anodes: MaVeCoDD dry-dock corrosion stills. Check its page too. SUIM (~1,635 frames, 8 classes) as extra pretrain or aug mix only, not as the headline set.

Model: same U-Net ResNet-34 first at 256px for a like-for-like read, then SegFormer-B0. One shared backbone later, separate heads per task.

Gate: per-class IoU table plus mean. Rare classes will score low. Say so in the table notes, keep the misses in the figure.

## Phase 3: sonar detector

Goal: boxes on forward-scan sonar for pipe, cable, wall, debris. ROV video stays the input for optical heads, sonar gets its own head.

Data: first pick SubPipe SSS waterfall (~10k frames) since it matches the pipe domain, fallback UATD (7,600 FLS frames). NKSID (2,617 frames, long tail) for rare-class study. SWDD harbour wall for wall proxy. Check each licence page before publishing. Real operator sonar is a gap. Train on these proxies and label it as such.

Model: YOLOv8 nano/small at 256px first, then 640px once the small run is defended.

Gate: mAP50 plus per-class AP, split by time order where timestamps exist. Report cross-set drop if a second sonar set is available.

## Phase 4: enhancement frontend

Goal: dehaze murk before the task heads, only if it lifts downstream scores.

Data: UIEB (890 paired frames) plus EUVP for extra pairs. Check licence pages. Test on pipe-seg val: score IoU with and without the frontend, same frozen head.

Gate: ship if downstream IoU or mAP lifts on a locked split. If no lift, keep the code, report no-lift, leave it off by default.

## Phase 5: edge export plus toggle UI plus tank demo

Goal: one toolkit the operator toggles by job, live on cheap hardware.

Export path: PyTorch to ONNX, then TensorRT on Jetson Orin Nano. Report latency vs accuracy curve: FP32 vs FP16 vs INT8, ms per frame at 256px plus mIoU or mAP at each point. INT8 needs a calibration split, never the eval split.

UI: minimal toggle. One input (file or camera), one head switch (pipe, hull, sonar, debris flag, enhance on/off), side-by-side view.

Demo: 60 second side-by-side clip, raw left and overlay right. Cut one first on public data, then one in the UCL tow tank via MARS (contacts: Yuanchang Liu, Yao Zhang, Giles Thomas). Use MechSpace mounts, no new vehicle.

## Data links

- SubPipe data: https://zenodo.org/doi/10.5281/zenodo.10053564
- SubPipe docs: https://github.com/remaro-network/SubPipe-dataset
- SubPipe paper: arXiv:2401.17907
- LIACI: via data.sintef.no (verify licence on page)
- SUIM, TrashCan, FathomNet (pretrain only), MaVeCoDD, UIEB, EUVP, SubPipe SSS, UATD, NKSID, SWDD: verify each licence on its page before publishing, keep a licence note per set in the task README.

Total usable across sets is 50GB plus. Local repo never holds data or weights.

## Budget (2k pot, self-funded, no funding applications)

- Colab T4 free tier first, cloud burst cap 150 GBP
- Jetson Orin Nano plus SSD: ~500 GBP
- Camera plus housing: ~200 GBP
- Expo plus poster travel: ~300 GBP
- No BlueROV2, over budget. Tow tank time free via MARS academics.

## Gaps stated plainly

Cables lack labelled optical data. Rare anodes are thin in every set. Real operator sonar differs from public proxies. Each task README will name its proxy and its gap. No silent substitution.
