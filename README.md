# pose2anim

**Video → 3D Pose → Animation pipeline** for YouTube Shorts and animated series.

## Pipeline Overview

```
┌─────────────┐    ┌──────────────┐    ┌────────────────┐    ┌───────────────┐
│  Video Input │ →  │ YOLO26 Pose  │ →  │ D3DP 3D Lift   │ →  │ BVH/FBX Export│
│  (mp4/webcam)│    │ (2D keypoints)│    │ (3D keypoints) │    │ (animation)   │
└─────────────┘    └──────────────┘    └────────────────┘    └───────────────┘
                                                                     │
                                                              ┌──────▼──────┐
                                                              │  Blender /  │
                                                              │  Retarget   │
                                                              └─────────────┘
```

### Stage 1: 2D Pose Estimation (YOLO26)
- Uses Ultralytics YOLO26-pose for real-time 17-point COCO keypoint detection
- NMS-free inference (~43% faster on CPU vs YOLO11)
- Small-Target-Aware Label Assignment for better occluded keypoint handling

### Stage 2: 3D Pose Lifting (D3DP)
- Diffusion-based 3D Human Pose Estimation (ICCV 2023)
- Generates multiple 3D pose hypotheses from 2D observations
- Joint-wise reProjection-based Multi-hypothesis Aggregation (JPMA)
- Configurable accuracy/speed trade-off via `num_proposals` and `sampling_timesteps`

### Stage 3: Animation Export
- Converts 3D joint positions → joint rotations → BVH format
- Optional FBX export via Blender scripting
- Retargeting support for custom 3D character rigs

## Quick Start

```bash
# Clone
git clone https://github.com/mroncka/pose2anim.git
cd pose2anim

# Install
pip install -e .

# Run on video
python -m pose2anim.cli process --input video.mp4 --output animation.bvh

# Run on webcam (real-time preview)
python -m pose2anim.cli live --camera 0
```

## Installation

### Requirements
- Python 3.10+
- PyTorch 2.0+ (CUDA recommended)
- Ultralytics (YOLO26)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Docker
```bash
docker build -t pose2anim .
docker run --gpus all -v $(pwd)/data:/app/data pose2anim process --input data/video.mp4
```

## Project Structure

```
pose2anim/
├── pose2anim/
│   ├── __init__.py
│   ├── cli.py              # CLI entry point
│   ├── pipeline.py         # Main pipeline orchestrator
│   ├── pose2d/
│   │   ├── __init__.py
│   │   └── yolo_estimator.py   # YOLO26 2D pose estimation
│   ├── pose3d/
│   │   ├── __init__.py
│   │   └── d3dp_lifter.py      # D3DP 3D pose lifting
│   ├── export/
│   │   ├── __init__.py
│   │   ├── bvh_writer.py       # BVH animation export
│   │   └── fbx_exporter.py     # FBX export via Blender
│   └── utils/
│       ├── __init__.py
│       ├── skeleton.py         # Skeleton definitions & mappings
│       └── visualization.py    # Debug visualization
├── configs/
│   └── default.yaml            # Pipeline configuration
├── scripts/
│   └── blender_retarget.py     # Blender retargeting script
├── tests/
│   └── test_pipeline.py
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Configuration

Edit `configs/default.yaml` to tune the pipeline:

```yaml
pose2d:
  model: yolo26m-pose     # yolo26n-pose | yolo26s-pose | yolo26m-pose
  confidence: 0.5
  device: auto             # auto | cpu | cuda:0

pose3d:
  num_proposals: 5         # More = better accuracy, slower
  sampling_timesteps: 5    # Diffusion steps
  model_checkpoint: null   # Auto-downloads if null

export:
  format: bvh              # bvh | fbx
  fps: 30
  skeleton: coco17         # coco17 | h36m
```

## Blender Retargeting

After exporting BVH, retarget to your 3D character:

```bash
blender --background --python scripts/blender_retarget.py -- \
  --bvh output.bvh \
  --character model.fbx \
  --output animated_character.fbx
```

## References

- [YOLO26](https://docs.ultralytics.com/) - Ultralytics YOLO26 pose estimation
- [D3DP](https://github.com/paTRICK-swk/D3DP) - Diffusion-Based 3D Human Pose Estimation (ICCV 2023)
- [video2bvh](https://github.com/KevinLTT/video2bvh) - Video to BVH conversion reference

## License

MIT
