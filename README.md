# SDMOT: A Scene-Driven Multi-Object Tracking Framework with Dynamic Feature Optimization and Nonlinear Motion Modeling

<div align="center">

**Multi-Object Tracking System Based on Dynamic Feature Optimization and Nonlinear Modeling**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2+-orange.svg)](https://pytorch.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](LICENSE)

</div>

## Overview

SDMOT is a multi-object tracking (MOT) framework built upon the [BoxMOT](https://github.com/mikel-brostrom/boxmot) library, incorporating scene-driven dynamic feature optimization and nonlinear motion modeling for robust pedestrian tracking. The system follows the tracking-by-detection paradigm and integrates custom-trained YOLO detectors, multiple ReID backends, and a suite of state-of-the-art tracking algorithms with our proposed enhancements.

This project accompanies the paper:

> **A Scene-Driven Multi-Object Tracking Framework with Dynamic Feature Optimization and Nonlinear Motion Modeling**

### Key Contributions

- **Adaptive Noise Extended Kalman Filter (EKF)** with a CTRA (Constant Turn Rate and Acceleration) nonlinear motion model, featuring dynamic Jacobian fusion that blends analytical and numerical computation for improved tracking accuracy under high nonlinearity.
- **Detector confidence-aware noise adaptation**, where the measurement noise is dynamically adjusted based on detection confidence scores, enhancing robustness in low-confidence scenarios.
- **Custom-trained YOLO11 detectors** optimized for pedestrian detection in diverse scenes, including a specialized model for the DanceTrack dataset.
- **Integrated GUI application** built with PyQt5 for end-to-end tracking pipeline execution and real-time visualization.
- **Trajectory visualization toolkit** with point-based trail rendering showing temporal depth through size-varying trajectory tails.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  GUI Layer (PyQt5)                           │
│   tracking_visualizer.py ─── Main application window         │
│   tracking_system.py ─────── Pipeline orchestrator           │
└──────────────┬──────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│               Pipeline Layer                                 │
│   track.py ──────────── Real-time tracking (YOLO + tracker) │
│   val.py ────────────── Batch evaluation on MOT benchmarks   │
│   visualize_track.py ── Trail visualization CLI tool         │
└──────────────┬──────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│              Core Library: boxmot/                           │
│                                                              │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐  │
│  │  Detectors    │  │  Appearance   │  │    Trackers      │  │
│  │  (tracking/   │  │  (boxmot/     │  │  (boxmot/        │  │
│  │   detectors/) │  │   appearance/)│  │   trackers/)     │  │
│  │               │  │               │  │                  │  │
│  │  - YOLO11    │  │  - OSNet      │  │  - ByteTrack     │  │
│  │  - YOLOv8    │  │  - CLIP-ReID  │  │  - BotSort       │  │
│  │  - YOLOv9    │  │  - LightMBN   │  │  - StrongSORT    │  │
│  │  - YOLOv10   │  │  - ResNet     │  │  - OC-SORT       │  │
│  │  - YOLO-NAS  │  │  - MobileNet  │  │  - DeepOCSORT    │  │
│  └──────────────┘  └───────────────┘  │  - HybridSort    │  │
│                                        │  - ImprAssoc     │  │
│  ┌──────────────────┐  ┌───────────────────────────────┐  │  │
│  │  Motion (motion/) │  │  Utils (utils/)               │  │  │
│  │  - Kalman Filters │  │  - IoU / ReID association     │  │  │
│  │    (KF, EKF-CTRA) │  │  - Linear assignment (lapx)   │  │  │
│  │  - Camera Motion  │  │  - Matching utilities          │  │  │
│  │    Compensation   │  └───────────────────────────────┘  │  │
│  └──────────────────┘                                      │  │
│  ┌──────────────────┐  ┌───────────────────────────────┐  │  │
│  │  Postprocessing   │  │  Data                         │  │  │
│  │  - GSI smoothing  │  │  - Image / Video loader       │  │  │
│  └──────────────────┘  └───────────────────────────────┘  │  │
└──────────────────────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│              Evaluation Layer (TrackEval)                    │
│  Metrics: HOTA, MOTA, IDF1, CLEAR                           │
│  Benchmarks: MOT15/16/17/20, DanceTrack, SportsMOT,         │
│              BDD100K, KITTI                                  │
└─────────────────────────────────────────────────────────────┘
```

## Supported Tracking Methods

| Tracker | Paper | Key Feature |
|:---:|:---:|:---|
| ByteTrack | [arXiv:2110.06864](https://arxiv.org/abs/2110.06864) | Every detection counts, low-score association |
| BotSort | [arXiv:2206.14651](https://arxiv.org/abs/2206.14651) | Camera motion compensation + ReID |
| StrongSORT | [arXiv:2202.13514](https://arxiv.org/abs/2202.13514) | Strong appearance descriptors |
| OC-SORT | [arXiv:2203.14360](https://arxiv.org/abs/2203.14360) | Observation-centric sorting |
| Deep OC-SORT | [arXiv:2302.11813](https://arxiv.org/abs/2302.11813) | Deep appearance integration |
| HybridSort | [arXiv:2308.00783](https://arxiv.org/abs/2308.00783) | Hybrid association strategy |
| ImprAssoc | [CVPRW 2023](https://openaccess.thecvf.com/content/CVPR2023W/E2EAD/papers/Stadler_An_Improved_Association_Pipeline_for_Multi-Person_Tracking_CVPRW_2023_paper.pdf) | Improved association pipeline |

Each tracker can be enhanced with our **EKF-CTRA** motion model variant (e.g., `bytetrack_with_ekf_improv`, `bytetrack_with_ukf_improv`).

## Installation

### Prerequisites

- Python >= 3.9
- CUDA-compatible GPU recommended (CPU mode supported)

### Setup

```bash
# Clone the repository
git clone https://github.com/kaneki-11/SDMOT.git
cd SDMOT

# Install Poetry (dependency management)
pip install poetry

# Install all dependencies (including YOLO support)
poetry install --with yolo

# Activate the virtual environment
poetry shell
```

### Dependencies

Core dependencies include PyTorch >= 2.2.1, OpenCV, filterpy (Kalman filtering), lapx (linear assignment), scikit-learn, and ultralytics (YOLO models). The full list is in `pyproject.toml`.

For GPU acceleration, ensure the appropriate CUDA toolkit is installed. The project supports CUDA 12.1 via the configured PyTorch source.

### Docker (Optional)

A Dockerfile is provided for containerized deployment:

```bash
docker build -t sdmot:latest .
docker run --gpus all -it sdmot:latest
```

## Usage

### Command-Line Interface

**Run tracking on a video or image sequence:**

```bash
python tracking/track.py --yolo-model yolov8n.pt \
                         --tracking-method bytetrack \
                         --source path/to/video.mp4
```

**Evaluate on MOT benchmarks:**

```bash
python tracking/val.py --benchmark MOT17 \
                       --yolo-model yolov8n.pt \
                       --reid-model osnet_x0_25_msmt17.pt \
                       --tracking-method bytetrack \
                       --source ./tracking/val_utils/MOT17/train \
                       --verbose
```

**Use custom EKF-enhanced tracking:**

```bash
python tracking/track.py --yolo-model spo_yolo11s_best.pt \
                         --tracking-method bytetrack_with_ekf_improv \
                         --source path/to/sequence/ \
                         --classes 0
```

**Visualize tracking results with trajectory trails:**

```bash
python visualize_track.py --seq-dir path/to/img1/ \
                          --mot-txt path/to/tracks.txt \
                          --out-video output.mp4 \
                          --trail 30 \
                          --fps 25
```

### GUI Application

Launch the graphical interface for interactive tracking:

```bash
python tracking_visualizer.py
```

The GUI provides:

- Video / sequence loading and management
- YOLO model and tracking method selection (including EKF variants)
- One-click pipeline execution (detection → ReID → tracking → visualization)
- Real-time video playback of tracking results with trajectory trails
- Detection and tracking result inspection

### ReID Model Selection

For trackers that use appearance features, you can select a ReID model based on your speed-accuracy tradeoff:

```bash
# Lightweight (fast, suitable for edge devices)
--reid-model lmbn_n_cuhk03_d.pt

# Balanced
--reid-model osnet_x0_25_msmt17.pt

# Heavy (best accuracy)
--reid-model clip_market1501.pt
```

### Model Export

Export ReID models to alternative inference backends:

```bash
python boxmot/appearance/reid_export.py --include onnx      --device cpu
python boxmot/appearance/reid_export.py --include openvino  --device cpu
python boxmot/appearance/reid_export.py --include engine    --device 0 --dynamic
```

Supported backends: PyTorch, ONNX, OpenVINO, TensorRT, TFLite, TorchScript.

## Project Structure

```
SDMOT/
├── boxmot/                          # Core tracking library
│   ├── appearance/                  # ReID models & inference backends
│   │   ├── backbones/               # Neural network architectures (OSNet, CLIP, ResNet, etc.)
│   │   ├── backends/                # Inference engines (PyTorch, ONNX, TensorRT, etc.)
│   │   └── exporters/               # Model format conversion utilities
│   ├── configs/                     # Tracker YAML configurations
│   ├── motion/                      # Motion modeling
│   │   ├── cmc/                     # Camera motion compensation (ECC, ORB, SIFT)
│   │   └── kalman_filters/          # Kalman filter variants (KF, EKF-CTRA, UKF)
│   ├── postprocessing/              # Global smoothing & interpolation
│   ├── trackers/                    # Tracking algorithm implementations
│   └── utils/                       # Association, IoU, matching utilities
├── tracking/                        # Pipeline & evaluation
│   ├── detectors/                   # YOLO detector interfaces
│   ├── val_utils/                   # TrackEval evaluation toolkit
│   └── weights/                     # Pre-trained model weights
├── examples/                        # Jupyter notebook examples
├── tests/                           # Unit & performance tests
├── tracking_system.py               # Pipeline orchestrator
├── tracking_visualizer.py           # PyQt5 GUI application
├── visualize_track.py               # CLI trail visualization tool
├── pyproject.toml                   # Project configuration & dependencies
├── Dockerfile                       # Docker build configuration
└── LICENSE                          # AGPL-3.0 license
```

## Evaluation

The system supports evaluation on standard MOT benchmarks using the TrackEval toolkit. Default metrics include HOTA, MOTA, and IDF1.

**Supported datasets:** MOT15, MOT16, MOT17, MOT20, DanceTrack, SportsMOT, BDD100K, KITTI.

Pre-generated detections and embeddings can be cached and reused across different tracker configurations, avoiding redundant computation during hyperparameter tuning.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{sdmot2025,
  title   = {SDMOT: A Scene-Driven Multi-Object Tracking Framework with Dynamic Feature Optimization and Nonlinear Motion Modeling},
  author  = {Yuheng Suo},
  year    = {2025},
  url     = {https://github.com/kaneki-11/SDMOT}
}
```

## Acknowledgments

This project is built upon the excellent [BoxMOT](https://github.com/mikel-brostrom/boxmot) library by Mikel Broström et al. We also acknowledge the [Ultralytics](https://github.com/ultralytics/ultralytics) team for the YOLO detection framework, and the authors of the individual tracking algorithms (ByteTrack, BotSort, StrongSORT, OC-SORT, Deep OC-SORT, HybridSort, ImprAssoc) whose implementations are integrated into this framework.

## License

This project is licensed under the [AGPL-3.0 License](LICENSE).
