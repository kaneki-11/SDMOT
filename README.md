# SDMOT: A Scene-Driven Multi-Object Tracking Framework with Dynamic Feature Optimization and Nonlinear Motion Modeling

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2+-orange.svg)](https://pytorch.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](LICENSE)

**Xi'an Technological University**

</div>

## Overview

This repository contains the official implementation of **SDMOT**, a multi-object tracking method that integrates dynamic feature optimization with nonlinear motion modeling for robust tracking in dynamic and complex scenarios such as sports events and dance performances.

SDMOT follows the tracking-by-detection paradigm, built upon YOLOv11 as the detector and ByteTrack as the association baseline. The method addresses four core challenges in complex MOT scenarios: dense occlusion, cross-frame scale variations and target deformations, motion-induced boundary blurring, and nonlinear motion trajectories.

> **Paper:** SDMOT: Scene-Driven Dynamic Feature Optimization and Nonlinear Modeling for Multi-Object Tracking
>
> **Authors:** Xiuhua Hu, Yuheng Suo, Yan Hui, Chao Shen
>
> **Affiliation:** School of Computer Science and Engineering, Xi'an Technological University, Xi'an, P. R. China

## Key Contributions

### 1. Gate-BiAxial Fusion Module (G-BiF)

Embedded into the C3k2 module of the YOLOv11 backbone, G-BiF enhances boundary discrimination among targets under dense occlusion through two components:

- **Dual-Stream Feature Gating:** Expands the feature space dimensionality from *C* to approximately *2C^3* through gated element-wise multiplication of dual-branch linear projections, enabling the model to capture target decision boundaries more accurately.
- **BiAxial Augmentation Module:** Deploys cascaded one-dimensional dynamic convolution kernels along horizontal and vertical axes to adaptively adjust the receptive field, handling cross-frame scale variations and target deformations caused by dynamic camera views.

### 2. Edge-Aware Convolutional Operator (EAC)

Integrated into the YOLOv11 backbone downsampling stages, EAC optimizes boundary feature extraction by encoding directional biases. Through zero-padding in the left, right, upper, and lower directions, EAC guides convolution kernels to extract boundary contour features from four spatial directions, mitigating pixel diffusion caused by motion-induced boundary blurring.

### 3. CTRA-based Adaptive Extended Kalman Filter

An 8-dimensional state vector **[x, y, a, h, v, yaw, yaw_rate, acc]** models nonlinear motion under the Constant Turn Rate and Acceleration (CTRA) model. Two key strategies are proposed:

- **Confidence-Aware Noise Adjustment:** The measurement noise covariance *R* is dynamically adjusted based on the detection confidence score *c*, reducing the impact of unreliable measurements during occlusion or motion blur.
- **Dynamic Jacobian Matrix Update:** Blends the analytical Jacobian with a numerically computed Jacobian via Frobenius-norm-based weighting, enabling accurate state transition modeling in highly nonlinear motion regimes.

## System Architecture

```
Input Video / Image Sequence
         │
         ▼
┌─────────────────────────────────────────────────────────
│              Detection Stage (YOLOv11 + G-BiF + EAC)     │
│                                                          │
│  ┌──────────┐    ┌──────────┐    ──────────────────┐   │
│  │  EAC     │───▶│ Backbone │───▶│  Neck (FPN/PAN)  │   │
│  │Operator  │    │ + G-BiF  │    │                  │   │
│  ──────────┘    └──────────┘    └────────┬─────────┘   │
│                                            │              │
│                                    ┌───────▼───────┐     │
│                                    │ Detection Head │     │
│                                    │ D_t = {x,y,a,h,c}│   │
│                                    └───────┬───────     │
└────────────────────────────────────────────┼─────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────┐
│            Association Stage (ByteTrack + Adaptive EKF)   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  CTRA-based Adaptive EKF                          │   │
│  │  - Confidence-aware noise adjustment (R matrix)   │   │
│  │  - Dynamic Jacobian fusion (F_analytical/F_numeric)│  │
│  │  - 8-dim state: [x, y, a, h, v, yaw, ω, acc]     │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  High-confidence matching (IoU > τ_h)                    │
│  Low-confidence matching  (IoU > τ_l)                    │
│  Hungarian algorithm for data association                │
└─────────────────────────────────────────────────────────┘
                                             │
                                             ▼
                              Output Trajectories T_{t+1}
```

## Experimental Results

### SportsMOT Benchmark

| Method | HOTA | DetA | AssA | MOTA | IDF1 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TransTrack | 68.9 | 82.7 | 57.5 | 92.6 | 71.5 |
| OC-SORT | 71.9 | 86.4 | 59.8 | 94.5 | 72.2 |
| Deep-EIoU | 74.1 | 87.2 | 63.1 | 95.1 | 75.0 |
| MambaMOT+ | 71.3 | 86.7 | 58.6 | 94.9 | 71.1 |
| **SDMOT (Ours)** | **76.5** | **91.7** | **63.9** | **98.2** | **76.1** |

### DanceTrack Benchmark

| Method | HOTA | DetA | AssA | MOTA | IDF1 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| OC-SORT | 55.1 | 80.3 | 38.3 | 92.0 | 54.6 |
| C-BIoU | 60.6 | 81.3 | 45.4 | 91.6 | 61.6 |
| DeepMoveSORT-TransFilter | 63.0 | 82.0 | 48.6 | 92.6 | 65.0 |
| MambaMOT+ | 56.1 | 80.8 | 39.0 | 90.3 | 54.9 |
| **SDMOT (Ours)** | **65.9** | **90.1** | 48.2 | **97.0** | 64.5 |

### Ablation Study (HOTA)

| Configuration | SportsMOT | DanceTrack |
|:---|:---:|:---:|
| Baseline (YOLOv11s + ByteTrack) | 66.3 | 52.8 |
| + G-BiF | 68.9 (+2.6) | 55.7 (+2.9) |
| + G-BiF + EAC | 69.8 (+0.9) | 59.6 (+3.9) |
| + G-BiF + EAC + Adaptive EKF | **76.5 (+6.7)** | **65.9 (+6.3)** |

## Installation

### Prerequisites

- Python >= 3.9
- CUDA-compatible GPU recommended (RTX 4090 used in experiments)

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

### Model Weights

Pre-trained model weights are **not** included in this repository due to file size constraints. The following weights are required:

| Weight File | Description | Source |
|:---|:---|:---|
| `spo_yolo11s_best.pt` | Custom YOLOv11s trained on SportsMOT | Train with provided config |
| `dance_yolo11s_best.pt` | Custom YOLOv11s trained on DanceTrack | Train with provided config |
| `osnet_x0_25_msmt17.pt` | ReID model (OSNet) | Auto-downloaded by BoxMOT |

To train the custom YOLOv11 detectors, use the Ultralytics training pipeline with the G-BiF and EAC modifications in the backbone.

### Docker (Optional)

```bash
docker build -t sdmot:latest .
docker run --gpus all -it sdmot:latest
```

## Usage

### Training the Custom Detector

The YOLOv11s detector is enhanced with G-BiF (embedded in C3k2 module) and EAC (integrated at downsampling stages). Training configuration:

- Input size: 640 x 640
- Optimizer: SGD, initial lr = 0.01, decay factor = 0.01
- Epochs: 250, Batch size: 32
- GPU: RTX 4090 (24 GB), AMP enabled

### Running Tracking

**Evaluate on SportsMOT or DanceTrack:**

```bash
python tracking/val.py --benchmark SportsMOT \
                       --yolo-model spo_yolo11s_best.pt \
                       --tracking-method bytetrack_with_ekf_improv \
                       --source ./tracking/val_utils/SportsMOT/test \
                       --verbose
```

**Use the EKF-enhanced tracker (core contribution):**

```bash
python tracking/track.py --yolo-model spo_yolo11s_best.pt \
                         --tracking-method bytetrack_with_ekf_improv \
                         --source path/to/sequence/ \
                         --classes 0
```

The `bytetrack_with_ekf_improv` tracking method uses the CTRA-based adaptive EKF with confidence-aware noise adjustment and dynamic Jacobian updates.

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

The GUI supports video loading, YOLO model selection, tracking method selection (including EKF variants), one-click pipeline execution, and real-time video playback with trajectory trails.

## Project Structure

```
SDMOT/
├── boxmot/                              # Core tracking library (based on BoxMOT)
│   ├── motion/
│   │   └── kalman_filters/
│   │       ├── base_kalman_filter.py    # Abstract Kalman filter base class
│   │       ├── xyah_kf.py               # Standard linear Kalman filter
│   │       └── xyah_kf_ekf_ctrv_acc_final.py  # ★ CTRA-based Adaptive EKF (core)
│   ├── trackers/
│   │   └── bytetrack/                   # ByteTrack association (baseline)
│   ├── configs/                         # Tracker YAML configurations
│   └── utils/                           # Association and matching utilities
├── tracking/
│   ├── val_utils/trackeval/             # TrackEval evaluation toolkit
│   ── weights/                         # Model weights directory (.gitkeep)
├── tracking_system.py                   # Pipeline orchestrator
├── tracking_visualizer.py               # PyQt5 GUI application
├── visualize_track.py                   # CLI trail visualization tool
├── pyproject.toml                       # Project configuration & dependencies
├── Dockerfile                           # Docker build configuration
└── LICENSE                              # AGPL-3.0 license
```

> **Note:** The `boxmot/` directory contains the full BoxMOT library as the underlying framework. The core contributions of this paper are primarily located in `boxmot/motion/kalman_filters/xyah_kf_ekf_ctrv_acc_final.py` (adaptive EKF) and the custom YOLOv11 backbone modifications (G-BiF and EAC, implemented in the detector training pipeline).

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{hu2025sdmot,
  title   = {SDMOT: Scene-Driven Dynamic Feature Optimization and Nonlinear Modeling for Multi-Object Tracking},
  author  = {Hu, Xiuhua and Suo, Yuheng and Hui, Yan and Shen, Chao},
  journal = {},
  year    = {2025},
  publisher = {Springer Nature},
  note    = {School of Computer Science and Engineering, Xi'an Technological University}
}
```

## Acknowledgments

This work was supported in part by the Shaanxi Provincial Natural Science Foundation under Grant 2025JC-YBMS-764 and in part by the National Natural Science Foundation of China under Grant 52302505.

This project is built upon the [BoxMOT](https://github.com/mikel-brostrom/boxmot) library by Mikel Broström et al. and the [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO framework. We also acknowledge the authors of ByteTrack for the association baseline.

## License

This project is licensed under the [AGPL-3.0 License](LICENSE).
