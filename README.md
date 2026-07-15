FL-YOLO: Curb-Guided Real-Time Instance Segmentation for Urban Waterlogging Severity Assessment

## Overview

Urban waterlogging caused by extreme precipitation poses substantial risks to transportation safety and urban infrastructure. Sensor-based monitoring systems are costly and often provide limited spatial coverage, while vision-based monitoring remains challenging because curbstones are narrow, frequently occluded, and difficult to distinguish under reflections, adverse illumination, and inundation.

FL-YOLO is a curb-guided real-time instance segmentation framework for urban waterlogging assessment. It uses curbstones as static geometric references and introduces a Direction-Aware Strip Feature Complementary Mapping (StripFCM) module into the mask prototype path of YOLOv11-Seg. StripFCM combines asymmetric strip convolutions with channel-wise directional gating to preserve the linear topology of elongated curbstones while suppressing interference from adjacent road surfaces, water reflections, and background regions.

We also construct CurbFloodSeg, a high-resolution dataset containing 1,407 annotated images for curb segmentation and waterlogging severity analysis. Across three independent runs, FL-YOLO achieves a Mask AP@[0.50:0.95] of 0.639 +/- 0.003, a Mask AP@0.50 of 0.821 +/- 0.004, a Mask AP@0.75 of 0.594 +/- 0.005, and a Foreground IoU of 0.716 +/- 0.002. On an independent external benchmark containing 80 images, FL-YOLO achieves a Foreground IoU of 0.5147. The framework processes images at 79.39 FPS on an NVIDIA RTX 3090.

The predicted curb masks can be converted into geometric exposure ratios for three-level waterlogging severity assessment. This downstream procedure assumes a fixed camera viewpoint and an aligned dry reference frame; it estimates relative curb exposure rather than absolute water depth.

## Environment

The code has been tested with the following core packages:

- Python 3.x
- PyTorch 2.7.0
- torchvision 0.22.0
- NumPy 1.26.3

Install the remaining dependencies required by the repository before running the training or evaluation scripts.

## Training and Validation

Use `train.py` for model training and `val.py` for model validation. Update the dataset path and training parameters in the corresponding configuration files before running the scripts.

```bash
python train.py
python val.py
```

## Dataset

The dataset configuration follows the format illustrated in [`curb.yaml`](./curb.yaml). A compatible dataset can be organized as follows:

```text
dataset/
|-- images/
|   |-- train/
|   `-- val/
|-- labels/
|   |-- train/
|   `-- val/
`-- curb.yaml
```

The labels must follow the instance segmentation format expected by the training pipeline.

Access to CurbFloodSeg is temporarily restricted while compliance and privacy reviews are in progress. Requests for controlled access may be directed to the corresponding author and will be considered subject to the applicable approvals. Data split files and non-sensitive dataset metadata will be released after completion of the review process.

## Models

The source code and trained FL-YOLO weights are provided in this repository. Please ensure that the model configuration and checkpoint correspond to the same code version when reproducing the reported results.

## Citation

If you find the code, trained models, or CurbFloodSeg useful in your research, please cite the following manuscript. The citation information will be updated after publication.

```bibtex
@unpublished{xiao2026flyolo,
  title={FL-YOLO: Curb-Guided Real-Time Instance Segmentation for Urban Waterlogging Severity Assessment},
  author={Xiao, Hao and Shen, Yang and Zhang, Yuming and Liu, Tingmin and Hu, Zhifeng and Gao, Yuncan and Cao, Runlong and Zang, Ying},
  note={Manuscript submitted to The Visual Computer},
  year={2026}
}
```

## Acknowledgements

Part of this code is adapted from [Ultralytics](https://github.com/ultralytics/ultralytics). We thank the authors and contributors for making their work publicly available.
