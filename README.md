# FL-YOLO: Curb-Guided Real-Time Instance Segmentation for Urban Waterlogging Severity Assessment

## Overview

Urban waterlogging resulting from extreme precipitation poses substantial risks to transportation safety and urban infrastructure. Sensor-based monitoring systems are costly and provide limited spatial coverage, whereas vision-based monitoring remains challenging because curbstones are narrow, frequently occluded, and difficult to distinguish under reflections, adverse illumination, and inundation.

FL-YOLO is a curb-guided real-time instance segmentation framework for urban waterlogging assessment. It uses curbstones as static geometric references and is built on the YOLOv11-Seg backbone and FPN+PAN neck. The proposed StripSegment head retains the original detection branches and inserts a Direction-Aware Strip Feature Complementary Mapping (StripFCM) module into the high-resolution mask prototype path. StripFCM combines asymmetric strip convolutions with channel-wise directional gating to preserve the linear topology of elongated curbstones while suppressing lateral background interference.

CurbFloodSeg contains 1,407 manually annotated images collected from 18 road scenes and 12 fixed traffic cameras in Beijing, China. Across three independent random seeds, FL-YOLO achieves a Mask AP@[0.50:0.95] of 0.639 ± 0.028, a Mask AP@0.50 of 0.821 ± 0.0304, a Mask AP@0.75 of 0.594 ± 0.018, and a Foreground IoU of 0.716 ± 0.027 on the in-domain validation set. On the independent external benchmark of 80 web-sourced images, FL-YOLO achieves a Foreground IoU of 0.5147 and a Foreground Dice of 0.6796. The framework processes images at 79.39 FPS on an NVIDIA RTX 3090.

The predicted curb masks can be converted into geometric exposure ratios for three-level waterlogging severity assessment. This downstream procedure assumes a fixed camera viewpoint, an aligned dry reference frame, and sufficient curb visibility; it estimates relative curb exposure rather than absolute water depth.

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

CurbFloodSeg is divided into 1,266 training images and 141 validation images using a group-wise protocol rather than random image-level sampling. Images from the same camera, road scene, rainfall event, or continuous video sequence are assigned exclusively to one subset, and consecutive or near-duplicate frames remain within the same subset. The validation set is used for in-domain evaluation and model comparison. The external benchmark is reserved for zero-shot cross-domain evaluation and is excluded from training, hyperparameter tuning, and model selection.

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

Images in CurbFloodSeg and the independent external benchmark were manually annotated with LabelMe. Three annotators independently delineated curbstone masks, after which two experienced experts reviewed the annotations and resolved discrepancies by consensus.

Access to CurbFloodSeg is temporarily restricted while compliance and privacy reviews are in progress. Requests for controlled access may be directed to the corresponding author and will be considered subject to the applicable approvals. Data split files and non-sensitive metadata will be released after completion of the review process.

## Models

The source code and trained FL-YOLO weights are provided in this repository. Please ensure that the model configuration and checkpoint correspond to the same code version when reproducing the reported results. The reported baseline comparisons use YOLOv11-Seg, YOLO12-Seg, YOLO26-Seg, and DDRNet. For YOLO12-Seg, the available m-scale detection checkpoint (`yolo12m.pt`) is used for initialization because an official segmentation-pretrained checkpoint was not available. YOLO26-Seg uses the official m-scale segmentation checkpoint (`yolo26m-seg.pt`).

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
