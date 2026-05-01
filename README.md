# <div style="text-align: center;">Curb-Guided Visual Perception and Multi-Scale Feature Learning for Urban Waterlogging Semantic Segmentation</div>


## Introduction 

Urban waterlogging induced by extreme precipitation poses significant threats to transportation safety
and municipal resilience. While conventional monitoring relies on costly sensor networks with limited
spatial coverage, efficient visual perception solutions remain underexplored. This work introduces FL-
YOLO, a specialized, curb-guided lightweight segmentation framework designed to quantify inunda-
tion depth by leveraging the standardized geometric height of road curbstones as a static reference. To
resolve the critical challenges of boundary ambiguity and structural fragmentation in flooded scenes,
we propose two pivotal components: the Feature Complementary Mapping (FCM) module for
preserving fine-grained spatial details in shallow hierarchies, and the Local-Region Self-Attention
(LRSA) module for parsing micro-textural dependencies at the water-concrete interface. Further-
more, a domain-specific dataset, CurbFloodSeg, is established to support high-precision model
training and benchmarking. Extensive experiments demonstrate that FL-YOLO achieves an mIoU of
72.03%, an mDice of 80.84%, and a pixel accuracy of 81.99%, consistently outperforming state-of-the-
art baselines. The proposed framework provides a robust, real-time, and scalable approach for urban
flood monitoring, exhibiting exceptional generalization across diverse and complex environments

## Document
### Recommended Environment

- [x] torch 2.7.0
- [x] torchvision 0.22.0
- [x] numpy 1.26.3
......

You only need to install the corresponding libraries to refer to train.by and val.by for training and validation.



## Dataset
Like [curb.yaml](./curb.yaml)  You can use your own dataset.

<details open>
  <summary><b>File structure</b></summary>

```
Your dataset
├── ...
├── train
|   ├── rgb
|   |   ├── images
|   |   ├── labels
|   ├── depth
|   |   ├── images
|   |   ├── labels
└── val
|   ├── rgb
|   |   ├── images
|   |   ├── labels
|   ├── depth
|   |   ├── images
|   |   ├── labels
```

</details>



## Citation

If you find our code, models, or the CurbFloodSeg dataset useful in your research, please consider citing our manuscript currently under review at *The Visual Computer*:

**Plain Text:**
Hao Xiao, Yang Shen, Yuming Zhang, Tingmin Liu, Zhifeng Hu, Yuncan Gao, Runlong Cao, and Ying Zang. "Curb-Guided Visual Perception and Multi-Scale Feature Learning for Urban Waterlogging Semantic Segmentation." *Submitted to The Visual Computer*, 2026.

**BibTeX:**
```bibtex
@article{xiao2026curb,
  title={Curb-Guided Visual Perception and Multi-Scale Feature Learning for Urban Waterlogging Semantic Segmentation},
  author={Xiao, Hao and Shen, Yang and Zhang, Yuming and Liu, Tingmin and Hu, Zhifeng and Gao, Yuncan and Cao, Runlong and Zang, Ying},
  journal={Submitted to The Visual Computer},
  year={2026}
}
```

## Acknowledgement
Part of the code is adapted from Ultralytics: [Ultralytics](https://github.com/ultralytics/ultralytics). We thank all the authors for their contributions.
