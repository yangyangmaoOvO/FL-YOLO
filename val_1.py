from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json

# 替换成你保存的路径
gt_path = '/home/lak/project/sy/ultralytics-main/ultralytics/cfg/datasets/tooths/json'  # val集标注（COCO格式）
dt_path = '/home/lak/project/sy/ultralytics-main/runs/segment/val25/predictions.json'  # YOLO 保存的预测结果

# 加载 COCO GT 和预测
coco_gt = COCO(gt_path)
coco_dt = coco_gt.loadRes(dt_path)

# 创建评估器
coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')  # 或 'bbox' 如果是检测

# 设定 IoU 阈值从 0.75 到 0.95
coco_eval.params.iouThrs = np.linspace(0.75, 0.95, 5)
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# 获取 mAP@0.75:0.95（mean over these 5 IoU levels）
map_075_095 = coco_eval.stats[0]
print(f"\n🎯 mAP@0.75:0.95: {map_075_095:.4f}")
