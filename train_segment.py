import torch
import numpy as np
import random

# ✅ 设置固定随机种子，确保结果可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # 调用

from ultralytics import YOLO


# ✅ 加载分割模型权重
model = YOLO("/home/lak/project/sy/ultralytics-main/ultralytics/cfg/models/11/yolo11m-seg.yaml").load("/home/lak/project/sy/ultralytics-main/ultralytics/cfg/yolo11m-seg.pt")  

# ✅ 训练参数
train_params = {
    'task': 'segment',
    'data': "/home/lak/project/sy/ultralytics-main/ultralytics/cfg/datasets/tooths.yaml",  # 请替换为你自己的数据集配置路径
    'epochs': 100,
    'imgsz': 640,
    'batch': 8,
    'device': 7,  # 使用 GPU 0，如需 CPU 可设置为 "cpu"
    'workers': 8,
    'project': "runs/segment",  # 训练日志保存位置
    'name': "yolo11m-seg-train26",
    'exist_ok': True,  # 若目录已存在，不报错
    'save': True,
    'val': True,
    'plots': True,
    # 你可以继续添加其余的数据增强参数...
}

# ✅ 开始训练
results = model.train(**train_params)
