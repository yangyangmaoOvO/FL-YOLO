from ultralytics import YOLO
import os
import shutil
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import re

# 加载模型
model = YOLO("/home/lak/project/sy/ultralytics-main/runs/segment/yolo11m-seg-train25/weights/best.pt")

# 评估
metrics = model.val(
    data="/home/lak/project/sy/ultralytics-main/ultralytics/cfg/datasets/tooths.yaml",
    split="val",
    device='7',
    half=False,
    save_json=True
)

print(f"\n========== 分割评估指标 ==========")
print(f"metrics.seg.maps: {metrics.seg.maps}")
print(f"Mask mAP@0.5:     {metrics.seg.map50:.4f}")
print(f"Mask mAP@0.75:    {metrics.seg.map75:.4f}")
print(f"Mask mAP@0.75:0.95:{metrics.seg.map:.4f}")


# 预测并保存推理结果（包括边框、标签等）
save_dir = "runs/segment/predict48"
results = model.predict(
    source="/home/lak/project/sy/ultralytics-main/ultralytics/cfg/datasets/tooths/images/val",
    conf=0.45,
    iou=0.6,
    imgsz=640,
    half=False,
    device='7',
    max_det=300,
    save=True,
    save_txt=True,
)

# 手动保存掩码为png图片
os.makedirs(save_dir, exist_ok=True)
mask_save_dir = os.path.join(save_dir, "masks")
os.makedirs(mask_save_dir, exist_ok=True)

for result in results:
    img_name = os.path.basename(result.path)
    base_name = os.path.splitext(img_name)[0]
    orig_h, orig_w = result.orig_shape  # 获取原图高和宽

    if result.masks is not None and len(result.masks.data) > 0:
        # 合并多个 mask 为一个
        combined_mask = result.masks.data.sum(dim=0).clamp(0, 1).cpu().numpy()
        # 使用双线性插值以获得更平滑的边缘
        combined_mask = cv2.resize(combined_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        mask_img = (combined_mask * 255).astype(np.uint8)
    else:
        # 无预测目标，保存全黑图
        mask_img = np.zeros((orig_h, orig_w), dtype=np.uint8)

    mask_path = os.path.join(mask_save_dir, f"{base_name}.png")
    cv2.imwrite(mask_path, mask_img)

# 改进的像素级 IoU 计算函数
def compute_pixel_iou(pred_dir, gt_dir, threshold=0.5):
    # 创建更可靠的文件匹配机制
    pred_files = glob(os.path.join(pred_dir, "*.png"))
    gt_files = glob(os.path.join(gt_dir, "*.png"))
    
    # 建立文件名到路径的映射
    pred_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in pred_files}
    gt_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in gt_files}
    
    # 找出共同的文件
    common_keys = set(pred_dict.keys()) & set(gt_dict.keys())
    
    if not common_keys:
        print("⚠️ 未找到匹配的预测和真实掩码文件！")
        return 0.0
    
    print(f"找到 {len(common_keys)} 对匹配的掩码文件")
    
    iou_list = []
    missing_gt = []
    missing_pred = []
    
    # 处理所有预测文件
    for key in sorted(common_keys):
        pred_path = pred_dict[key]
        gt_path = gt_dict[key]
        
        # 读取并处理预测掩码
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        # 使用与保存时一致的阈值
        pred_bin = (pred_mask > 127).astype(np.uint8)
        
        # 读取并处理真实掩码
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        # 假设真实掩码是二值的，但进行安全检查
        if np.max(gt_mask) > 1:
            gt_bin = (gt_mask > 127).astype(np.uint8)
        else:
            gt_bin = gt_mask.astype(np.uint8)
        
        # 计算IoU
        intersection = np.logical_and(pred_bin, gt_bin).sum()
        union = np.logical_or(pred_bin, gt_bin).sum()
        
        # 如果真实掩码为空，检查预测是否也为空
        if union == 0:
            if intersection == 0:
                iou = 1.0  # 预测和真实都是空，IoU为1
            else:
                iou = 0.0  # 不可能的情况
        else:
            iou = intersection / union
        
        iou_list.append(iou)
    
    # 检查是否有不匹配的文件
    for key in set(pred_dict.keys()) - set(gt_dict.keys()):
        missing_gt.append(key)
    for key in set(gt_dict.keys()) - set(pred_dict.keys()):
        missing_pred.append(key)
    
    if missing_gt:
        print(f"⚠️ 有 {len(missing_gt)} 个预测文件没有对应的真实标签")
    if missing_pred:
        print(f"⚠️ 有 {len(missing_pred)} 个真实标签没有对应的预测")
    
    return np.mean(iou_list)

# 计算改进后的 IoU
val_mask_dir = "/home/lak/project/sy/ultralytics-main/ultralytics/cfg/datasets/tooths/val_masks"
pixel_iou = compute_pixel_iou(mask_save_dir, val_mask_dir)
print(f"改进后的 Mean Pixel IoU: {pixel_iou:.4f}")