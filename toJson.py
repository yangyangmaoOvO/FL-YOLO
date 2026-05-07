import os
import json
from tqdm import tqdm

# YOLO标签路径
label_path = "/home/lak/project/sy/ultralytics-main/ultralytics/cfg/datasets/tooths/labels/val"
image_path = "/home/lak/project/sy/ultralytics-main/ultralytics/cfg/datasets/tooths/images/val"
output_json_path = "/home/lak/project/sy/ultralytics-main/ultralytics/cfg/datasets/toothsval.json"
categories = [{"id": 0, "name": "tooth"}]  # 根据你实际的类别修改

def yolo_to_coco():
    images = []
    annotations = []
    ann_id = 0
    img_id = 0

    for file in tqdm(os.listdir(label_path)):
        if not file.endswith(".txt"):
            continue
        txt_path = os.path.join(label_path, file)
        img_name = file.replace(".txt", ".jpg")  # 或 .png
        img_path = os.path.join(image_path, img_name)

        if not os.path.exists(img_path):
            continue

        h, w = 640, 640  # 如果你知道图像大小可以写成固定的，或用 cv2 读取

        images.append({
            "file_name": img_name,
            "height": h,
            "width": w,
            "id": img_id
        })

        with open(txt_path, "r") as f:
            for line in f.readlines():
                cls, cx, cy, bw, bh = map(float, line.strip().split())
                x = (cx - bw / 2) * w
                y = (cy - bh / 2) * h
                width = bw * w
                height = bh * h

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cls),
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                })
                ann_id += 1

        img_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json_path, "w") as f:
        json.dump(coco_format, f)

    print(f"✅ 转换完成，保存至：{output_json_path}")

yolo_to_coco()
