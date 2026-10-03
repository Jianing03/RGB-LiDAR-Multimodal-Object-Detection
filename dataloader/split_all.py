import os
import json
import random
from glob import glob

# 路径配置
KITTI_ROOT = "D:/KITTI"
IMAGE_DIR = os.path.join(KITTI_ROOT, "image_2")
SPLIT_PATH = os.path.join(KITTI_ROOT, "kitti_split_6_2_2.json")

# 获取所有图片编号（去掉扩展名）
image_paths = glob(os.path.join(IMAGE_DIR, "*.png"))
image_ids = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

# ✅ 打乱顺序（设定随机种子确保可复现）
random.seed(42)
random.shuffle(image_ids)

# 检查数量
assert len(image_ids) == 7481, f"数量异常，找到了 {len(image_ids)} 张图像，预期为 7481"

# ✅ 按比例划分：60% 训练，20% 验证，20% 测试
num_total = len(image_ids)
num_train = int(num_total * 0.6)
num_val   = int(num_total * 0.2)
num_test  = num_total - num_train - num_val  # 剩下的为测试集

train_ids = image_ids[:num_train]
val_ids   = image_ids[num_train:num_train + num_val]
test_ids  = image_ids[num_train + num_val:]

# ✅ 保存到 JSON 文件
split_dict = {
    "train": train_ids,
    "val": val_ids,
    "test": test_ids
}
with open(SPLIT_PATH, "w") as f:
    json.dump(split_dict, f, indent=2)

print(f"✅ 划分完成：训练集 {len(train_ids)}，验证集 {len(val_ids)}，测试集 {len(test_ids)}")
print(f"📁 JSON 保存至: {SPLIT_PATH}")
