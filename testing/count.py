import os
import json

# 配置路径
KITTI_ROOT = "D:/KITTI"
LABEL_DIR = os.path.join(KITTI_ROOT, "label_2")
SPLIT_PATH = os.path.join(KITTI_ROOT, "kitti_split_6_2_2.json")

# 加载测试集索引
with open(SPLIT_PATH, "r") as f:
    split_data = json.load(f)
test_ids = split_data["test"]

# 难度判定规则（参考 KITTI 官方标准）
def get_difficulty(obj):
    trunc = float(obj[1])
    occl = int(obj[2])
    height = float(obj[7]) - float(obj[5])

    if trunc <= 0.15 and occl == 0 and height >= 40:
        return "Easy"
    elif trunc <= 0.3 and occl <= 1 and height >= 25:
        return "Moderate"
    elif trunc <= 0.5 and occl <= 2 and height >= 25:
        return "Hard"
    else:
        return "Unknown"

# 初始化统计计数器
difficulty_count = {"Easy": 0, "Moderate": 0, "Hard": 0, "Unknown": 0}

# 遍历测试集每张图片的 label 文件
for img_id in test_ids:
    label_path = os.path.join(LABEL_DIR, f"{img_id}.txt")
    if not os.path.exists(label_path):
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()

    difficulties = set()
    for line in lines:
        fields = line.strip().split()
        obj_type = fields[0]
        if obj_type != "Car":  # 只关注 Car 类别
            continue
        diff = get_difficulty(fields)
        difficulties.add(diff)

    # 判断整张图的最“困难”难度等级（以最大难度为准）
    if "Hard" in difficulties:
        difficulty_count["Hard"] += 1
    elif "Moderate" in difficulties:
        difficulty_count["Moderate"] += 1
    elif "Easy" in difficulties:
        difficulty_count["Easy"] += 1
    else:
        difficulty_count["Unknown"] += 1

# 输出结果
print("📊 测试集中不同难度的图像数量：")
for k, v in difficulty_count.items():
    print(f"  {k:<8}: {v}")
