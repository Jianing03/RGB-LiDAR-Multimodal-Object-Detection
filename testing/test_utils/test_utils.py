# 保存路径：test_utils.py
import os
import torch
import numpy as np
def get_difficulty(bbox, trunc, occ):
    h = bbox[3] - bbox[1]
    if occ == 0 and trunc <= 0.15 and h >= 40:
        return "easy"
    elif occ <= 1 and trunc <= 0.3 and h >= 25:
        return "moderate"
    elif occ <= 2 and trunc <= 0.5 and h >= 25:
        return "hard"
    else:
        return "invalid"

def get_image_difficulty(label_lines):
    # 计算图像中所有目标的最大难度
    difficulty_order = {"easy": 0, "moderate": 1, "hard": 2}
    max_diff = None
    for line in label_lines:
        fields = line.split()
        if fields[0] != "Car":
            continue
        bbox = list(map(float, fields[4:8]))
        trunc, occ = float(fields[1]), int(fields[2])
        diff = get_difficulty(bbox, trunc, occ)
        if diff == "invalid":
            continue
        if max_diff is None or difficulty_order[diff] > difficulty_order[max_diff]:
            max_diff = diff
    return max_diff  # 返回 easy / moderate / hard / None

def split_by_difficulty(dataset, save_path=None):

    easy, moderate, hard = [], [], []

    for i in range(len(dataset)):
        sample = dataset[i]
        if sample is None:
            continue
        label_path = os.path.join(dataset.label_dir, f"{sample['id']}.txt")
        if not os.path.exists(label_path):
            continue
        with open(label_path, 'r') as f:
            lines = f.readlines()

        max_diff = get_image_difficulty(lines)

        if max_diff == "easy" :
            easy.append(i)
        elif max_diff == "moderate" :
            moderate.append(i)
        elif max_diff == "hard" :
            hard.append(i)

    if save_path:
        np.savez(save_path, easy=easy, moderate=moderate, hard=hard)
    return easy, moderate, hard

