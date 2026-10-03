import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

CLASS_DICT = {"Car": 0}  # 只保留 Car
"""
# 只提取了 P2 投影矩阵（相机2的投影矩阵，常用于图像投影）
# 使用简单直接，适合只进行3D 点 → 图像投影的场景（比如做深度图生成）
def read_kitti_calib(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    P2_line = [line for line in lines if line.startswith("P2:")][0]
    P2_values = list(map(float, P2_line.strip().split()[1:]))
    P2 = np.array(P2_values).reshape(3, 4)
    return {"P2": P2}
"""
def read_kitti_calib(calib_path):
    """ 读取 KITTI 标定文件，返回 P2, Tr_velo_to_cam, R0_rect """
    with open(calib_path, 'r') as f:
        lines = f.readlines()

    def parse_matrix(line):
        return np.array(list(map(float, line.strip().split()[1:])))

    calib = {}
    for line in lines:
        if line.startswith("P2:"):
            calib["P2"] = parse_matrix(line).reshape(3, 4)
        elif line.startswith("R0_rect:"):
            calib["R0_rect"] = parse_matrix(line).reshape(3, 3)
        elif line.startswith("Tr_velo_to_cam:"):
            calib["Tr_velo_to_cam"] = parse_matrix(line).reshape(3, 4)

    return calib


class KITTIDataset(Dataset):
    def __init__(self, root_dir, split_json, split_key="train", transform=None):
        import json
        with open(split_json, 'r') as f:
            split_data = json.load(f)
        self.ids = split_data[split_key]

        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "image_2")
        self.lidar_dir = os.path.join(root_dir, "velodyne")
        self.label_dir = os.path.join(root_dir, "label_2")
        self.calib_dir = os.path.join(root_dir, "calib")

        # 统一调整图像尺寸为256x256
        self.input_size = (256, 256)
        # self.input_size = (192, 192)

        self.transform = transform or T.Compose([
            T.Resize(self.input_size),
            T.ToTensor()
        ])

        print(f"[KITTI Dataset] 加载 split={split_key}，样本数={len(self.ids)}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]

        img_path = os.path.join(self.image_dir, f"{sample_id}.png")
        lidar_path = os.path.join(self.lidar_dir, f"{sample_id}.bin")
        label_path = os.path.join(self.label_dir, f"{sample_id}.txt")
        calib_path = os.path.join(self.calib_dir, f"{sample_id}.txt")

        # 先打开图像获取原始尺寸，再进行转换
        img_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_pil.size  # 原始图像尺寸
        image = self.transform(img_pil)  # 转换为256x256尺寸

        # 计算缩放因子：原始尺寸 → 256x256
        scale_w = self.input_size[0] / orig_w
        scale_h = self.input_size[1] / orig_h

        # ★ 先加载标定文件
        calib = read_kitti_calib(calib_path)

        # lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]

        # ★ LiDAR → 相机坐标系
        lidar_homo = np.hstack((lidar, np.ones((lidar.shape[0], 1))))  # [N,4]
        Tr = calib['Tr_velo_to_cam']  # (3,4)
        R0 = calib['R0_rect']  # (3,3)
        cam_xyz = (R0 @ (Tr @ lidar_homo.T)).T  # [N,3]

        # ★ 在相机坐标系裁剪
        mask_x = (cam_xyz[:, 0] >= -40) & (cam_xyz[:, 0] <= 40)  # 左右
        mask_z = (cam_xyz[:, 2] >= 0) & (cam_xyz[:, 2] <= 70)  # 前后
        lidar = lidar[mask_x & mask_z]  # ★ 裁剪原始 LiDAR 点云（不是 cam_xyz）

        bboxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    fields = line.strip().split()
                    cls_name = fields[0]
                    if cls_name != "Car":
                        continue
                    # KITTI 标注坐标基于原始图像尺寸
                    x1, y1, x2, y2 = map(float, fields[4:8])
                    w, h = x2 - x1, y2 - y1
                    if w < 1e-2 or h < 1e-2:
                        continue  # 排除无效框
                    # 缩放边界框坐标到 256x256
                    x1_scaled = x1 * scale_w
                    y1_scaled = y1 * scale_h
                    x2_scaled = x2 * scale_w
                    y2_scaled = y2 * scale_h
                    bboxes.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled])
                    labels.append(CLASS_DICT[cls_name])

        if len(bboxes) == 0:
            return None  # 直接跳过无效样本

        return {
            "id": sample_id,
            "image": image,
            "lidar": lidar,
            "calib": calib,
            "bboxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


if __name__ == '__main__':
    dataset = KITTIDataset(
        root_dir="D:/KITTI",
        split_json="D:/KITTI/kitti_split_6_2_2.json",
        split_key="train"
    )
    print("样本数:", len(dataset))
    print(dataset[0])
