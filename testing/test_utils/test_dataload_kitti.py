import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

CLASS_DICT = {"Car": 0}  # Keep only the Car class.
"""
# This legacy helper reads only P2, the camera 2 projection matrix.
# Suitable for projecting points already expressed in rectified camera coordinates.
def read_kitti_calib(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    P2_line = [line for line in lines if line.startswith("P2:")][0]
    P2_values = list(map(float, P2_line.strip().split()[1:]))
    P2 = np.array(P2_values).reshape(3, 4)
    return {"P2": P2}
"""
def read_kitti_calib(calib_path):
    """ Read KITTI calibration and return P2, Tr_velo_to_cam, and R0_rect. """
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

        # Set the default image size to 256x256.
        self.input_size = (256, 256)
        # self.input_size = (192, 192)

        self.transform = transform or T.Compose([
            T.Resize(self.input_size),
            T.ToTensor()
        ])

        print(f"[KITTI Dataset] Loaded split={split_key}, samples={len(self.ids)}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]

        img_path = os.path.join(self.image_dir, f"{sample_id}.png")
        lidar_path = os.path.join(self.lidar_dir, f"{sample_id}.bin")
        label_path = os.path.join(self.label_dir, f"{sample_id}.txt")
        calib_path = os.path.join(self.calib_dir, f"{sample_id}.txt")

        # Read the original image dimensions before applying transforms.
        img_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_pil.size  # Original image dimensions.
        image = self.transform(img_pil)  # Apply the image transform; the default resizes to 256x256.

        # Compute scale factors from the original dimensions to the input size.
        scale_w = self.input_size[0] / orig_w
        scale_h = self.input_size[1] / orig_h

        # ★ Load calibration before transforming points.
        calib = read_kitti_calib(calib_path)

        # lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]

        # ★ Transform LiDAR points into rectified camera coordinates.
        lidar_homo = np.hstack((lidar, np.ones((lidar.shape[0], 1))))  # [N,4]
        Tr = calib['Tr_velo_to_cam']  # (3,4)
        R0 = calib['R0_rect']  # (3,3)
        cam_xyz = (R0 @ (Tr @ lidar_homo.T)).T  # [N,3]

        # ★ Compute cropping masks in camera coordinates.
        mask_x = (cam_xyz[:, 0] >= -40) & (cam_xyz[:, 0] <= 40)  # Lateral range.
        mask_z = (cam_xyz[:, 2] >= 0) & (cam_xyz[:, 2] <= 70)  # Forward depth range.
        lidar = lidar[mask_x & mask_z]  # ★ Apply the camera-space mask to the original LiDAR points.

        bboxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    fields = line.strip().split()
                    cls_name = fields[0]
                    if cls_name != "Car":
                        continue
                    # KITTI box coordinates refer to the original image dimensions.
                    x1, y1, x2, y2 = map(float, fields[4:8])
                    w, h = x2 - x1, y2 - y1
                    if w < 1e-2 or h < 1e-2:
                        continue  # Skip invalid boxes.
                    # Scale box coordinates to the input dimensions.
                    x1_scaled = x1 * scale_w
                    y1_scaled = y1 * scale_h
                    x2_scaled = x2 * scale_w
                    y2_scaled = y2 * scale_h
                    bboxes.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled])
                    labels.append(CLASS_DICT[cls_name])

        if len(bboxes) == 0:
            return None  # Skip samples with no valid Car boxes.

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
        root_dir="./data/KITTI",
        split_json="./data/KITTI/kitti_split_6_2_2.json",
        split_key="train"
    )
    print("Sample count:", len(dataset))
    print(dataset[0])
