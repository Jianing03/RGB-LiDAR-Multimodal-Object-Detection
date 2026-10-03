# data_fusion.py
import numpy as np
import torch
from utils.project_lidar import project_lidar_to_image
def fuse_data(image, lidar, calib):
    """
    image: [3, H, W] 的 Tensor
    lidar: (N, 3) 的 numpy 数组
    calib: 包含 P2, Tr_velo_to_cam, R0_rect 的 dict
    返回: [4, H, W] 的 Tensor，拼接了深度图
    """
    _, H, W = image.shape
    if "P2" not in calib or "Tr_velo_to_cam" not in calib or "R0_rect" not in calib:
        zero_depth = torch.zeros((1, H, W), dtype=image.dtype, device=image.device)
        return torch.cat([image, zero_depth], dim=0)

    depth_map = project_lidar_to_image(lidar, calib, W=W, H=H)
    if depth_map is None or depth_map.shape != (H, W):
        depth_map = np.zeros((H, W), dtype=np.float32)

    depth_tensor = torch.from_numpy(depth_map).to(image.device).unsqueeze(0)  # [1, H, W]
    depth_tensor = torch.clamp(depth_tensor / 70.0, 0.0, 1.0)  # 稳定归一化
    fused = torch.cat([image, depth_tensor], dim=0)  # [4, H, W]
    return fused
