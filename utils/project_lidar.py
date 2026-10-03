import numpy as np

def project_lidar_to_image(lidar: np.ndarray, calib: dict, W: int, H: int) -> np.ndarray:
    """
    将激光点云投影到图像上，生成深度图（几何校正版本）
    - lidar: 原始激光点云 (N, 3)
    - calib: 包含 P2, Tr_velo_to_cam, R0_rect 的 dict
    - W, H: 输出图像大小
    返回: (H, W) 深度图（Z值）
    """
    N = lidar.shape[0]
    lidar_homo = np.concatenate([lidar, np.ones((N, 1))], axis=1).T  # shape: [4, N]

    # 1. 从校准字典中取出矩阵
    Tr = calib["Tr_velo_to_cam"]  # (3, 4)
    R0 = calib["R0_rect"]         # (3, 3)
    P2 = calib["P2"]              # (3, 4)

    # 2. 激光雷达点 → 相机坐标
    cam_xyz = R0 @ (Tr @ lidar_homo)  # shape: [3, N]

    # 3. 去除 Z <= 0 的点（在相机后方）
    valid = cam_xyz[2, :] > 0
    cam_xyz = cam_xyz[:, valid]

    # 4. 投影到图像平面
    uv_depth = P2 @ np.vstack([cam_xyz, np.ones((1, cam_xyz.shape[1]))])  # [3, N]
    uv = uv_depth[:2] / uv_depth[2:]  # 归一化 (2, N)
    uv = uv.T  # [N, 2]
    z = cam_xyz[2]  # 深度Z

    # 5. 生成深度图
    depth = np.zeros((H, W), dtype=np.float32)
    for i in range(uv.shape[0]):
        u, v = int(uv[i, 0]), int(uv[i, 1])
        if 0 <= u < W and 0 <= v < H:
            if depth[v, u] == 0 or z[i] < depth[v, u]:
                depth[v, u] = z[i]  # 深度小者优先（最近点）

    return depth
