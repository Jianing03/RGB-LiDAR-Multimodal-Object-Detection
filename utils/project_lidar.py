import numpy as np

def project_lidar_to_image(lidar: np.ndarray, calib: dict, W: int, H: int) -> np.ndarray:
    """
    Project LiDAR points onto the image using rectified camera geometry.
    - lidar: raw LiDAR points (N, 3).
    - calib: dictionary containing P2, Tr_velo_to_cam, and R0_rect.
    - W, H: output image width and height.
    Return: (H, W) depth map containing camera Z values.
    """
    N = lidar.shape[0]
    lidar_homo = np.concatenate([lidar, np.ones((N, 1))], axis=1).T  # shape: [4, N]

    # 1. Read matrices from the calibration dictionary.
    Tr = calib["Tr_velo_to_cam"]  # (3, 4)
    R0 = calib["R0_rect"]         # (3, 3)
    P2 = calib["P2"]              # (3, 4)

    # 2. Transform LiDAR points into rectified camera coordinates.
    cam_xyz = R0 @ (Tr @ lidar_homo)  # shape: [3, N]

    # 3. Remove points with Z <= 0, on or behind the camera plane.
    valid = cam_xyz[2, :] > 0
    cam_xyz = cam_xyz[:, valid]

    # 4. Project onto the image plane.
    uv_depth = P2 @ np.vstack([cam_xyz, np.ones((1, cam_xyz.shape[1]))])  # [3, N]
    uv = uv_depth[:2] / uv_depth[2:]  # Normalize homogeneous image coordinates (2, N).
    uv = uv.T  # [N, 2]
    z = cam_xyz[2]  # Camera Z depth.

    # 5. Build the depth map.
    depth = np.zeros((H, W), dtype=np.float32)
    for i in range(uv.shape[0]):
        u, v = int(uv[i, 0]), int(uv[i, 1])
        if 0 <= u < W and 0 <= v < H:
            if depth[v, u] == 0 or z[i] < depth[v, u]:
                depth[v, u] = z[i]  # Keep the nearest point at each pixel.

    return depth
