import cv2
import numpy as np

def draw_boxes2d_on_bev(bev_img, bboxes, color=(255, 0, 0), thickness=2,
                        x_range=(-40, 40), y_range=(0, 70), voxel_size=0.4):
    """
    在 BEV 图像上绘制 2D 检测框。

    参数:
        bev_img (np.ndarray): 输入的 BEV 图像，shape=[H, W, 3]，uint8
        bboxes (Tensor or np.ndarray): [N, 4]，每行是 (x1, y1, x2, y2)，图像平面坐标
        color (Tuple[int]): 绘制颜色，默认红色
        thickness (int): 线条宽度
        x_range, y_range (Tuple[float]): BEV 空间范围
        voxel_size (float): 体素大小，影响 BEV 分辨率

    返回:
        bev_img (np.ndarray): 绘制后的图像
    """
    H_bev = int((y_range[1] - y_range[0]) / voxel_size)
    W_bev = int((x_range[1] - x_range[0]) / voxel_size)

    for box in bboxes:
        x1, y1, x2, y2 = box

        # 计算中心点与尺寸
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        # 映射到 BEV 图像上的坐标（注意顺序：x 对应列，y 对应行）
        x_bev = int((cx - x_range[0]) / voxel_size)
        y_bev = int((cy - y_range[0]) / voxel_size)
        w_bev = int(w / voxel_size)
        h_bev = int(h / voxel_size)

        x1b = max(0, x_bev - w_bev // 2)
        y1b = max(0, y_bev - h_bev // 2)
        x2b = min(W_bev - 1, x_bev + w_bev // 2)
        y2b = min(H_bev - 1, y_bev + h_bev // 2)

        cv2.rectangle(bev_img, (x1b, y1b), (x2b, y2b), color, thickness)

    return bev_img
