import cv2
import numpy as np

def draw_boxes2d_on_bev(bev_img, bboxes, color=(255, 0, 0), thickness=2,
                        x_range=(-40, 40), y_range=(0, 70), voxel_size=0.4):
    """
    Draw 2D boxes on a BEV image.

    Args:
        bev_img (np.ndarray): Input BEV image, shape [H, W, 3], dtype uint8.
        bboxes (Tensor or np.ndarray): [N, 4], each row (x1, y1, x2, y2), interpreted in the BEV spatial frame.
        color (Tuple[int]): OpenCV BGR color; defaults to blue.
        thickness (int): Line thickness.
        x_range, y_range (Tuple[float]): BEV spatial ranges.
        voxel_size (float): Voxel size, which determines BEV resolution.

    Returns:
        bev_img (np.ndarray): Annotated image.
    """
    H_bev = int((y_range[1] - y_range[0]) / voxel_size)
    W_bev = int((x_range[1] - x_range[0]) / voxel_size)

    for box in bboxes:
        x1, y1, x2, y2 = box

        # Compute box centers and dimensions.
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        # Map to BEV image coordinates: x indexes columns, y indexes rows.
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
