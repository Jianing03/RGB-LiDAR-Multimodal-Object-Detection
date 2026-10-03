import cv2

def draw_boxes_with_scores(image, boxes, scores, label_prefix="", color=(0, 255, 0)):
    """
    在图像上绘制带分数的框。

    :param image: 原始图像（BGR）
    :param boxes: List of [x1, y1, x2, y2]
    :param scores: List of float
    :param label_prefix: 可选的标签前缀（如 'E', 'F', 'R'）
    :param color: 绘图颜色 (B, G, R)
    """
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        label = f"{label_prefix}:{score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
