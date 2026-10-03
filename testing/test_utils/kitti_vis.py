import cv2

def draw_boxes_with_scores(image, boxes, scores, label_prefix="", color=(0, 255, 0)):
    """
    Draw boxes with confidence scores on an image.

    :param image: Original image (BGR)
    :param boxes: List of [x1, y1, x2, y2]
    :param scores: List of float
    :param label_prefix: Optional label prefix, e.g. 'E', 'F', or 'R'
    :param color: Drawing color (B, G, R)
    """
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        label = f"{label_prefix}:{score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
