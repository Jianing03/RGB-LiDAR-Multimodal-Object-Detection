import cv2
import numpy as np
import os

# === 文件路径配置 ===
# input_dir = "D:/Desktop/easy"
# output_dir = "D:/Desktop/easy_processed"
# input_dir = "D:/Desktop/moderate"
# output_dir = "D:/Desktop/modarate_processed"
input_dir = "D:/Desktop/hard"
output_dir = "D:/Desktop/hard_processed"
os.makedirs(output_dir, exist_ok=True)

# 模型颜色和标题对应
model_infos = {
    "early": {"color": (255, 128, 0), "title": "图 4.9 Hard 样本早期融合模型"},
    "feature": {"color": (0, 0, 255), "title": "图 4.10 Hard 样本中期融合模型"},
    "result": {"color": (0, 255, 0), "title": "图 4.11 Hard 样本晚期融合模型"},
}

# ✅ 统一放大区域（x, y, w, h）
PATCH_REGION = (495, 140, 290, 130)

def extract_boxes_by_color(img, color_bgr, tol=40):
    lower = np.array([max(c - tol, 0) for c in color_bgr])
    upper = np.array([min(c + tol, 255) for c in color_bgr])
    mask = cv2.inRange(img, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 10]
    return boxes

def process_image(img_path, color, title_text, output_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"图像读取失败: {img_path}")
        return

    h, w = img.shape[:2]
    boxes = extract_boxes_by_color(img, color)

    x1, y1, pw, ph = PATCH_REGION
    x2, y2 = x1 + pw, y1 + ph

    img_boxed = img.copy()
    cv2.rectangle(img_boxed, (x1, y1), (x2, y2), (0, 0, 255), 2)

    crop = img[y1:y2, x1:x2].copy()
    for (bx, by, bw, bh) in boxes:
        cx, cy = bx + bw // 2, by + bh // 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            cv2.rectangle(crop, (bx - x1, by - y1), (bx + bw - x1, by + bh - y1), color, 2)

    scale = min(w / crop.shape[1], h / crop.shape[0])
    resized = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)), interpolation=cv2.INTER_LANCZOS4)

    canvas_w = max(w, resized.shape[1])
    combined = np.ones((h + resized.shape[0], canvas_w, 3), dtype=np.uint8) * 255
    combined[:h, :w] = img_boxed
    offset_x = (canvas_w - resized.shape[1]) // 2
    combined[h:h + resized.shape[0], offset_x:offset_x + resized.shape[1]] = resized

    # 红线连接逻辑
    red = (0, 0, 255)
    pt_left = (x1, (y1 + y2) // 2)
    pt_right = (x2, (y1 + y2) // 2)
    pt_zoom_left = (offset_x, h)
    pt_zoom_right = (offset_x + resized.shape[1] - 1, h)

    cv2.line(combined, pt_left, pt_zoom_left, red, 2)
    cv2.line(combined, pt_right, pt_zoom_right, red, 2)

    cv2.imwrite(output_path, combined)
    print(f"✅ 已保存: {output_path}")

# === 主调入口 ===
if __name__ == "__main__":
    # base_id = "000179"
    # base_id = "002356"
    base_id = "000401"  # 改为当前样本编号

    for key, info in model_infos.items():
        img_path = os.path.join(input_dir, f"{base_id}_{key}.jpg")
        output_path = os.path.join(output_dir, f"{base_id}_{key}_processed_final.jpg")
        process_image(img_path, info["color"], info["title"], output_path)
