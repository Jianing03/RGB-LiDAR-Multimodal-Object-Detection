# vis_result.py
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from training.result_train import build_result_components
from testing.test_utils.load_yaml import load_yaml
from testing.test_utils.test_dataload_kitti import KITTIDataset
from testing.test_utils.kitti_vis import draw_boxes_with_scores
# from utils.yolo_postprocess import decode_yolo_output
from testing.test_utils.test_yolo_postprocess import decode_yolo_output

# Configure paths.
CKPT_PATH = "./results/result_saved_model_best_f1.pth"
CFG_PATH = "./configs/result_train_config.yaml"
SPLIT_PATH = "./test_indices.npz"
KITTI_ROOT = "./data/KITTI"
SAVE_ROOT = "./results/result_test_picture"
SAVE_DIRS = {
    "easy": os.path.join(SAVE_ROOT, "easy"),
    "moderate": os.path.join(SAVE_ROOT, "moderate"),
    "hard": os.path.join(SAVE_ROOT, "hard"),
}
os.makedirs(SAVE_ROOT, exist_ok=True)
for d in SAVE_DIRS.values():
    os.makedirs(d, exist_ok=True)

# Load the configuration and model.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = load_yaml(CFG_PATH)
model, _, _, _, _, _ = build_result_components(cfg, device)
ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

# Load the test split and sample indices.
dataset = KITTIDataset(KITTI_ROOT, os.path.join(KITTI_ROOT, "kitti_split_6_2_2.json"), "test")
split = np.load(SPLIT_PATH)
subset_indices = {
    "easy": split["easy"][:20],
    "moderate": split["moderate"][:20],
    "hard": split["hard"][:20],
}

# Process each image.
for diff_level, indices in subset_indices.items():
    for idx in tqdm(indices, desc=f"Processing {diff_level}"):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        lidar = [sample["lidar"]]
        calib = [sample["calib"]]

        with torch.no_grad():
            feats_rgb = model.bb.image_forward(image)
            feats_lid = model.bb.lidar_forward(lidar, calib)
            out_rgb = model.h_rgb([feats_rgb["P3_img"], feats_rgb["P4_img"], feats_rgb["P5_img"]])
            out_lid = model.h_lidar([feats_lid["P3_lidar"], feats_lid["P4_lidar"], feats_lid["P5_lidar"]])

        # Decode predictions, merge branches, and apply NMS.
        # ✅ Use consistent strides across branches.
        strides = [8, 16, 32]
        pred_boxes = []

        for pred, anchor, stride in zip(out_rgb, model.anchors, strides):
            pred_boxes += decode_yolo_output(pred, anchor, stride=stride, conf_thresh=0.3, nms_thresh=0.5)[0]

        for pred, anchor, stride in zip(out_lid, model.anchors, strides):
            pred_boxes += decode_yolo_output(pred, anchor, stride=stride, conf_thresh=0.3, nms_thresh=0.5)[0]

        # Apply NMS to merged predictions across feature levels.
        from torchvision.ops import nms

        if len(pred_boxes) > 0:
            boxes_tensor = torch.tensor([b["box"] for b in pred_boxes], dtype=torch.float32)
            scores_tensor = torch.tensor([b["score"] for b in pred_boxes], dtype=torch.float32)
            keep = nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
            pred_boxes = [pred_boxes[i] for i in keep]
        else:
            pred_boxes = []

        # Get the original image dimensions.
        H_in, W_in = sample["image"].shape[1:]
        raw_img_path = os.path.join(KITTI_ROOT, "image_2", sample["id"] + ".png")
        raw_img = cv2.imread(raw_img_path)
        if raw_img is None:
            continue
        H_raw, W_raw = raw_img.shape[:2]

        # Rescale boxes and visualize predictions.
        boxes = []
        scores = []
        for box in pred_boxes:  # ✅ Read each prediction's box and score.
            x1, y1, x2, y2 = box["box"]
            score = box["score"]
            x1 *= W_raw / W_in
            x2 *= W_raw / W_in
            y1 *= H_raw / H_in
            y2 *= H_raw / H_in
            boxes.append([x1, y1, x2, y2])
            scores.append(score)

        draw_boxes_with_scores(
            image=raw_img,
            boxes=boxes,
            scores=scores,
            label_prefix="R",
            color=(0, 255, 0)
        )

        save_name = os.path.basename(sample["id"]) + ".jpg"
        save_path = os.path.join(SAVE_DIRS[diff_level], save_name)
        cv2.imwrite(save_path, raw_img)
