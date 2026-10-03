# vis_early.py
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from testing.test_utils.test_yolo_postprocess import decode_yolo_output
from training.early_train import build_early_components
from testing.test_utils.load_yaml import load_yaml
from testing.test_utils.test_dataload_kitti import KITTIDataset
from testing.test_utils.kitti_vis import draw_boxes_with_scores

# Configure paths.
CKPT_PATH = "./results/early_saved_model_best_f1.pth"
CFG_PATH = "./configs/early_train_config.yaml"
SPLIT_PATH = "./test_indices.npz"
KITTI_ROOT = "./data/KITTI"
SAVE_ROOT = "./results/early_test_picture"
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
model, _, _, _, _, _ = build_early_components(cfg, device)
ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model"])
model.anchors = ckpt["anchors"]
model.eval()

# Load the test split and sample indices.
dataset = KITTIDataset(KITTI_ROOT, os.path.join(KITTI_ROOT, "kitti_split_6_2_2.json"), "test")
split = np.load(SPLIT_PATH)
subset_indices = {
    "easy": split["easy"][:20],
    "moderate": split["moderate"][:20],
    "hard": split["hard"][:20],
}

# Load the cached fusion function.
from utils.fuse_preprocess_cache_kitti import load_fused_from_cache
id_map = {sid: idx for idx, sid in enumerate(dataset.ids)}

# Process each image.
for diff_level, indices in subset_indices.items():
    for idx in tqdm(indices, desc=f"Processing {diff_level}"):
        sample = dataset[idx]
        fused = load_fused_from_cache(sample["id"], id_map.get).unsqueeze(0).to(device)  # [1, 4, H, W]
        with torch.no_grad():
            feats = model.bb(fused)
            outputs = model.hd([feats["P3"], feats["P4"], feats["P5"]])
        pred_boxes = []
        for lvl, stride in enumerate([8, 16, 32]):
            dec = decode_yolo_output(outputs[lvl], model.anchors[lvl], stride, conf_thresh=0.3, nms_thresh=0.5)
            pred_boxes += dec[0]
        # Apply NMS to merged predictions across feature levels.
        from torchvision.ops import nms
        if len(pred_boxes) > 0:
            boxes_tensor = torch.tensor([b["box"] for b in pred_boxes], dtype=torch.float32)
            scores_tensor = torch.tensor([b["score"] for b in pred_boxes], dtype=torch.float32)

            keep = nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
            pred_boxes = [pred_boxes[i] for i in keep]

        # Get the input dimensions and original image path.
        H_in, W_in = sample["image"].shape[1:]
        raw_img_path = os.path.join(KITTI_ROOT, "image_2", os.path.basename(sample["id"]) + ".png")
        raw_img = cv2.imread(raw_img_path)

        # Handle image loading failures.
        if raw_img is None:
            print(f"[Warning] Failed to load image: {raw_img_path}")
            continue

        H_raw, W_raw = raw_img.shape[:2]

        # Rescale boxes to the original image dimensions.
        boxes = []
        scores = []
        for box in pred_boxes:
            x1, y1, x2, y2 = box["box"]
            score = box["score"]
            x1 *= W_raw / W_in
            x2 *= W_raw / W_in
            y1 *= H_raw / H_in
            y2 *= H_raw / H_in
            boxes.append([x1, y1, x2, y2])
            scores.append(score)

        # Draw predicted boxes.
        draw_boxes_with_scores(
            image=raw_img,
            boxes=boxes,
            scores=scores,
            label_prefix="E",
            color=(255, 128, 0)  # blue
        )

        # Save the annotated image.
        save_name = os.path.basename(sample["id"]) + ".jpg"
        save_path = os.path.join(SAVE_DIRS[diff_level], save_name)
        cv2.imwrite(save_path, raw_img)
