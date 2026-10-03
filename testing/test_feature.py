import os
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from testing.test_utils.load_yaml import load_yaml
from testing.test_utils.test_utils import split_by_difficulty
from testing.test_utils.test_dataload_kitti import KITTIDataset
from training.feature_train import build_feature_components, _LIST_KEYS
from utils.metrics import evaluate_dataset


def kitti_collate(batch):
    batch = [b for b in batch if b]
    if not batch:
        return None
    out = {}
    for k in batch[0]:
        out[k] = [b[k] for b in batch] if k in _LIST_KEYS else torch.stack([b[k] for b in batch])
    return out


def evaluate_subset(model, dataloader, device):
    model.eval()
    all_preds, all_gts = [], []
    total_time = 0.0
    with torch.no_grad():
        for batch in dataloader:
            if batch is None: continue
            imgs = batch["image"].to(device)
            lidars = batch["lidar"]
            calibs = batch["calib"]

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            feats = model.bb(imgs, lidars, calibs)
            outputs = model.hd([feats["P3"], feats["P4"], feats["P5"]])

            end.record()
            torch.cuda.synchronize()
            total_time += start.elapsed_time(end) / 1000.0

            B = len(batch["id"])
            per_img = [[] for _ in range(B)]
            from utils.yolo_postprocess import decode_yolo_output
            for lvl, stride in enumerate([8, 16, 32]):
                dec = decode_yolo_output(outputs[lvl], model.anchors[lvl], stride, 0.3, 0.5)
                for b in range(B):
                    per_img[b] += dec[b]
            all_preds.extend(per_img)
            all_gts.extend([batch["bboxes"][b].to(device) for b in range(B)])

    metrics = evaluate_dataset(all_preds, all_gts)
    metrics = {k: round(v, 3) for k, v in metrics.items()}
    metrics["fps"] = round(len(dataloader.dataset) / total_time, 3)
    return metrics


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_yaml("D:/Desktop/mod/configs/feature_train_config.yaml")
    model, _, _, _, _, _ = build_feature_components(cfg, device)
    ckpt = torch.load("D:/Desktop/final_results/feature_saved_model_best_f1.pth", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.anchors = ckpt["anchors"]

    dataset = KITTIDataset("D:/KITTI", "D:/KITTI/kitti_split_6_2_2.json", "test")
    npz_path = "D:/Desktop/mod/test_indices.npz"
    if os.path.exists(npz_path):
        splits = np.load(npz_path)
        easy_ids, moderate_ids, hard_ids = splits["easy"], splits["moderate"], splits["hard"]
    else:
        easy_ids, moderate_ids, hard_ids = split_by_difficulty(dataset, save_path=npz_path)

    dl_easy     = DataLoader(Subset(dataset, easy_ids), batch_size=8, collate_fn=kitti_collate)
    dl_moderate = DataLoader(Subset(dataset, moderate_ids), batch_size=8, collate_fn=kitti_collate)
    dl_hard     = DataLoader(Subset(dataset, hard_ids), batch_size=8, collate_fn=kitti_collate)

    print("\U0001f9e0 [Easy] 测试中...")
    print(evaluate_subset(model, dl_easy, device))

    print("\U0001f9e0 [Moderate] 测试中...")
    print(evaluate_subset(model, dl_moderate, device))

    print("\U0001f9e0 [Hard] 测试中...")
    print(evaluate_subset(model, dl_hard, device))

    # ✅ 新增：完整验证集评估
    print("🚀 [Overall 全体样本] 测试中...")
    full_loader = DataLoader(dataset, batch_size=8, collate_fn=kitti_collate)
    overall_metrics = evaluate_subset(model, full_loader, device)
    print(overall_metrics)