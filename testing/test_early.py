import os
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from testing.test_utils.load_yaml import load_yaml
from testing.test_utils.test_utils import split_by_difficulty
from testing.test_utils.test_dataload_kitti import KITTIDataset
from training.early_train import build_early_components, _LIST_KEYS
from utils.metrics import evaluate_dataset
import time

def kitti_collate(batch):
    batch = [b for b in batch if b]
    if not batch:
        return None
    out = {}
    for k in batch[0]:
        out[k] = [b[k] for b in batch] if k in _LIST_KEYS else torch.stack([b[k] for b in batch])
    return out

def project_lidar_to_image(lidar: np.ndarray, calib: dict, W: int, H: int) -> np.ndarray:
    N = lidar.shape[0]
    lidar_homo = np.concatenate([lidar, np.ones((N, 1))], axis=1).T
    Tr = calib["Tr_velo_to_cam"]
    R0 = calib["R0_rect"]
    P2 = calib["P2"]
    cam_xyz = R0 @ (Tr @ lidar_homo)
    valid = cam_xyz[2, :] > 0
    cam_xyz = cam_xyz[:, valid]
    uv_depth = P2 @ np.vstack([cam_xyz, np.ones((1, cam_xyz.shape[1]))])
    uv = uv_depth[:2] / uv_depth[2:]
    uv = uv.T
    z = cam_xyz[2]
    depth = np.zeros((H, W), dtype=np.float32)
    for i in range(uv.shape[0]):
        u, v = int(uv[i, 0]), int(uv[i, 1])
        if 0 <= u < W and 0 <= v < H:
            if depth[v, u] == 0 or z[i] < depth[v, u]:
                depth[v, u] = z[i]
    return depth

def fuse_data(image, lidar, calib):
    _, H, W = image.shape
    if "P2" not in calib or "Tr_velo_to_cam" not in calib or "R0_rect" not in calib:
        zero_depth = torch.zeros((1, H, W), dtype=image.dtype, device=image.device)
        return torch.cat([image, zero_depth], dim=0)
    depth_map = project_lidar_to_image(lidar, calib, W=W, H=H)
    if depth_map is None or depth_map.shape != (H, W):
        depth_map = np.zeros((H, W), dtype=np.float32)
    depth_tensor = torch.from_numpy(depth_map).to(image.device).unsqueeze(0)
    depth_tensor = torch.clamp(depth_tensor / 70.0, 0.0, 1.0)
    return torch.cat([image, depth_tensor], dim=0)

def format_metrics(metrics):
    return {k: round(float(v), 3) if isinstance(v, float) else v for k, v in metrics.items()}

def evaluate_subset(model, dataloader, device):
    model.eval()
    all_preds, all_gts = [], []
    total_time = 0.0
    with torch.no_grad():
        for batch in dataloader:
            if batch is None: continue
            start = time.time()
            fused_batch = [fuse_data(img.to(device), lidar, calib) for img, lidar, calib in zip(batch["image"], batch["lidar"], batch["calib"])]
            fused_batch = torch.stack(fused_batch).to(device)
            # start = time.time()
            feats = model.bb(fused_batch)
            outputs = model.hd([feats["P3"], feats["P4"], feats["P5"]])
            torch.cuda.synchronize()
            total_time += time.time() - start
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
    metrics["fps"] = len(dataloader.dataset) / total_time
    return format_metrics(metrics)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_yaml("./configs/early_train_config.yaml")
    model, _, _, _, _, _ = build_early_components(cfg, device)
    ckpt = torch.load("./results/early_saved_model_best_f1.pth", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.anchors = ckpt["anchors"]

    dataset = KITTIDataset("./data/KITTI", "./data/KITTI/kitti_split_6_2_2.json", "test")
    npz_path = "./test_indices.npz"
    if os.path.exists(npz_path):
        splits = np.load(npz_path)
        easy_ids, moderate_ids, hard_ids = splits["easy"], splits["moderate"], splits["hard"]
    else:
        easy_ids, moderate_ids, hard_ids = split_by_difficulty(dataset, save_path=npz_path)
        # ✅ Print sample counts by difficulty.
    print(f"[Easy] Sample count: {len(easy_ids)}")
    print(f"[Moderate] Sample count: {len(moderate_ids)}")
    print(f"[Hard] Sample count: {len(hard_ids)}")

    dl_easy = DataLoader(Subset(dataset, easy_ids), batch_size=8, collate_fn=kitti_collate)
    dl_moderate = DataLoader(Subset(dataset, moderate_ids), batch_size=8, collate_fn=kitti_collate)
    dl_hard = DataLoader(Subset(dataset, hard_ids), batch_size=8, collate_fn=kitti_collate)

    print("\U0001f697 [Easy] Evaluating...")
    print(evaluate_subset(model, dl_easy, device))
    print("\U0001f697 [Moderate] Evaluating...")
    print(evaluate_subset(model, dl_moderate, device))
    print("\U0001f697 [Hard] Evaluating...")
    print(evaluate_subset(model, dl_hard, device))

    print("🚀 [Overall, all samples] Evaluating...")
    full_loader = DataLoader(dataset, batch_size=8, collate_fn=kitti_collate)
    overall_metrics = evaluate_subset(model, full_loader, device)
    print(overall_metrics)