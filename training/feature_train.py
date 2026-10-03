# feature_train.py —— 特征融合训练脚本（统一验证逻辑）

import os, torch, yaml
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from prefetch_generator import BackgroundGenerator
from training.trainer_core import Trainer
from utils.unified_validation import unified_validate

# ---------- DataLoader with background prefetch ----------
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# ---------- Collate Fn ----------
_LIST_KEYS = {"lidar", "bboxes", "labels", "calib", "id"}

def kitti_collate(batch):
    batch = [b for b in batch if b]
    if not batch:
        return None
    out = {}
    for k in batch[0]:
        out[k] = [b[k] for b in batch] if k in _LIST_KEYS else torch.stack([b[k] for b in batch])
    return out

# ---------- IoU util ----------
def compute_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt    = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb    = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh    = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)

# ---------- 组件构建 ----------
def build_feature_components(cfg: dict, device):
    from dataloader.dataload_kitti import KITTIDataset
    from models.backbones.feature_fusion_model import FeatureFusionResNetFPN
    from models.detection_head.yolo_detection_head import YOLODetectionHead
    from loss.yolo_loss_1 import YOLOLoss
    from utils.cluster_anchors import cluster_anchors_from_dataset

    ds_train = KITTIDataset("D:/KITTI", "D:/KITTI/kitti_split_6_2_2.json", "train")
    ds_val   = KITTIDataset("D:/KITTI", "D:/KITTI/kitti_split_6_2_2.json", "val")

    dl_train = DataLoaderX(ds_train, batch_size=cfg["batch_size"], shuffle=True,
                           num_workers=cfg.get("num_workers", 0), prefetch_factor=cfg.get("prefetch_factor", 2),
                           pin_memory=True, collate_fn=kitti_collate)
    dl_val   = DataLoaderX(ds_val, batch_size=24, shuffle=False,
                           num_workers=cfg.get("num_workers", 0), prefetch_factor=cfg.get("prefetch_factor", 2),
                           pin_memory=True, collate_fn=kitti_collate)

    if "anchors" in cfg:
        P3, P4, P5 = (cfg["anchors"][k] for k in ("P3", "P4", "P5"))
    else:
        raw, _ = cluster_anchors_from_dataset(ds_train, 9)
        P3, P4, P5 = raw[:3], raw[3:6], raw[6:9]
    anchors = [P3, P4, P5]
    print("✅ anchors:", anchors)

    # 适配新骨干网络（Pillar + P5 融合）
    backbone = FeatureFusionResNetFPN(fpn_out_channels=256, voxel_size=0.4, drop_p=0.3).to(device)
    head     = YOLODetectionHead([256, 256, 256], 3, 1).to(device)

    class FeatureModel(torch.nn.Module):
        def __init__(self, bb, hd):
            super().__init__(); self.bb, self.hd = bb, hd
        def forward_batch(self, batch, dev):
            imgs   = batch["image"].to(dev)
            lidars = batch["lidar"]
            calibs = batch["calib"]  # ★ 加这一行
            feats = self.bb(imgs, lidars, calibs)  # ★ 传入 calibs
            preds  = self.hd([feats["P3"], feats["P4"], feats["P5"]])
            targets = []
            for bxs, lbs in zip(batch["bboxes"], batch["labels"]):
                bxs, lbs = bxs.to(dev), lbs.to(dev)
                m  = ((bxs[:, 2] - bxs[:, 0]) > 1e-2) & ((bxs[:, 3] - bxs[:, 1]) > 1e-2)
                bxs, lbs = bxs[m], lbs[m]
                if not bxs.numel():
                    targets.append(torch.zeros(0, 5, device=dev)); continue
                wh   = bxs[:, 2:] - bxs[:, :2]
                cxcy = (bxs[:, :2] + bxs[:, 2:]) / 2
                t    = torch.zeros((len(bxs), 5), device=dev)
                t[:, :2], t[:, 2:4], t[:, 4] = cxcy, wh, lbs.float()
                targets.append(t)
            return preds, targets

    model = FeatureModel(backbone, head)
    model.anchors = anchors
    criterion = YOLOLoss(anchors, 1, [8, 16, 32]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    misc = {"anchors": anchors}
    return model, criterion, optimizer, dl_train, dl_val, misc


if __name__ == "__main__":
    Trainer("D:/Desktop/mod/configs/feature_train_config.yaml", build_feature_components).fit(unified_validate)
