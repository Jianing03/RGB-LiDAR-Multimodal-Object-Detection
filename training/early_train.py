# early_train.py -- Early-fusion training using Trainer and unified validation.

import torch
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from training.trainer_core import Trainer
from utils.unified_validation import unified_validate  # ✅ Unified validation function.

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
def compute_iou_matrix(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)

# ----------------------------------------------------------------------
# Build training components.
# ----------------------------------------------------------------------
def build_early_components(cfg, device):
    from dataloader.dataload_kitti import KITTIDataset
    from models.backbones.data_fusion_model import DataFusionResNetFPN
    from models.detection_head.yolo_detection_head import YOLODetectionHead
    from loss.yolo_loss_1 import YOLOLoss
    from utils.fuse_preprocess_cache_kitti import load_fused_from_cache
    from utils.cluster_anchors import cluster_anchors_from_dataset

    ds_train = KITTIDataset("./data/KITTI", "./data/KITTI/kitti_split_6_2_2.json", "train")
    ds_val   = KITTIDataset("./data/KITTI", "./data/KITTI/kitti_split_6_2_2.json", "val")
    id_map = {sid: idx for idx, sid in enumerate(ds_train.ids + ds_val.ids)}

    if "anchors" in cfg:
        P3, P4, P5 = (cfg["anchors"][k] for k in ("P3", "P4", "P5"))
    else:
        raw, _ = cluster_anchors_from_dataset(ds_train, 9)
        P3, P4, P5 = raw[:3], raw[3:6], raw[6:9]
    anchors = [P3, P4, P5]
    print("✅ anchors:", anchors)

    dl_train = DataLoaderX(ds_train,
        batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg.get("num_workers", 2),  # 0
        prefetch_factor=cfg.get("prefetch_factor", 21),
        pin_memory=True, collate_fn=kitti_collate)
    dl_val = DataLoaderX(ds_val,
        batch_size=24, shuffle=False,
        num_workers=cfg.get("num_workers", 2),  # 0
        prefetch_factor=cfg.get("prefetch_factor", 2),
        pin_memory=True, collate_fn=kitti_collate)

    backbone = DataFusionResNetFPN(pretrained=True, fpn_out_channels=256).to(device)
    head = YOLODetectionHead([256]*3, 3, 1).to(device)

    class EarlyModel(torch.nn.Module):
        def __init__(self, bb, hd):
            super().__init__()
            self.bb, self.hd = bb, hd
            # self.backbone = bb  # ✅ Optional module aliases.
            # self.head = hd  # ✅ Optional module aliases.
        def forward_batch(self, batch, dev):
            x = torch.stack([
                load_fused_from_cache(s, id_map.get).to(dev)
                for s in batch["id"]
            ])
            feats = self.bb(x)
            preds = self.hd([feats["P3"], feats["P4"], feats["P5"]])
            targets = []
            for bxs, lbs in zip(batch["bboxes"], batch["labels"]):
                bxs, lbs = bxs.to(dev), lbs.to(dev)
                wh = bxs[:, 2:] - bxs[:, :2]
                m  = (wh[:, 0] > 1e-2) & (wh[:, 1] > 1e-2)
                bxs, lbs = bxs[m], lbs[m]
                if not bxs.numel():
                    targets.append(torch.zeros(0, 5, device=dev)); continue
                cxcy = (bxs[:, :2] + bxs[:, 2:]) / 2
                t = torch.zeros((len(bxs), 5), device=dev)
                t[:, :2], t[:, 2:4], t[:, 4] = cxcy, wh, lbs.float()
                targets.append(t)
            return preds, targets

    model = EarlyModel(backbone, head)
    model.anchors = anchors
    criterion = YOLOLoss(anchors, 1, [8, 16, 32]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    misc = {"anchors": anchors}

    return model, criterion, optimizer, dl_train, dl_val, misc

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    Trainer("./configs/early_train_config.yaml", build_early_components).fit(unified_validate)
