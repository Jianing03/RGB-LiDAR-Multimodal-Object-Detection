# result_train.py —— 结果融合训练脚本（统一验证逻辑）

import os, torch, yaml

from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from prefetch_generator import BackgroundGenerator
from training.trainer_core import Trainer
from utils.unified_validation import unified_validate

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

LIST_KEYS = {"lidar", "bboxes", "labels", "calib", "id"}

def kitti_collate(batch):
    batch = [b for b in batch if b]
    if not batch:
        return None
    out = {}
    for k in batch[0]:
        out[k] = [b[k] for b in batch] if k in LIST_KEYS else torch.stack([b[k] for b in batch])
    return out

def compute_iou_matrix(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)

def build_result_components(cfg: dict, device):
    from dataloader.dataload_kitti import KITTIDataset
    from models.backbones.result_fusion_model import ResultFusionModel
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

    backbone   = ResultFusionModel(256).to(device)
    head_rgb   = YOLODetectionHead([256, 256, 256], 3, 1).to(device)
    head_lidar = YOLODetectionHead([256, 256, 256], 3, 1).to(device)

    class ResultModel(torch.nn.Module):
        def __init__(self, bb, h_rgb, h_lidar, anchors):
            super().__init__()
            self.bb = bb
            self.h_rgb = h_rgb
            self.h_lidar = h_lidar
            self.anchors = anchors
        def forward_batch(self, batch, dev):
            img   = batch["image"].to(dev)
            lidar = batch["lidar"]
            calib = batch["calib"]
            # 调用 image_forward / lidar_forward
            feats_rgb = self.bb.image_forward(img)
            feats_lidar = self.bb.lidar_forward(lidar, calib)

            p_rgb = self.h_rgb([feats_rgb["P3_img"], feats_rgb["P4_img"], feats_rgb["P5_img"]])
            p_lidar = self.h_lidar([feats_lidar["P3_lidar"], feats_lidar["P4_lidar"], feats_lidar["P5_lidar"]])

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
                t[:, :2], t[:, 2:4], t[:, 4] = cxcy, wh[m], lbs.float()
                targets.append(t)
            return p_rgb, p_lidar, targets

    model = ResultModel(backbone, head_rgb, head_lidar, anchors)

    class DualLoss(nn.Module):
        def __init__(self, anchors):
            super().__init__()
            self.base = YOLOLoss(anchors, 1, [8, 16, 32])

        def forward(self, p_rgb, p_lidar, targets):
            loss_rgb, det_rgb = self.base(p_rgb, targets)
            loss_lid, det_lid = self.base(p_lidar, targets)

            # ① 取两路损失之和（或平均）
            total_loss = loss_rgb + loss_lid

            # ② 把三个分项做同样的求和/平均，直接放顶层
            details = {
                "ciou_loss": det_rgb["ciou_loss"] + det_lid["ciou_loss"],
                "obj_loss": det_rgb["obj_loss"] + det_lid["obj_loss"],
                "cls_loss": det_rgb["cls_loss"] + det_lid["cls_loss"],
                # ③ 如需保留分路信息，可继续嵌套，但别影响顶层字段
                "rgb": det_rgb,
                "lidar": det_lid
            }
            return total_loss, details

    # criterion = YOLOLoss(anchors, num_classes=1, strides=[8, 16, 32]).to(device)
    criterion = DualLoss(anchors).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    misc = {"anchors": anchors}
    return model, criterion, optimizer, dl_train, dl_val, misc

if __name__ == "__main__":
    Trainer("D:/Desktop/mod/configs/result_train_config.yaml", build_result_components).fit(unified_validate)