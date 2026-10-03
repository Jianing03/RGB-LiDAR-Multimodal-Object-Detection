import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------- CIoU 损失 --------------------------- #
def bbox_ciou(pred_boxes, target_boxes, eps: float = 1e-6):
    px, py, pw, ph = pred_boxes.unbind(-1)
    tx, ty, tw, th = target_boxes.unbind(-1)

    inter_w = torch.min(pw, tw)
    inter_h = torch.min(ph, th)
    inter_area = inter_w * inter_h
    union_area = pw * ph + tw * th - inter_area
    iou = inter_area / (union_area + eps)

    center_dist = (px - tx).pow(2) + (py - ty).pow(2)
    cw = torch.max(px + pw / 2, tx + tw / 2) - torch.min(px - pw / 2, tx - tw / 2)
    ch = torch.max(py + ph / 2, ty + th / 2) - torch.min(py - ph / 2, ty - th / 2)
    c2 = cw.pow(2) + ch.pow(2)

    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(tw / (th + eps)) - torch.atan(pw / (ph + eps)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - center_dist / (c2 + eps) - alpha * v
    return 1.0 - ciou

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        prob = logits.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        modulating_factor = (1 - p_t) ** self.gamma
        loss = alpha_factor * modulating_factor * ce_loss
        return loss.mean() if self.reduction == "mean" else loss.sum()

# --------------------------- YOLO Loss 主类 --------------------------- #
import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, strides, lambda_list=[1.0, 0.5, 0.25], iou_pos_thr=0.3, iou_ignore_thr=0.1):
        super().__init__()
        # self.anchors = anchors
        self.anchors = [torch.tensor(a, dtype=torch.float32) for a in anchors]
        self.num_classes = num_classes
        self.strides = strides
        self.lambda_list = lambda_list
        self.iou_pos_thr = iou_pos_thr
        self.iou_ignore_thr = iou_ignore_thr
        self.focal_obj = FocalLoss(2.0, 0.25, "mean")

    def forward(self, preds, targets):
        return self._compute_loss(preds, targets)

    def forwardV2(self, preds_img, preds_lidar, targets):
        loss_img, details_img = self._compute_loss(preds_img, targets)
        loss_lidar, details_lidar = self._compute_loss(preds_lidar, targets)
        total_loss = (loss_img + loss_lidar) / 2
        details = {
            "loss_img": details_img,
            "loss_lidar": details_lidar,
            "total_loss": total_loss.item()
        }
        return total_loss, details

    def _compute_loss(self, preds, targets):
        device = preds[0].device
        total_loss = 0.0
        # ciou_total, obj_total, cls_total = 0.0, 0.0, 0.0
        ciou_total = torch.tensor(0.0, device=device)
        obj_total = torch.tensor(0.0, device=device)
        cls_total = torch.tensor(0.0, device=device)

        for lvl, pred in enumerate(preds):
            stride = self.strides[lvl]
            B, A, H, W, _ = pred.shape
            anchors_scaled = (self.anchors[lvl] / stride).to(device).view(1, A, 1, 1, 2)
            # print(f"anchors[lvl] (before): {self.anchors[lvl]}")
            # print(f"type: {type(self.anchors[lvl])}")

            obj_mask = torch.zeros(B, A, H, W, dtype=torch.bool, device=device)
            ignore_mask = torch.zeros_like(obj_mask)
            tgt_xywh = torch.zeros(B, A, H, W, 4, device=device)
            tgt_obj = torch.zeros(B, A, H, W, device=device)
            tgt_cls = torch.zeros(B, A, H, W, device=device)

            for b in range(B):
                if targets[b].numel() == 0:
                    continue
                t_xy = targets[b][:, :2] / stride
                t_wh = targets[b][:, 2:4] / stride
                t_grid = t_xy.long()
                offset = t_xy - t_grid

                for i in range(t_xy.size(0)):
                    gx, gy = t_grid[i]
                    if gx >= W or gy >= H:
                        continue
                    gt_wh = t_wh[i]
                    for a in range(A):
                        anc_wh = anchors_scaled[0, a, 0, 0]
                        ratio = gt_wh / (anc_wh + 1e-6)
                        iou_wh = torch.min(ratio, 1/ratio).min()
                        if iou_wh >= self.iou_pos_thr:
                            obj_mask[b, a, gy, gx] = True
                            tgt_obj[b, a, gy, gx] = 1.0
                            tgt_xywh[b, a, gy, gx, :2] = offset[i]
                            tgt_xywh[b, a, gy, gx, 2:] = torch.log((gt_wh / anc_wh + 1e-6).clamp(1e-2, 10.0))
                            tgt_cls[b, a, gy, gx] = 1.0
                        elif iou_wh > self.iou_ignore_thr:
                            ignore_mask[b, a, gy, gx] = True

            # fallback 确保每个 GT 至少有正样本
            for bb in range(B):
                if targets[bb].numel() == 0:
                    continue
                t_xy = targets[bb][:, :2] / stride
                t_wh = targets[bb][:, 2:4] / stride
                t_grid = t_xy.long()
                for i in range(t_xy.size(0)):
                    gx, gy = t_grid[i]
                    if gx >= W or gy >= H:
                        continue
                    gt_wh = t_wh[i]
                    if gt_wh.min() < 1e-2:  # 太小的 gt 框直接跳过
                        continue

                    ratio_all = t_wh[i].unsqueeze(0) / (anchors_scaled[0, :, 0, 0] + 1e-6)
                    best_a = torch.min(ratio_all, 1/ratio_all).min(dim=-1)[0].argmax().item()
                    if not obj_mask[bb, best_a, gy, gx]:
                        obj_mask[bb, best_a, gy, gx] = True
                        tgt_obj[bb, best_a, gy, gx] = 1.0
                        tgt_xywh[bb, best_a, gy, gx, :2] = (t_xy[i] - t_grid[i])
                        tgt_xywh[bb, best_a, gy, gx, 2:] = torch.log((t_wh[i] / anchors_scaled[0, best_a, 0, 0] + 1e-6).clamp(1e-2, 10.0))
                        tgt_cls[bb, best_a, gy, gx] = 1.0

            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0).expand(B, A, H, W, 2).float()

            pred_xy = (pred[..., :2].sigmoid() + grid) * stride
            wh_logit = torch.clamp(pred[..., 2:4], min=-5.0, max=4.0)
            pred_wh = wh_logit.exp() * anchors_scaled
            pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)

            ciou_loss = torch.tensor(0.0, device=device)
            if obj_mask.any():
                tgt_center = (grid[obj_mask] + tgt_xywh[..., :2][obj_mask]) * stride
                tgt_wh_pix = (anchors_scaled * tgt_xywh[..., 2:].exp())[obj_mask]
                tgt_xywh_pix = torch.cat([tgt_center, tgt_wh_pix], dim=-1)
                # ciou_loss = bbox_ciou(pred_xywh[obj_mask], tgt_xywh_pix).mean()
                ciou_loss = bbox_ciou(pred_xywh[obj_mask], tgt_xywh_pix).clamp(min=1e-6).mean()

            obj_target = tgt_obj.clone()
            obj_target[ignore_mask] = -1.0
            loss_obj = self.focal_obj(pred[..., 4][obj_target >= 0], obj_target[obj_target >= 0])

            loss_cls = (
                F.binary_cross_entropy_with_logits(pred[..., 5][obj_mask], tgt_cls[obj_mask])
                if obj_mask.any() else torch.tensor(0.0, device=device)
            )

            ciou_total += self.lambda_list[lvl] * ciou_loss
            obj_total += self.lambda_list[lvl] * loss_obj
            cls_total += self.lambda_list[lvl] * loss_cls

            total_loss += self.lambda_list[lvl] * (ciou_loss + loss_obj + loss_cls)

        details = {
            "ciou_loss": ciou_total.item(),
            "obj_loss": obj_total.item(),
            "cls_loss": cls_total.item()
        }
        return total_loss, details


