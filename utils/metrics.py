import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from torchvision.ops import nms

# ---------- IoU ----------
def compute_iou_matrix(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)

# ---------- Single-image evaluation ----------
def evaluate_image(pred_boxes, gt_boxes, pred_scores):
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes), [], []

    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0, [0] * len(pred_boxes), pred_scores.tolist()

    device = pred_boxes.device
    gt_boxes = gt_boxes.to(device)
    ious = compute_iou_matrix(pred_boxes, gt_boxes)
    matched = torch.zeros(len(gt_boxes), dtype=torch.bool, device=device)

    tp, y_true, y_score = 0, [], []
    for i in range(len(pred_boxes)):
        iou, idx = ious[i].max(0)
        score = pred_scores[i].item()
        if iou > 0.5 and not matched[idx]:
            tp += 1
            matched[idx] = True
            y_true.append(1)
        else:
            y_true.append(0)
        y_score.append(score)

    fp = len(pred_boxes) - tp
    fn = (~matched).sum().item()
    return tp, fp, fn, y_true, y_score

# ---------- Compute precision, recall, and F1 ----------
def compute_precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1

# ---------- Plotting ----------
def plot_pr_curve(gt_labels, pred_scores, save_path):
    precision, recall, _ = precision_recall_curve(gt_labels, pred_scores)
    ap = average_precision_score(gt_labels, pred_scores)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'AP={ap:.4f}')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR Curve')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(save_path); plt.close()
    print(f"✅ PR curve saved to: {save_path}")

# ---------- Dataset evaluation ----------
def evaluate_dataset(all_preds, all_gts, save_pr_curve_path=None):
    total_tp = total_fp = total_fn = 0
    y_true, y_scores = [], []

    for preds, gts in zip(all_preds, all_gts):
        if preds:
            boxes = torch.tensor([p["box"] for p in preds])
            scores = torch.tensor([p["score"] for p in preds])
            keep = nms(boxes, scores, 0.5)
            boxes = boxes[keep]
            scores = scores[keep]
        else:
            boxes = torch.zeros((0, 4))
            scores = torch.zeros((0,))

        if isinstance(gts, torch.Tensor):
            gt_boxes = gts
        elif isinstance(gts, list) and len(gts) > 0:
            gt_boxes = torch.stack(gts)
        else:
            gt_boxes = torch.zeros((0, 4))

        tp, fp, fn, local_true, local_scores = evaluate_image(boxes, gt_boxes, scores)
        total_tp += tp; total_fp += fp; total_fn += fn
        y_true.extend(local_true)
        y_scores.extend(local_scores)

    precision, recall, f1 = compute_precision_recall_f1(total_tp, total_fp, total_fn)
    ap = average_precision_score(y_true, y_scores) if y_true else 0.0

    if save_pr_curve_path and y_true:
        plot_pr_curve(y_true, y_scores, save_pr_curve_path)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap": ap,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn
    }

# ---------- Trainer evaluation entry point ----------
def eval_metrics_and_pr(model, ema, dl_val, device, save_path):
    from utils.yolo_postprocess import decode_yolo_output

    all_preds, all_gts = [], []
    stride = [8, 16, 32]

    with ema.average_parameters():
        model.eval()
        with torch.no_grad():
            for batch in dl_val:
                if batch is None: continue
                preds, _ = model.forward_batch(batch, device)
                B = len(batch["id"])
                per_img = [[] for _ in range(B)]
                for lvl in range(3):
                    dec = decode_yolo_output(preds[lvl], model.anchors[lvl], stride[lvl], 0.3, 0.5)
                    for b in range(B):
                        per_img[b] += dec[b]
                for b in range(B):
                    all_preds.append(per_img[b])
                    all_gts.append(batch["bboxes"][b].to(device))

    result = evaluate_dataset(all_preds, all_gts, save_path)
    for lvl in range(3):
        print(f"Level {lvl} pred shape:", preds[lvl].shape)

    return result["ap"], result["f1"]
