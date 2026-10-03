import torch

def decode_yolo_output(output, anchors, stride, conf_thresh=0.3, nms_thresh=0.5, top_k=None):
    assert output.dim() == 5, f"Expect 5-D tensor, got {output.shape}"

    """
    解码 YOLO 输出，支持 batch 和多 anchor。

    :param output: Tensor [B, A, H, W, 5+C]
    :param anchors: List[Tuple(w, h)] 或 Tensor [A, 2]
    :param stride: int（当前输出层的 stride，如 P3=8, P4=16, P5=32）
    :param conf_thresh: 置信度阈值
    :param nms_thresh: NMS 阈值
    :param top_k: 每张图像保留最多多少个框（NMS 后再排序）
    :return: List[B]，每个样本是 List[dict{box, score, label}]
    """
    B, A, H, W, pred_dim = output.shape
    num_classes = pred_dim - 5
    device = output.device
    anchors = torch.tensor(anchors, device=device).view(1, A, 1, 1, 2)

    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).to(device)
    grid_xy = grid_xy.view(1, 1, H, W, 2)

    pred = output.clone()
    pred_xy = (pred[..., 0:2].sigmoid() + grid_xy) * stride
    pred_wh = torch.exp(pred[..., 2:4]) * anchors
    pred_obj = pred[..., 4].sigmoid()
    pred_cls = pred[..., 5:].sigmoid()

    pred_boxes = torch.cat([pred_xy - pred_wh / 2, pred_xy + pred_wh / 2], dim=-1)
    pred_scores = pred_obj.unsqueeze(-1) * pred_cls

    all_results = []

    for b in range(B):
        boxes = pred_boxes[b].reshape(-1, 4)
        scores = pred_scores[b].reshape(-1, num_classes)

        result_b = []
        for cls in range(num_classes):
            cls_scores = scores[:, cls]
            mask = cls_scores > conf_thresh
            if mask.sum() == 0:
                continue

            selected_boxes = boxes[mask]
            selected_scores = cls_scores[mask]
            box_wh = selected_boxes[:, 2:] - selected_boxes[:, :2]
            valid = (box_wh[:, 0] > 1) & (box_wh[:, 1] > 1)
            selected_boxes = selected_boxes[valid]
            selected_scores = selected_scores[valid]

            keep = torch.ops.torchvision.nms(selected_boxes, selected_scores, nms_thresh)
            for idx in keep:
                result_b.append({
                    "box": selected_boxes[idx].tolist(),
                    "score": selected_scores[idx].item(),
                    "label": cls
                })

        # 排序 + top_k 截断
        result_b = sorted(result_b, key=lambda x: x["score"], reverse=True)
        all_results.append(result_b[:top_k])

    return all_results
