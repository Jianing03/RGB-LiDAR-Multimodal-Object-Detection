import torch


def decode_yolo_output(output, anchors, stride, conf_thresh=0.3, nms_thresh=0.5, top_k=None):
    assert output.dim() == 5, f"Expect 5-D tensor, got {output.shape}"

    """
    Decode batched YOLO predictions with multiple anchors.

    :param output: Tensor [B, A, H, W, 5+C]
    :param anchors: List[Tuple(w, h)] or Tensor [A, 2]
    :param stride: int: output stride, e.g. P3=8, P4=16, P5=32
    :param conf_thresh: Confidence threshold
    :param nms_thresh: NMS IoU threshold
    :return: List[B], each item a List[dict{box, score, label}]
    """
    B, A, H, W, pred_dim = output.shape
    num_classes = pred_dim - 5
    device = output.device
    anchors = torch.tensor(anchors, device=device).view(1, A, 1, 1, 2)

    # Build grid offsets.
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).to(device)  # [H, W, 2]
    grid_xy = grid_xy.view(1, 1, H, W, 2)

    # Decode predictions.
    pred = output.clone()
    pred_xy = (pred[..., 0:2].sigmoid() + grid_xy) * stride  # Box centers.
    pred_wh = torch.exp(pred[..., 2:4]) * anchors  # Box width and height.
    pred_obj = pred[..., 4].sigmoid()  # Objectness probability.
    pred_cls = pred[..., 5:].sigmoid()  # Per-class probabilities.

    pred_boxes = torch.cat([
        pred_xy - pred_wh / 2,  # x1, y1
        pred_xy + pred_wh / 2  # x2, y2
    ], dim=-1)  # [B, A, H, W, 4]

    pred_scores = pred_obj.unsqueeze(-1) * pred_cls  # [B, A, H, W, C]

    all_results = []

    for b in range(B):
        boxes = pred_boxes[b].reshape(-1, 4)  # [A*H*W, 4]
        scores = pred_scores[b].reshape(-1, num_classes)  # [A*H*W, C]

        result_b = []
        for cls in range(num_classes):
            cls_scores = scores[:, cls]
            mask = cls_scores > conf_thresh
            if mask.sum() == 0:
                continue

            selected_boxes = boxes[mask]
            selected_scores = cls_scores[mask]
            # Filter invalid boxes after decoding.
            # Require both width and height to exceed one pixel.
            box_wh = selected_boxes[:, 2:] - selected_boxes[:, :2]
            valid = (box_wh[:, 0] > 1) & (box_wh[:, 1] > 1)
            selected_boxes = selected_boxes[valid]
            selected_scores = selected_scores[valid]

            # NMS
            keep = torch.ops.torchvision.nms(selected_boxes, selected_scores, nms_thresh)
            # ✅ Limit retained boxes per class to top_k.
            if top_k is not None:
                keep = keep[:top_k]

            for idx in keep:
                result_b.append({
                    "box": selected_boxes[idx].tolist(),
                    "score": selected_scores[idx].item(),
                    "label": cls
                })

        # Sort results by descending score.
        result_b = sorted(result_b, key=lambda x: x['score'], reverse=True)
        all_results.append(result_b)

    return all_results  # List[B], containing a List[dict] per sample.
