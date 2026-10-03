import torch
from utils.yolo_postprocess import decode_yolo_output
from utils.metrics import evaluate_dataset

# ---------- Unified validation with optional PR curve output ----------
def unified_validate(model, ema, dl_val, device, save_pr_curve=False):
    from torchvision.ops import nms
    from utils.yolo_postprocess import decode_yolo_output
    from utils.metrics import evaluate_dataset

    # Read thresholds from the model, falling back to defaults.
    conf_thres = getattr(model, "conf_thres", 0.3)
    nms_thres  = getattr(model, "nms_thres", 0.5)

    with ema.average_parameters():
        model.eval()
        strides = [8, 16, 32]
        all_preds, all_gts = [], []

        with torch.no_grad():
            for batch in dl_val:
                if batch is None:
                    continue

                outputs = model.forward_batch(batch, device)
                # Detect dual-branch result-level outputs.
                if isinstance(outputs, tuple) and len(outputs) >= 2 \
                   and isinstance(outputs[0], list) and isinstance(outputs[1], list):
                    preds_rgb, preds_lidar = outputs[:2]
                    is_result = True
                else:
                    preds = outputs[0] if isinstance(outputs, tuple) else outputs
                    is_result = False

                B = len(batch["id"])
                per_img = [[] for _ in range(B)]

                if isinstance(outputs, tuple) and len(outputs) == 3 \
                        and all(isinstance(x, list) for x in outputs[:2]):
                    # ---------- Result-level fusion ----------
                    preds_rgb, preds_lidar, _ = outputs
                    # is_result = True
                    # Decode the RGB and LiDAR branches separately.
                    # print(f"[Validate] P{lvl + 3} output shape: {preds[lvl].shape}")

                    for lvl, stride in enumerate(strides):
                        rgb_dec = decode_yolo_output(
                            preds_rgb[lvl], model.anchors[lvl], stride,
                            conf_thres, nms_thres
                        )
                        lid_dec = decode_yolo_output(
                            preds_lidar[lvl], model.anchors[lvl], stride,
                            conf_thres, nms_thres
                        )
                        for b in range(B):
                            per_img[b] += rgb_dec[b] + lid_dec[b]
                    # Apply joint NMS per image.
                    for b in range(B):
                        preds_b = per_img[b]
                        if not preds_b:
                            continue
                        boxes  = torch.tensor(
                            [p["box"] for p in preds_b],
                            dtype=torch.float32, device=device
                        )
                        scores = torch.tensor(
                            [p["score"] for p in preds_b],
                            dtype=torch.float32, device=device
                        )
                        keep = nms(boxes, scores, iou_threshold=nms_thres)
                        per_img[b] = [preds_b[k] for k in keep.cpu()]
                else:
                    # ---------- Early / feature fusion ----------
                    preds = outputs[0] if isinstance(outputs, tuple) else outputs
                    is_result = False

                    for lvl, stride in enumerate(strides):
                        dec = decode_yolo_output(
                            preds[lvl], model.anchors[lvl], stride,
                            conf_thres, nms_thres
                        )
                        for b in range(B):
                            per_img[b] += dec[b]

                # Collect predictions and ground-truth boxes.
                for b in range(B):
                    all_preds.append(per_img[b])
                    all_gts.append(batch["bboxes"][b].to(device))

        result = evaluate_dataset(
            all_preds, all_gts,
            save_pr_curve_path="./results/result_pr_curve.png" if save_pr_curve else None
        )
        return result["precision"], result["recall"], result["f1"], result["ap"]
