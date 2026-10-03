import torch
import torch.nn as nn


class YOLODetectionHead(nn.Module):
    def __init__(self, in_channels_list=[256, 256, 256], num_anchors=3, num_classes=1):
        """
        支持多尺度输入的 YOLO 风格检测头。
        每个尺度都有自己的卷积模块，输出 [B, A, H, W, 5+C]

        :param in_channels_list: List[int]，FPN中每个尺度的输入通道数（通常为[256, 256, 256]）
        :param num_anchors: 每个位置的 anchor 数
        :param num_classes: 类别数（只检测 car 时为 1）
        """
        super(YOLODetectionHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.pred_dim = 5 + num_classes  # [x, y, w, h, obj_conf] + 类别

        self.detect_layers = nn.ModuleList()

        for in_channels in in_channels_list:
            self.detect_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_channels),
                    nn.LeakyReLU(0.1),
                    nn.Dropout2d(p=0.3),  # ✅ 添加 Dropout，防止过拟合
                    nn.Conv2d(in_channels, num_anchors * self.pred_dim, kernel_size=1)
                )
            )

    def forward(self, features):
        """
        多尺度特征图的前向传播
        :param features: List[Tensor]，每个Tensor形状为 [B, C, H, W]
        :return: List[Tensor]，每层输出形状为 [B, A, H, W, 5+C]
        """
        outputs = []

        for x, detect_layer in zip(features, self.detect_layers):
            B, _, H, W = x.shape
            out = detect_layer(x)  # [B, A*(5+C), H, W]
            out = out.view(B, self.num_anchors, self.pred_dim, H, W)
            out = out.permute(0, 1, 3, 4, 2).contiguous()  # [B, A, H, W, 5+C]
            outputs.append(out)
            # print(f"[YOLOHead] P5 before reshape: {p5.shape}")

        return outputs  # List of feature maps per level
