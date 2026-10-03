import torch
import torch.nn as nn


class YOLODetectionHead(nn.Module):
    def __init__(self, in_channels_list=[256, 256, 256], num_anchors=3, num_classes=1):
        """
        YOLO-style detection head for multiscale inputs.
        Each scale has its own convolutional block and outputs [B, A, H, W, 5+C].

        :param in_channels_list: List[int]: FPN input channels per scale, typically [256, 256, 256]
        :param num_anchors: Number of anchors per spatial location
        :param num_classes: Number of classes (1 for Car-only detection)
        """
        super(YOLODetectionHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.pred_dim = 5 + num_classes  # [x, y, w, h, obj_conf] + Class scores

        self.detect_layers = nn.ModuleList()

        for in_channels in in_channels_list:
            self.detect_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_channels),
                    nn.LeakyReLU(0.1),
                    nn.Dropout2d(p=0.3),  # ✅ Apply dropout to reduce overfitting.
                    nn.Conv2d(in_channels, num_anchors * self.pred_dim, kernel_size=1)
                )
            )

    def forward(self, features):
        """
        Forward pass for multiscale feature maps.
        :param features: List[Tensor], each shaped [B, C, H, W]
        :return: List[Tensor], each level shaped [B, A, H, W, 5+C]
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
