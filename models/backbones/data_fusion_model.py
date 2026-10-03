import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# from torchvision.models import ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights


class DataFusionResNetFPN(nn.Module):
    """
    Multimodal ResNet-18 backbone with an FPN.
    Accept four-channel input (RGB and depth) and produce multiscale detection features.
    """
    def __init__(self, pretrained=True, fpn_out_channels=256, drop_p=0.3):
        super(DataFusionResNetFPN, self).__init__()
        # Load ResNet-18 with optional ImageNet pretrained weights.
        chosen_weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=chosen_weights)
        # Adapt the first ResNet convolution from three input channels to four.
        orig_conv1 = self.backbone.conv1  # Original Conv2d(3, 64, kernel_size=7, stride=2, padding=3).
        new_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Copy RGB weights to the first three channels and the first channel's weights to the depth channel.
        with torch.no_grad():
            new_conv1.weight[:, :3] = orig_conv1.weight  # Copy the original RGB weights.
            new_conv1.weight[:, 3:4] = orig_conv1.weight[:, :1]  # Initialize the depth channel with the first channel's weights.
        self.backbone.conv1 = new_conv1
        # Keep BatchNorm and ReLU unchanged because the convolution still outputs 64 channels.
        # Disable the classifier; forward bypasses global average pooling and uses convolutional features.
        self.backbone.fc = None  # Disable the unused classifier.
        # Extract features stage by stage instead of calling self.backbone.forward.

        self.lat_conv2 = nn.Conv2d(64, fpn_out_channels, 1)
        self.lat_conv3 = nn.Conv2d(128, fpn_out_channels, 1)
        self.lat_conv4 = nn.Conv2d(256, fpn_out_channels, 1)
        self.lat_conv5 = nn.Conv2d(512, fpn_out_channels, 1)
        # Define 3x3 smoothing convolutions for FPN outputs.
        self.smooth_conv2 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        self.smooth_conv3 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        self.smooth_conv4 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        self.smooth_conv5 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)

        # ✅ Apply functional Dropout2d in forward for regularization.
        # self.dropout = nn.Dropout2d(p=0.3)
        self.drop_p = drop_p

    def forward(self, x):
        """
        Forward pass:
        Input:
            x (Tensor): [N, 4, H, W], four-channel RGB and depth input.
        Output:
            features (dict): multiscale feature maps:
                C2, C3, C4, C5: intermediate ResNet features, used internally.
                P3, P4, P5: returned FPN features, each with fpn_out_channels channels.
        """
        # 1. Compute C2, C3, C4, and C5 with the ResNet-18 backbone.
        # Stem: Conv1 -> BN1 -> ReLU -> MaxPool; downsample by a factor of four.
        x = self.backbone.conv1(x)            # [N, 64, H/2, W/2], Initial 7x7 convolution; downsample by two.
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)          # [N, 64, H/4, W/4], 3x3 max pooling; downsample by another factor of two.

        # Extract residual stage outputs.
        c2 = self.backbone.layer1(x)          # [N, 64, H/4, W/4], first residual stage; preserve spatial resolution.
        c3 = self.backbone.layer2(c2)         # [N, 128, H/8, W/8], second residual stage; downsample by two.
        c4 = self.backbone.layer3(c3)         # [N, 256, H/16, W/16], third residual stage; downsample by two.
        c5 = self.backbone.layer4(c4)         # [N, 512, H/32, W/32], fourth residual stage; downsample by two.

        # 2. Build P3, P4, and P5 with top-down FPN fusion.
        p5 = self.lat_conv5(c5)
        p4 = self.lat_conv4(c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p3 = self.lat_conv3(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")

        # Dropout
        p5 = F.dropout2d(p5, self.drop_p, self.training)
        p4 = F.dropout2d(p4, self.drop_p, self.training)
        p3 = F.dropout2d(p3, self.drop_p, self.training)

        # 3x3 smoothing convolutions.
        p5 = self.smooth_conv5(p5)
        p4 = self.smooth_conv4(p4)
        p3 = self.smooth_conv3(p3)

        return {"P3": p3, "P4": p4, "P5": p5}

