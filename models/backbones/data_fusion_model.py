import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# from torchvision.models import ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights


class DataFusionResNetFPN(nn.Module):
    """
    多模态ResNet50 + FPN骨干网络.
    接受4通道输入 (RGB图像 + 深度图), 输出多尺度特征用于目标检测.
    """
    def __init__(self, pretrained=True, fpn_out_channels=256, drop_p=0.3):
        super(DataFusionResNetFPN, self).__init__()
        # 加载预训练的ResNet-50 模型 (ImageNet预训练权重)
        chosen_weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=chosen_weights)
        # 修改ResNet 第一层卷积层，以接受4通道输入 (原为3通道)
        orig_conv1 = self.backbone.conv1  # 原始Conv2d(3,64, kernel_size=7, stride=2, padding=3)
        new_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 初始化新卷积核权重：复制原始权重到前3个通道，第四个通道可以用平均或复制原权重来初始化
        with torch.no_grad():
            new_conv1.weight[:, :3] = orig_conv1.weight  # 拷贝原来3个通道的权重
            new_conv1.weight[:, 3:4] = orig_conv1.weight[:, :1]  # 用第1通道权重初始化第4通道
        self.backbone.conv1 = new_conv1
        # （BatchNorm层和ReLU层不需要修改，它们对通道数不敏感）
        # 去除ResNet最后的全局平均池化和全连接分类层，只需骨干的卷积特征
        self.backbone.fc = None  # 删除全连接分类层 (不使用)
        # 注意：在forward中不会使用self.backbone.forward直接得到结果，而是逐层提取特征

        self.lat_conv2 = nn.Conv2d(64, fpn_out_channels, 1)
        self.lat_conv3 = nn.Conv2d(128, fpn_out_channels, 1)
        self.lat_conv4 = nn.Conv2d(256, fpn_out_channels, 1)
        self.lat_conv5 = nn.Conv2d(512, fpn_out_channels, 1)
        # 定义FPN各层输出的3x3卷积平滑层
        self.smooth_conv2 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        self.smooth_conv3 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        self.smooth_conv4 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)
        self.smooth_conv5 = nn.Conv2d(fpn_out_channels, fpn_out_channels, kernel_size=3, padding=1)

        # ✅ 加入 Dropout2d 以增强正则化
        # self.dropout = nn.Dropout2d(p=0.3)
        self.drop_p = drop_p

    def forward(self, x):
        """
        前向传播:
        输入:
            x (Tensor) - 尺寸 [N, 4, H, W], 4通道图像 (RGB + 深度)
        输出:
            features (dict): 包含多个尺度的特征图:
                C2, C3, C4, C5: ResNet各阶段输出特征 (用于调试或进一步利用)
                P2, P3, P4, P5: FPN融合后的特征图 (通道数均为fpn_out_channels)
        """
        # 1. ResNet-50 主干网络计算各层特征 C2, C3, C4, C5
        # Stem部分：Conv1 -> BN1 -> ReLU -> MaxPool，输出尺寸降为1/4
        x = self.backbone.conv1(x)            # [N, 64, H/2, W/2], 第一层7x7卷积，下采样2倍
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)          # [N, 64, H/4, W/4], 3x3最大池化，再下采样2倍

        # ResNet各残差层输出
        c2 = self.backbone.layer1(x)          # [N, 256, H/4, W/4], 第1层残差块组 (不改变尺寸)
        c3 = self.backbone.layer2(c2)         # [N, 512, H/8, W/8], 第2层残差块组 (下采样2倍)
        c4 = self.backbone.layer3(c3)         # [N, 1024, H/16, W/16], 第3层残差块组 (下采样2倍)
        c5 = self.backbone.layer4(c4)         # [N, 2048, H/32, W/32], 第4层残差块组 (下采样2倍)

        # 2. 自顶向下构建 FPN 金字塔特征 P3, P4, P5
        p5 = self.lat_conv5(c5)
        p4 = self.lat_conv4(c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p3 = self.lat_conv3(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")

        # Dropout
        p5 = F.dropout2d(p5, self.drop_p, self.training)
        p4 = F.dropout2d(p4, self.drop_p, self.training)
        p3 = F.dropout2d(p3, self.drop_p, self.training)

        # 3x3 平滑卷积
        p5 = self.smooth_conv5(p5)
        p4 = self.smooth_conv4(p4)
        p3 = self.smooth_conv3(p3)

        return {"P3": p3, "P4": p4, "P5": p5}

