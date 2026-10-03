import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PillarFeatureNet(nn.Module):
    """
    增强版 Pillar Feature Net：
    - 输入: 点云 (N, 3)，包含 (x, y, z)
    - 加入位置编码: Δx, Δy, Δz, 距离 √(x²+y²)
    - 输出: BEV 特征图 [B, C, H_bev, W_bev]，使用 max pooling 聚合
    """
    def __init__(self, x_range=(-40, 40), y_range=(0, 70), voxel_size=0.4, out_channels=32, drop_p=0.3):
        super().__init__()
        self.x_range = x_range
        self.y_range = y_range
        self.voxel_size = voxel_size
        self.drop_p = drop_p

        self.bev_W = int((x_range[1] - x_range[0]) / voxel_size)
        self.bev_H = int((y_range[1] - y_range[0]) / voxel_size)

        # 增强版 MLP，输入 7 维特征 (x, y, z, Δx, Δy, Δz, 距离)
        self.fc = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Dropout(p=self.drop_p),  # 加在这
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=self.drop_p),  # 加在这
            nn.Linear(128, out_channels)
        )

    def forward(self, lidars, calib_list):
        batch_bev = []
        max_points = 10000

        for points, calib in zip(lidars, calib_list):
            if isinstance(points, np.ndarray):
                points = torch.from_numpy(points).float().to(next(self.parameters()).device)

            Tr = torch.tensor(calib['Tr_velo_to_cam'], dtype=torch.float32, device=points.device)
            R0 = torch.tensor(calib['R0_rect'], dtype=torch.float32, device=points.device)
            points_homo = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=1).T
            cam_xyz = R0 @ (Tr @ points_homo)

            cam_x, cam_z = cam_xyz[0, :], cam_xyz[2, :]
            points = torch.stack([cam_x, cam_z, cam_xyz[1, :]], dim=1)

            mask_x = (points[:, 0] >= self.x_range[0]) & (points[:, 0] <= self.x_range[1])
            mask_y = (points[:, 1] >= self.y_range[0]) & (points[:, 1] <= self.y_range[1])
            mask = mask_x & mask_y
            points = points[mask]

            if points.shape[0] == 0:
                bev = torch.zeros((self.fc[-1].out_features, self.bev_H, self.bev_W), device=points.device)
                batch_bev.append(bev)
                continue

            if points.shape[0] > max_points:
                idx = torch.randperm(points.shape[0])[:max_points]
                points = points[idx]

            # 增加位置编码
            pillar_center_x = ((points[:, 0] // self.voxel_size) * self.voxel_size + self.x_range[0] + self.voxel_size / 2)
            pillar_center_y = ((points[:, 1] // self.voxel_size) * self.voxel_size + self.y_range[0] + self.voxel_size / 2)
            delta = torch.stack([points[:, 0] - pillar_center_x, points[:, 1] - pillar_center_y, points[:, 2]], dim=1)
            distance = torch.norm(points[:, :2], dim=1, keepdim=True)
            feat_input = torch.cat([points, delta, distance], dim=1)  # (x, y, z, Δx, Δy, Δz, √(x²+y²))

            # 提取点特征
            feat = self.fc(feat_input).float()

            # BEV 索引
            x_idx = ((points[:, 0] - self.x_range[0]) / self.voxel_size).long()
            y_idx = ((points[:, 1] - self.y_range[0]) / self.voxel_size).long()
            x_idx = x_idx.clamp(0, self.bev_W - 1)
            y_idx = y_idx.clamp(0, self.bev_H - 1)

            flat_idx = y_idx * self.bev_W + x_idx  # [N]

            bev_feat_flat = torch.zeros((feat.shape[1], self.bev_H * self.bev_W), device=points.device)
            bev_feat_flat.scatter_reduce_(
                dim=1,
                index=flat_idx.unsqueeze(0).expand(feat.shape[1], -1),
                src=feat.T,
                reduce='amax',
                include_self=False
            )

            bev_feat = bev_feat_flat.view(feat.shape[1], self.bev_H, self.bev_W)
            batch_bev.append(bev_feat)

        return torch.stack(batch_bev, dim=0)

class FeatureFusionResNetFPN(nn.Module):
    def __init__(self, fpn_out_channels=256, voxel_size=0.4, drop_p=0.3):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        self.img_backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.img_backbone.fc = None

        self.lidar_net = PillarFeatureNet(voxel_size=voxel_size, drop_p=drop_p)
        self.lidar_conv = nn.Sequential(
            nn.Conv2d(32, fpn_out_channels, 3, padding=1),
            nn.BatchNorm2d(fpn_out_channels),
            nn.ReLU()
        )
        self.lidar_down4 = nn.Conv2d(fpn_out_channels, fpn_out_channels, 3, stride=2, padding=1)
        self.lidar_down5 = nn.Conv2d(fpn_out_channels, fpn_out_channels, 3, stride=2, padding=1)

        self.lat_conv3 = nn.Conv2d(128 + fpn_out_channels, fpn_out_channels, 1)
        self.lat_conv4 = nn.Conv2d(256 + fpn_out_channels, fpn_out_channels, 1)
        self.lat_conv5 = nn.Conv2d(512 + fpn_out_channels, fpn_out_channels, 1)

        self.smooth3 = nn.Conv2d(fpn_out_channels, fpn_out_channels, 3, padding=1)
        self.smooth4 = nn.Conv2d(fpn_out_channels, fpn_out_channels, 3, padding=1)
        self.smooth5 = nn.Conv2d(fpn_out_channels, fpn_out_channels, 3, padding=1)

        self.drop_p = drop_p

    def forward(self, image, lidar_list, calib_list):
        x = self.img_backbone.conv1(image)
        x = self.img_backbone.bn1(x)
        x = self.img_backbone.relu(x)
        x = self.img_backbone.maxpool(x)
        c2_img = self.img_backbone.layer1(x)
        c3_img = self.img_backbone.layer2(c2_img)
        c4_img = self.img_backbone.layer3(c3_img)
        c5_img = self.img_backbone.layer4(c4_img)

        bev_feat = self.lidar_net(lidar_list, calib_list)
        bev_feat = F.dropout2d(bev_feat, self.drop_p, self.training)

        p3_lidar = self.lidar_conv(bev_feat)
        p4_lidar = self.lidar_down4(p3_lidar)
        p5_lidar = self.lidar_down5(p4_lidar)

        p5_lidar_resized = F.adaptive_max_pool2d(p5_lidar, c5_img.shape[2:])
        c5 = torch.cat([c5_img, p5_lidar_resized], dim=1)
        p5 = self.lat_conv5(c5)
        p5 = F.dropout2d(p5, self.drop_p, self.training)

        p4_lidar_resized = F.adaptive_max_pool2d(p4_lidar, c4_img.shape[2:])
        c4 = torch.cat([c4_img, p4_lidar_resized], dim=1)
        p4 = self.lat_conv4(c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p4 = F.dropout2d(p4, self.drop_p, self.training)

        p3_lidar_resized = F.adaptive_max_pool2d(p3_lidar, c3_img.shape[2:])
        c3 = torch.cat([c3_img, p3_lidar_resized], dim=1)
        p3 = self.lat_conv3(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        p3 = F.dropout2d(p3, self.drop_p, self.training)

        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)

        return {"P3": p3, "P4": p4, "P5": p5}
