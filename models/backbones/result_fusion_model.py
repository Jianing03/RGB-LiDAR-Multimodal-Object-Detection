import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class PillarFeatureNet(nn.Module):
    """
    Pillar feature encoder:
    - Input: a batch of LiDAR point clouds, each shaped (N, 3).
    - Features: reordered camera coordinates, two pillar offsets, camera height, and planar distance.
    - Output: BEV features [B, C, H_bev, W_bev], aggregated by max pooling.
    """
    def __init__(self, x_range=(-40, 40), y_range=(0, 70), voxel_size=0.4, out_channels=32, drop_p=0.3):
        super().__init__()
        self.x_range = x_range
        self.y_range = y_range
        self.voxel_size = voxel_size
        self.drop_p = drop_p

        self.bev_W = int((x_range[1] - x_range[0]) / voxel_size)
        self.bev_H = int((y_range[1] - y_range[0]) / voxel_size)

        # MLP with seven input features: three coordinates, two offsets, height, and planar distance.
        self.fc = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Dropout(p=self.drop_p),  # Regularize point features.
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=self.drop_p),  # Regularize point features.
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

            # Add pillar offsets, camera height, and planar distance.
            pillar_center_x = ((points[:, 0] // self.voxel_size) * self.voxel_size + self.x_range[0] + self.voxel_size / 2)
            pillar_center_y = ((points[:, 1] // self.voxel_size) * self.voxel_size + self.y_range[0] + self.voxel_size / 2)
            delta = torch.stack([points[:, 0] - pillar_center_x, points[:, 1] - pillar_center_y, points[:, 2]], dim=1)
            distance = torch.norm(points[:, :2], dim=1, keepdim=True)
            feat_input = torch.cat([points, delta, distance], dim=1)  # (cam_x, cam_z, cam_y, offset_x, offset_z, cam_y, planar_distance)

            # Extract point features.
            feat = self.fc(feat_input).float()

            # Compute BEV cell indices.
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

class ResultFusionModel(nn.Module):
    def __init__(self, fpn_out_channels=256, drop_p=0.3):
        super().__init__()
        self.drop_p = drop_p
        # Image branch.
        self.img_backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.img_backbone.fc = None
        self.lat_conv3_img = nn.Conv2d(128, fpn_out_channels, 1)
        self.lat_conv4_img = nn.Conv2d(256, fpn_out_channels, 1)
        self.lat_conv5_img = nn.Conv2d(512, fpn_out_channels, 1)
        # LiDAR branch.
        self.lidar_net = PillarFeatureNet()
        self.lidar_conv = nn.Conv2d(32, fpn_out_channels, 3, padding=1)
        self.lidar_down4 = nn.Conv2d(fpn_out_channels, fpn_out_channels, 3, stride=2, padding=1)  # P4
        self.lidar_down5 = nn.Conv2d(fpn_out_channels, fpn_out_channels, 3, stride=2, padding=1)  # P5

    def image_forward(self, image):
        # Image branch.
        x = self.img_backbone.conv1(image)
        x = self.img_backbone.bn1(x)
        x = self.img_backbone.relu(x)
        x = self.img_backbone.maxpool(x)
        c2 = self.img_backbone.layer1(x)
        c3 = self.img_backbone.layer2(c2)
        c4 = self.img_backbone.layer3(c3)
        c5 = self.img_backbone.layer4(c4)

        p5_img = self.lat_conv5_img(c5)
        p5_img = F.dropout2d(p5_img, self.drop_p, self.training)

        p4_img = self.lat_conv4_img(c4) + F.interpolate(p5_img, scale_factor=2, mode="nearest")
        p4_img = F.dropout2d(p4_img, self.drop_p, self.training)

        p3_img = self.lat_conv3_img(c3) + F.interpolate(p4_img, scale_factor=2, mode="nearest")
        p3_img = F.dropout2d(p3_img, self.drop_p, self.training)

        return {"P3_img": p3_img, "P4_img": p4_img, "P5_img": p5_img}

    def lidar_forward(self, lidar_list, calib_list):
        bev_feat = self.lidar_net(lidar_list, calib_list)
        bev_feat = F.dropout2d(bev_feat, self.drop_p, self.training)
        bev_feat = self.lidar_conv(bev_feat)

        p3_lidar = bev_feat
        p4_lidar = self.lidar_down4(p3_lidar)
        p5_lidar = self.lidar_down5(p4_lidar)

        p3_lidar = F.dropout2d(p3_lidar, self.drop_p, self.training)
        p4_lidar = F.dropout2d(p4_lidar, self.drop_p, self.training)
        p5_lidar = F.dropout2d(p5_lidar, self.drop_p, self.training)

        return {"P3_lidar": p3_lidar, "P4_lidar": p4_lidar, "P5_lidar": p5_lidar}
