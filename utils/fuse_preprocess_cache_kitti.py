import os
import torch
import numpy as np
from tqdm import tqdm
from dataloader.dataload_kitti import KITTIDataset
from models.fusion.data_fusion import fuse_data
from sklearn.cluster import KMeans

CACHE_DIR = "D:/KITTI/fused_cache_70_all"
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_fused_split(split_key):
    dataset = KITTIDataset(
        root_dir="D:/KITTI",
        split_json="D:/KITTI/kitti_split_6_2_2.json",
        split_key=split_key
    )
    print(f"开始缓存 [{split_key}] 集，共 {len(dataset)} 个样本")

    for sample in tqdm(dataset):
        if sample is None:
            continue
        sample_id = sample['id']
        save_path = os.path.join(CACHE_DIR, f"{sample_id}.pt")
        if os.path.exists(save_path):
            continue
        try:
            fused = fuse_data(sample['image'], sample['lidar'], sample['calib'])
            torch.save(fused, save_path)
        except Exception as e:
            print(f"⚠️ 缓存失败 {sample_id}: {e}")

    print(f"✅ [{split_key}] 缓存完成！")


def load_fused_from_cache(sample_id, dataset_lookup_fn):
    path = os.path.join(CACHE_DIR, f"{sample_id}.pt")
    if os.path.exists(path):
        return torch.load(path)
    else:
        print(f"⚠️ 缓存缺失: {sample_id}, 实时补充计算")
        sample = dataset_lookup_fn(sample_id)
        return fuse_data(sample['image'], sample['lidar'], sample['calib'])


if __name__ == '__main__':
    for split in ["train", "val", "test"]:
        cache_fused_split(split)
