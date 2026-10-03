import numpy as np
from sklearn.cluster import KMeans
from dataloader.dataload_kitti import KITTIDataset
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"  # 手动指定线程数，比如 8


def cluster_anchors_from_dataset(dataset, num_anchors=9, input_size=(256, 256), original_size=(1242, 375)):
    whs = []
    id_map = {}

    for sample in dataset:
        if sample is None:
            continue
        id_map[sample["id"]] = sample
        for box in sample['bboxes']:
            w = box[2] - box[0]
            h = box[3] - box[1]
            if w > 15 and h > 15:
                whs.append([min(w.item(), 300), min(h.item(), 300)])

    kmeans = KMeans(n_clusters=num_anchors, n_init=10, random_state=42)
    kmeans.fit(np.array(whs))  # ✅ 注意是两步法

    raw_anchors = [tuple(map(float, c)) for c in kmeans.cluster_centers_]

    # 缩放到输入尺寸
    scale_w = input_size[0] / original_size[0]
    scale_h = input_size[1] / original_size[1]
    scaled_anchors = [(w * scale_w, h * scale_h) for (w, h) in raw_anchors]

    # 按面积从小到大排序后分组
    scaled_anchors.sort(key=lambda x: x[0] * x[1])  # 按面积升序
    anchors_per_level = [scaled_anchors[i * 3:(i + 1) * 3] for i in range(3)]
    # anchors_per_level = [[a1,a2,a3], [a4,a5,a6], [a7,a8,a9]]

    print("🔧 聚类出的 anchors（原图尺寸）:", raw_anchors)
    print("🔧 缩放后的 anchors（输入尺寸）（按面积从小到大排序后分组）:", anchors_per_level)

    return scaled_anchors, id_map

if __name__ == '__main__':
    dataset_train = KITTIDataset("D:/KITTI", "D:/KITTI/kitti_split_6_2_2.json", split_key="train")

    anchors, id_map = cluster_anchors_from_dataset(dataset_train, num_anchors=9)
    # print("✅ 自动聚类 anchors:", anchors)
    # 跑完后打印每层平均 GT 匹配数
    scaled, _ = cluster_anchors_from_dataset(dataset_train, 9)
    P3, P4, P5 = scaled[:3], scaled[3:6], scaled[6:9]

    # quick sanity-check
    print('P3 anchors', P3)
    print('P4 anchors', P4)
    print('P5 anchors', P5)