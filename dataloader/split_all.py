import os
import json
import random
from glob import glob

# Path configuration.
KITTI_ROOT = "./data/KITTI"
IMAGE_DIR = os.path.join(KITTI_ROOT, "image_2")
SPLIT_PATH = os.path.join(KITTI_ROOT, "kitti_split_6_2_2.json")

# Collect image IDs without file extensions.
image_paths = glob(os.path.join(IMAGE_DIR, "*.png"))
image_ids = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

# ✅ Shuffle IDs with a fixed seed for reproducibility.
random.seed(42)
random.shuffle(image_ids)

# Check the image count.
assert len(image_ids) == 7481, f"Unexpected image count: found {len(image_ids)}, expected 7481"

# ✅ Split into 60% training, 20% validation, and 20% testing.
num_total = len(image_ids)
num_train = int(num_total * 0.6)
num_val   = int(num_total * 0.2)
num_test  = num_total - num_train - num_val  # Use the remaining samples for testing.

train_ids = image_ids[:num_train]
val_ids   = image_ids[num_train:num_train + num_val]
test_ids  = image_ids[num_train + num_val:]

# ✅ Save split IDs to JSON.
split_dict = {
    "train": train_ids,
    "val": val_ids,
    "test": test_ids
}
with open(SPLIT_PATH, "w") as f:
    json.dump(split_dict, f, indent=2)

print(f"✅ Split complete: train {len(train_ids)}, val {len(val_ids)}, test {len(test_ids)}")
print(f"📁 JSON saved to: {SPLIT_PATH}")
