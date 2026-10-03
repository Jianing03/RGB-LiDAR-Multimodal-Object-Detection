import os
import json

# Configure paths.
KITTI_ROOT = "./data/KITTI"
LABEL_DIR = os.path.join(KITTI_ROOT, "label_2")
SPLIT_PATH = os.path.join(KITTI_ROOT, "kitti_split_6_2_2.json")

# Load test split IDs.
with open(SPLIT_PATH, "r") as f:
    split_data = json.load(f)
test_ids = split_data["test"]

# Classify difficulty using KITTI height, truncation, and occlusion criteria.
def get_difficulty(obj):
    trunc = float(obj[1])
    occl = int(obj[2])
    height = float(obj[7]) - float(obj[5])

    if trunc <= 0.15 and occl == 0 and height >= 40:
        return "Easy"
    elif trunc <= 0.3 and occl <= 1 and height >= 25:
        return "Moderate"
    elif trunc <= 0.5 and occl <= 2 and height >= 25:
        return "Hard"
    else:
        return "Unknown"

# Initialize difficulty counters.
difficulty_count = {"Easy": 0, "Moderate": 0, "Hard": 0, "Unknown": 0}

# Read each test image's label file.
for img_id in test_ids:
    label_path = os.path.join(LABEL_DIR, f"{img_id}.txt")
    if not os.path.exists(label_path):
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()

    difficulties = set()
    for line in lines:
        fields = line.strip().split()
        obj_type = fields[0]
        if obj_type != "Car":  # Evaluate only the Car class.
            continue
        diff = get_difficulty(fields)
        difficulties.add(diff)

    # Assign the hardest recognized Car difficulty; use Unknown if none qualifies.
    if "Hard" in difficulties:
        difficulty_count["Hard"] += 1
    elif "Moderate" in difficulties:
        difficulty_count["Moderate"] += 1
    elif "Easy" in difficulties:
        difficulty_count["Easy"] += 1
    else:
        difficulty_count["Unknown"] += 1

# Print results.
print("📊 Test image counts by difficulty:")
for k, v in difficulty_count.items():
    print(f"  {k:<8}: {v}")
