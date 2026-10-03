import os
import re
import matplotlib.pyplot as plt

# Log file paths; set these to your local logs.
log_files = {
    "Data level Fusion": "./results/log/data_fusion.txt",
    "Feature level Fusion": "./results/log/feature_fusion.txt",
    "Decision level Fusion": "./results/log/decision_fusion.txt"
}
# Plot output directory.
save_dir = r"./results/picture"
os.makedirs(save_dir, exist_ok=True)

# Initialize metric storage.
metrics = {name: {"epoch": [], "P": [], "R": [], "F1": [], "AP": []} for name in log_files}

# Match metrics with a regular expression.
pattern = re.compile(r"\[Epoch (\d+)] P: ([\d.]+) R: ([\d.]+) F1: ([\d.]+) AP: ([\d.]+)")

# Read and parse log files.
for name, path in log_files.items():
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch, p, r, f1, ap = match.groups()
                metrics[name]["epoch"].append(int(epoch))
                metrics[name]["P"].append(float(p))
                metrics[name]["R"].append(float(r))
                metrics[name]["F1"].append(float(f1))
                metrics[name]["AP"].append(float(ap))

# Plotting function.
def plot_metric(metric_key, ylabel, title, filename):
    plt.figure()
    for name in log_files:
        plt.plot(metrics[name]["epoch"], metrics[name][metric_key], label=name)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved: {save_path}")

# Generate plots.
plot_metric("P", "Precision", "Precision vs Epoch", "precision.png")
plot_metric("R", "Recall", "Recall vs Epoch", "recall.png")
plot_metric("F1", "F1 Score", "F1 Score vs Epoch", "f1.png")
plot_metric("AP", "Average Precision", "AP vs Epoch", "ap.png")