import os
import numpy as np
import torch
import json
from sklearn.model_selection import train_test_split

base_dir = input("Enter the base directory containing episode folders: ").strip()
output_dir = os.path.join(base_dir, "processed_data")
os.makedirs(output_dir, exist_ok=True)

X, y = [], []

episode_folders = sorted([
    os.path.join(base_dir, d)
    for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d))
])

for ep in episode_folders:
    label_path = os.path.join(ep, "labels.json")
    if not os.path.isfile(label_path):
        print(f"Missing labels.json in {ep}, skipping.")
        continue

    with open(label_path, 'r') as f:
        labels_data = json.load(f)

    if "labels" not in labels_data:
        print(f"No labels in {ep}, skipping.")
        continue

    for item in labels_data["labels"]:
        step_file = item["step"]
        label = item["label"]

        full_path = os.path.join(ep, step_file)
        if not os.path.isfile(full_path):
            print(f"Missing {full_path}, skipping.")
            continue

        try:
            data = np.load(full_path, allow_pickle=True).item()

            au = np.array(data["outputs/au"])
            eu = np.array(data["outputs/eu"])
            entropy = np.array(data["outputs/entropy"])
            perplexity = np.array(data["outputs/perplexity"])

            # Remove zero-padding and drop edge tokens
            mask = ~((au == 0) & (eu == 0) & (entropy == 0) & (perplexity == 0))
            au, eu, entropy, perplexity = au[mask], eu[mask], entropy[mask], perplexity[mask]

            if len(au) > 5:
                au, eu, entropy, perplexity = au[3:-2], eu[3:-2], entropy[3:-2], perplexity[3:-2]
            else:
                continue  # Skip if too short

            features = np.stack([au, eu, entropy, perplexity], axis=-1)  # [seq_len, 4]
            X.append(features)
            y.append(label)

        except Exception as e:
            print(f"Failed loading {full_path}: {e}")
            continue

# Convert to arrays
X = np.array(X, dtype=object)  # Variable-length
y = np.array(y)

print(f"\n✅ Collected {len(X)} labeled examples.")
print(f"   Positive (HELP): {np.sum(y)}, Negative (NO HELP): {len(y) - np.sum(y)}")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save datasets
torch.save((X_train, y_train), os.path.join(output_dir, 'train_data.pt'))
torch.save((X_test, y_test), os.path.join(output_dir, 'test_data.pt'))

print(f"📦 Data saved to '{output_dir}'")
