import torch
import matplotlib.pyplot as plt
import numpy as np

# Load processed data
train_path = '/home/tetsu/Desktop/helpDetector/processed_data/train_data.pt'
X_train, y_train = torch.load(train_path,weights_only=False)

# Convert to numpy if needed
X_train = np.array(X_train, dtype=object)
y_train = np.array(y_train)

# Feature names and indices
features = {
    "AU (Aleatoric Uncertainty)": 0,
    "EU (Epistemic Uncertainty)": 1,
    "Entropy": 2,
    "Perplexity": 3,
}

# Set up subplots: 2 rows (NO HELP, HELP) x 4 features
fig, axs = plt.subplots(2, 4, figsize=(20, 8))
label_names = ['NO HELP', 'HELP']
colors = ['blue', 'red']

for col, (feature_name, feat_idx) in enumerate(features.items()):
    for row, label_val in enumerate([0, 1]):  # 0 = NO HELP, 1 = HELP
        ax = axs[row, col]
        for seq, label in zip(X_train, y_train):
            if label == label_val:
                values = seq[:, feat_idx]
                ax.plot(values, color=colors[row], alpha=0.2)
        
        ax.set_title(f"{feature_name} ({label_names[row]})")
        ax.set_xlabel("Token Index")
        if col == 0:
            ax.set_ylabel("Value")
        ax.grid(True)

plt.suptitle("Uncertainty Feature Sequences Grouped by Label", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()