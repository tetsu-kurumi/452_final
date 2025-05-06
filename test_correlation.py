import pandas as pd

import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns

def extract_band_features(seq, bands=3):
    length = len(seq)
    thirds = np.array_split(seq, bands)
    features = []
    for part in thirds:
        if len(part) == 0:
            features.extend([0, 0, 0, 0])  # mean, var, skew, kurt
        else:
            features.extend([
                np.mean(part),
                np.var(part),
                skew(part),
                kurtosis(part)
            ])
    return features

def extract_features(X_seq, mode="all"):
    features = []
    for seq in X_seq:
        au, eu, entropy, perplexity = seq[:, 0], seq[:, 1], seq[:, 2], seq[:, 3]

        if mode == "all":
            feat = (
                extract_band_features(au) +
                extract_band_features(eu) +
                extract_band_features(entropy) +
                extract_band_features(perplexity)
            )
        elif mode == "entropy_only":
            feat = extract_band_features(entropy)
        elif mode == "perplexity_only":
            feat = extract_band_features(perplexity)
        elif mode == "au_only":
            feat = extract_band_features(au)
        elif mode == "eu_only":
            feat = extract_band_features(eu)
        elif mode == "entropy_perplexity":
            feat = extract_band_features(entropy) + extract_band_features(perplexity)
        features.append(feat)
    return np.array(features)

# ---- Load data ----
train_X, train_y = torch.load('/home/tetsu/Desktop/helpDetector/processed_data/train_data.pt',weights_only=False)
test_X, test_y = torch.load('/home/tetsu/Desktop/helpDetector/processed_data/test_data.pt',weights_only=False)

# ---- Feature extraction ----
X_train_feat = extract_features(train_X, mode="all")
X_test_feat = extract_features(test_X, mode="all")

# suppose X_train_feat is your N×M NumPy array and train_y your length-N label array
feature_names = [f"au_band{b}_{stat}" for b in range(3) for stat in ["mean","var","skew","kurt"]] + \
                [f"eu_band{b}_{stat}" for b in range(3) for stat in ["mean","var","skew","kurt"]] + \
                [f"ent_band{b}_{stat}" for b in range(3) for stat in ["mean","var","skew","kurt"]] + \
                [f"ppl_band{b}_{stat}" for b in range(3) for stat in ["mean","var","skew","kurt"]]

df = pd.DataFrame(X_train_feat, columns=feature_names)
df["label"] = train_y  # 0 = NO HELP, 1 = HELP

# compute correlation of every feature with the label
corr_with_label = df.corr()["label"].sort_values(ascending=False)
print(corr_with_label)



plt.figure(figsize=(6,8))
sns.heatmap(corr_with_label.to_frame(), annot=True, cmap="vlag")
plt.title("Feature vs. Label Correlation")
plt.show()
