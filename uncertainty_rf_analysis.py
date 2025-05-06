import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split
from scipy.stats import skew, kurtosis

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
X_train_feat = extract_features(train_X, mode="entropy_perplexity")
X_test_feat = extract_features(test_X, mode="entropy_perplexity")

# ---- Classifier ----
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_feat, train_y)

# ---- Predict and Evaluate ----
y_pred = clf.predict(X_test_feat)
y_prob = clf.predict_proba(X_test_feat)[:, 1]  # Probability of class '1'

print("\n--- Random Forest on Band Features ---")
print(classification_report(test_y, y_pred, target_names=["NO HELP", "HELP"]))
print("Confusion Matrix:")
print(confusion_matrix(test_y, y_pred))
print(f"ROC AUC Score: {roc_auc_score(test_y, y_prob):.4f}")
