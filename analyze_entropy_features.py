import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------- Feature Extraction --------
def extract_entropy_features(sequence):
    seq = np.array(sequence).squeeze()
    length = len(seq)
    thirds = length // 3

    low_band = seq[:thirds] if thirds > 0 else seq
    mid_band = seq[thirds:2*thirds] if thirds > 0 else seq
    high_band = seq[2*thirds:] if thirds > 0 else seq

    return [
        np.mean(low_band),
        np.mean(mid_band),
        np.mean(high_band),
        np.var(seq),
        np.mean(high_band) - np.mean(low_band),  # trend
    ]

# -------- Load Data --------
# Change path if needed
data_path = '/home/tetsu/Desktop/helpDetector/processed_data/train_data.pt'
X_raw, y = torch.load(data_path, weights_only=False)

# -------- Extract Features --------
X_features = []
for seq in X_raw:
    entropy_column = np.array(seq)[:, 2:3]  # Assuming entropy is column index 2
    feat = extract_entropy_features(entropy_column)
    X_features.append(feat)

X_features = np.array(X_features)
y = np.array(y)

# -------- Split and Normalize --------
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------- Train Classifier --------
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_probs = clf.predict_proba(X_test)[:, 1]

# -------- Evaluation --------
print("\n--- Logistic Regression on Entropy Band Features ---")
print(classification_report(y_test, y_pred, target_names=["NO HELP", "HELP"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_probs):.4f}")
