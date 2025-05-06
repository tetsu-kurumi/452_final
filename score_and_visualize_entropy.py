import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# ====== Configuration ======
DATA_PATH = "/home/tetsu/Desktop/helpDetector/processed_data/train_data.pt"  # <- change this
SAVE_DIR = "./entropy_score_viz"
TOP_K = 5  # Number of top and bottom sequences to visualize

os.makedirs(SAVE_DIR, exist_ok=True)


def evaluate_entropy_scores(scores, threshold=0.3, plot_roc=True):
    """
    Args:
        scores: list of tuples (score: float, idx: int, label: int)
        threshold: float for classifying help vs no help
        plot_roc: bool, whether to plot ROC curve
    """

    # Unpack scores and labels
    all_scores = np.array([s[0] for s in scores])
    true_labels = np.array([s[2] for s in scores])

    # Predict labels using threshold
    pred_labels = (all_scores >= threshold).astype(int)

    # Print evaluation metrics
    print(f"\n--- Threshold = {threshold:.2f} ---")
    print(classification_report(true_labels, pred_labels, target_names=["NO HELP", "HELP"]))
    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, pred_labels))

    # Compute and print ROC AUC
    try:
        auc = roc_auc_score(true_labels, all_scores)
        print(f"ROC AUC Score: {auc:.4f}")
    except:
        print("ROC AUC could not be computed (e.g., only one class present).")

    # Plot ROC curve
    if plot_roc:
        fpr, tpr, thresholds = roc_curve(true_labels, all_scores)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'Entropy Score (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Frequency-Weighted Entropy Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# ====== Frequency-aware scoring ======
def sequence_should_ask_for_help(entropy_seq, weight_fn=None):
    T = len(entropy_seq)
    if T == 0:
        return 0.0

    if weight_fn is None:
        weight_fn = lambda i: 1.0 - (i / T)

    weights = np.array([weight_fn(i) for i in range(T)])
    return float(np.average(entropy_seq, weights=weights))


# ====== Load Data ======
X, y = torch.load(DATA_PATH, weights_only=False)
print(f"Loaded {len(X)} sequences")

# ====== Compute Scores ======
scores = []
for i, seq in enumerate(X):
    entropy = seq[:, 2]  # entropy column
    score = sequence_should_ask_for_help(entropy)
    scores.append((score, i, y[i]))

evaluate_entropy_scores(scores, threshold=0.3)

# ====== Sort and Select Sequences ======
scores.sort(reverse=True)
top = scores[:TOP_K]
bottom = scores[-TOP_K:]

# ====== Visualization Function ======
def plot_entropy_sequence(seq, label, score, index, save_path):
    entropy = seq[:, 2]
    plt.figure(figsize=(8, 3))
    plt.plot(entropy, marker='o')
    plt.title(f"Seq #{index} | Label: {label} | Score: {score:.4f}")
    plt.xlabel("Token Index (Low → High Freq)")
    plt.ylabel("Entropy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ====== Generate Plots ======
for score, idx, label in top:
    plot_entropy_sequence(X[idx], label, score, idx, os.path.join(SAVE_DIR, f"top_{idx}_score_{score:.3f}.png"))

for score, idx, label in bottom:
    plot_entropy_sequence(X[idx], label, score, idx, os.path.join(SAVE_DIR, f"bottom_{idx}_score_{score:.3f}.png"))

print(f"Saved visualizations to {SAVE_DIR}")
