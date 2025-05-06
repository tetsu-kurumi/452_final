import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import warnings
warnings.filterwarnings("ignore")

folder = input("Enter folder with all inference .npy steps: ").strip()
out_dir = os.path.join(folder, "global_summary_output")
os.makedirs(out_dir, exist_ok=True)

# First pass: gather all AU for global normalization
all_au_raw = []
step_files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
print(f"Found {len(step_files)} step files.")

# for f in tqdm(step_files, desc="Collecting AU for normalization"):
#     try:
#         data = np.load(os.path.join(folder, f), allow_pickle=True).item()
#         au = np.array(data["outputs/au"])
#         if len(au) > 5:
#             au = au[3:-2]
#             all_au_raw.extend(au)
#     except:
#         continue

# all_au_raw = np.array(all_au_raw)
# mean = np.mean(all_au_raw)
# std = np.std(all_au_raw)

# # Normalization function
# def normalize_au(x):
#     return 1 / (1 + np.exp(-(x - mean) / std))

# Initialize containers
all_au, all_eu, all_entropy, all_unreliability = [], [], [], []
step_perplexities, step_mean_unreliability, step_mean_entropy = [], [], []
first_unreliability = []

# Second pass: process files
for f in tqdm(step_files, desc="Processing steps"):
    try:
        data = np.load(os.path.join(folder, f), allow_pickle=True).item()

        au = np.array(data["outputs/au"])
        eu = np.array(data["outputs/eu"])
        entropy = np.array(data["outputs/entropy"])
        log_probs = np.array(data["outputs/perplexity"])  # actually log-probs

        if len(au) > 5:
            au = au[3:-2]
            eu = eu[3:-2]
            entropy = entropy[3:-2]
            log_probs = log_probs[3:-2]
        else:
            continue

        # norm_au = normalize_au(au)
        # unreliability = norm_au * eu

        unreliability = au * eu

        # Token-level storage
        # all_au.extend(norm_au)
        all_au.extend(au)
        all_eu.extend(eu)
        all_entropy.extend(entropy)
        all_unreliability.extend(unreliability)
        first_unreliability.append(unreliability[0])

        if len(log_probs) > 0:
            ppl = np.exp(-np.mean(log_probs))
            step_perplexities.append(ppl)
            step_mean_unreliability.append(np.mean(unreliability))
            step_mean_entropy.append(np.mean(entropy))

    except Exception as e:
        print(f"Skipping {f}: {e}")

# Convert to arrays
all_au = np.array(all_au)
all_eu = np.array(all_eu)
all_entropy = np.array(all_entropy)
all_unreliability = np.array(all_unreliability)
step_perplexities = np.array(step_perplexities)
step_mean_unreliability = np.array(step_mean_unreliability)
step_mean_entropy = np.array(step_mean_entropy)
first_unreliability = np.array(first_unreliability)

# --- Save Summary Statistics ---
summary_path = os.path.join(out_dir, "summary.txt")
with open(summary_path, "w") as f:
    f.write("=== Token-Level Statistics ===\n")
    for name, array in [("AU", all_au), ("EU", all_eu), ("Entropy", all_entropy), ("Unreliability", all_unreliability)]:
        f.write(f"\n{name}:\n")
        f.write(f"  Mean: {np.mean(array):.4f}\n")
        f.write(f"  Std: {np.std(array):.4f}\n")
        f.write(f"  Max: {np.max(array):.4f}\n")
        f.write(f"  Min: {np.min(array):.4f}\n")

    f.write("\n=== Step-Level Statistics ===\n")
    f.write(f"\nPerplexity:\n")
    f.write(f"  Mean: {np.mean(step_perplexities):.4f}\n")
    f.write(f"  Std: {np.std(step_perplexities):.4f}\n")
    f.write(f"  Max: {np.max(step_perplexities):.4f}\n")
    f.write(f"  Min: {np.min(step_perplexities):.4f}\n")

    f.write(f"\nStep Mean Unreliability:\n")
    f.write(f"  Mean: {np.mean(step_mean_unreliability):.4f}\n")
    f.write(f"  Std: {np.std(step_mean_unreliability):.4f}\n")
    f.write(f"  Max: {np.max(step_mean_unreliability):.4f}\n")
    f.write(f"  Min: {np.min(step_mean_unreliability):.4f}\n")

    f.write(f"\nStep Mean Entropy:\n")
    f.write(f"  Mean: {np.mean(step_mean_entropy):.4f}\n")
    f.write(f"  Std: {np.std(step_mean_entropy):.4f}\n")
    f.write(f"  Max: {np.max(step_mean_entropy):.4f}\n")
    f.write(f"  Min: {np.min(step_mean_entropy):.4f}\n")

    f.write("\n=== First Token Statistics ===\n")
    f.write(f"\nUnreliability:\n")
    f.write(f"  Mean: {np.mean(first_unreliability):.4f}\n")
    f.write(f"  Std: {np.std(first_unreliability):.4f}\n")
    f.write(f"  Max: {np.max(first_unreliability):.4f}\n")
    f.write(f"  Min: {np.min(first_unreliability):.4f}\n")

print(f"Saved summary stats to {summary_path}")

# --- Save Histograms ---
def save_hist(data, name):
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=50, alpha=0.75)
    plt.title(f"{name} Histogram")
    plt.xlabel(name)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name.lower().replace(' ', '_')}_hist.png"))
    plt.close()

save_hist(all_au, "AU")
save_hist(all_eu, "EU")
save_hist(all_entropy, "Entropy")
save_hist(all_unreliability, "Unreliability")
save_hist(step_perplexities, "Step Perplexity")
save_hist(step_mean_unreliability, "Step Mean Unreliability")
save_hist(step_mean_entropy, "Step Mean Entropy")
save_hist(first_unreliability, "First Token Unreliability")

print("Saved histograms.")
