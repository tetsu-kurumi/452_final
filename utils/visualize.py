import torch
import matplotlib.pyplot as plt
import argparse
import os

def plot_sequence(seq, label, seq_idx, save_dir=None):
    """
    seq: Tensor of shape [seq_len, 4] or [seq_len, 5]
    label: int (0 or 1)
    seq_idx: int (index of the sequence in the dataset)
    save_dir: optional directory to save plots
    """
    # seq = seq.cpu().numpy()
    has_pos = seq.shape[1] == 5

    labels = ['AU', 'EU', 'Entropy', 'Perplexity']
    colors = ['red', 'blue', 'green', 'orange']

    plt.figure(figsize=(12, 5))
    for i in range(4):
        plt.plot(seq[:, i], label=labels[i], color=colors[i])
    
    if has_pos:
        plt.plot(seq[:, 4], label='Pos', linestyle='dotted', color='gray')

    plt.title(f'Sequence #{seq_idx} - Label: {"HELP" if label == 1 else "NO HELP"}')
    plt.xlabel('Token Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'seq_{seq_idx}_label_{label}.png')
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main(dataset_path, num_samples=10, save_dir=None):
    X, y = torch.load(dataset_path, weights_only=False)
    print(f"Loaded {len(X)} sequences from {dataset_path}")
    
    for i in range(min(len(X), num_samples)):
        plot_sequence(X[i], y[i], i, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to the .pt dataset file")
    parser.add_argument("--num", type=int, default=10, help="Number of sequences to plot")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save plots instead of showing")
    args = parser.parse_args()

    main(args.dataset, args.num, args.save_dir)
