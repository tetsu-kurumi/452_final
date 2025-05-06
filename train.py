import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.lstm import UncertaintyLSTM
from models.rnn import UncertaintyRNN
from models.transformer import UncertaintyTransformer
from models.mlp import SimpleMLP
import numpy as np

# ----- Dataset Definition -----
class UncertaintyDataset(Dataset):
    def __init__(self, data_path):
        self.X, self.y = torch.load(data_path, weights_only=False)
        # self.add_positional_encoding()
        # self.use_entropy_and_position_only()
        # self.use_au_eu_with_position()
        self.use_entropy_and_perplexity_with_pos()

    def add_positional_encoding(self):
        new_X = []
        for seq in self.X:
            seq_len = len(seq)
            positions = np.arange(seq_len).reshape(-1, 1) / seq_len  # shape: [seq_len, 1]
            seq_with_pos = np.concatenate([seq, positions], axis=-1)  # now [seq_len, 5]
            new_X.append(seq_with_pos)
        self.X = new_X

    def use_entropy_and_position_only(self):
        new_X = []
        for seq in self.X:
            seq_len = len(seq)
            if seq_len == 0:
                continue
            entropy = seq[:, 2:3]  # Only the entropy column
            positions = np.arange(seq_len).reshape(-1, 1) / seq_len  # Normalize position
            entropy_pos = np.concatenate([entropy, positions], axis=-1)  # Shape: [seq_len, 2]
            new_X.append(entropy_pos)
        self.X = new_X
    
    def use_entropy_and_perplexity_with_pos(self):
        new_X = []
        for seq in self.X:
            seq_len = len(seq)
            if seq_len == 0:
                continue
            entropy = seq[:, 2:3]  # Only the entropy column
            perplexity = seq[:, 3:4]
            positions = np.arange(seq_len).reshape(-1, 1) / seq_len  # Normalize position
            features = np.concatenate([entropy, perplexity, positions], axis=-1)  # Shape: [seq_len, 3]
            new_X.append(features)
        self.X = new_X

    def use_au_eu_with_position(self):
        new_X = []
        for seq in self.X:
            seq_len = len(seq)
            if seq_len == 0:
                continue
            au = seq[:, 0:1]  # AU column
            eu = seq[:, 1:2]  # EU column
            positions = np.arange(seq_len).reshape(-1, 1) / seq_len  # Normalize position
            features = np.concatenate([au, eu, positions], axis=-1)  # Shape: [seq_len, 3]
            new_X.append(features)
        self.X = new_X

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    padded_seqs = nn.utils.rnn.pad_sequence(
        [torch.tensor(seq, dtype=torch.float32) for seq in sequences],
        batch_first=True
    )
    return padded_seqs, torch.tensor(labels), torch.tensor(lengths)

# ----- Data Loaders -----
train_dataset = UncertaintyDataset('/home/tetsu/Desktop/helpDetector/processed_data/train_data.pt')
test_dataset = UncertaintyDataset('/home/tetsu/Desktop/helpDetector/processed_data/test_data.pt')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# ----- Training Setup -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = UncertaintyLSTM(input_dim=5, hidden_dim=64).to(device)
# model = UncertaintyRNN(input_dim=5, hidden_dim=64).to(device)
model = UncertaintyTransformer(input_dim=3).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ----- Training Loop -----

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y, lengths in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device).float()
        
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# ----- Testing / Evaluation -----
model.eval()
correct, total = 0, 0
tp, fp, tn, fn = 0, 0, 0, 0
with torch.no_grad():
    for batch_X, batch_y, lengths in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        predictions = (outputs >= 0.5).long()
        
        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)
        
        # Compute detailed metrics
        tp += ((predictions == 1) & (batch_y == 1)).sum().item()
        fp += ((predictions == 1) & (batch_y == 0)).sum().item()
        tn += ((predictions == 0) & (batch_y == 0)).sum().item()
        fn += ((predictions == 0) & (batch_y == 1)).sum().item()

accuracy = correct / total
precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1_score = 2 * precision * recall / (precision + recall + 1e-8)

print("\n--- Test Results ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1_score:.4f}")

print("\nConfusion Matrix:")
print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
