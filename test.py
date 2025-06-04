import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import STRWP


class WaveDataset(Dataset):
    def __init__(self, X_path, Y_path, device="cpu"):
        """
        X_path: path to “X_*.npy” with shape (N, K, 3, H, W)
        Y_path: path to “Y_*.npy” with shape (N, 3, H, W)
        """
        self.X = np.load(X_path)  # dtype float32 or float64
        self.Y = np.load(Y_path)
        assert self.X.shape[0] == self.Y.shape[0], "Mismatched samples"

        # Convert once to torch tensors
        self.X_tensor = torch.from_numpy(self.X).float().to(device)  # (N, K, 3, H, W)
        self.Y_tensor = torch.from_numpy(self.Y).float().to(device)  # (N, 3, H, W)

    def __len__(self):
        return self.X_tensor.shape[0]

    def __getitem__(self, idx):
        # Return: (input_seq, target_frame)
        return self.X_tensor[idx], self.Y_tensor[idx]
    

def main():
    # -------------------------------
    # 2) Configuration & Hyperparams
    # -------------------------------
    data_dir = "./data"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 128
    checkpoint_dir = "./checkpoints"

    X_test_path  = os.path.join(data_dir, "X_test.npy")
    Y_test_path  = os.path.join(data_dir, "Y_test.npy")

    test_dataset  = WaveDataset(X_test_path,  Y_test_path, device)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    criterion = nn.MSELoss()

    model = STRWP(input_time_steps=6, input_channels=3, embed_dim=60, num_heads=10, num_blocks=4).to(device)

    checkpoint = torch.load(os.path.join(checkpoint_dir, "strwp_best_epoch27.pth"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    test_loss_accum = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            test_loss_accum += loss.item() * X_batch.size(0)
    test_loss = test_loss_accum / len(test_dataset)
    print(f"Test Loss (MSE): {test_loss:.6f}")

if __name__ == "__main__":
    main()