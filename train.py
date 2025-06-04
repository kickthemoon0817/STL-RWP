import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import STRWP

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -------------------------------
# 1) Dataset Definition (no change)
# -------------------------------
class WaveDataset(Dataset):
    def __init__(self, X_path, Y_path):
        """
        X_path: path to “X_*.npy” with shape (N, K, 3, H, W)
        Y_path: path to “Y_*.npy” with shape (N, 3, H, W)
        """
        self.X = np.load(X_path)  # dtype float32 or float64
        self.Y = np.load(Y_path)
        assert self.X.shape[0] == self.Y.shape[0], "Mismatched samples"

        # Convert once to torch tensors
        self.X_tensor = torch.from_numpy(self.X).float()  # (N, K, 3, H, W)
        self.Y_tensor = torch.from_numpy(self.Y).float()  # (N, 3, H, W)

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

    # Paths for numpy files
    X_train_path = os.path.join(data_dir, "X_train.npy")
    Y_train_path = os.path.join(data_dir, "Y_train.npy")
    X_val_path   = os.path.join(data_dir, "X_val.npy")
    Y_val_path   = os.path.join(data_dir, "Y_val.npy")
    X_test_path  = os.path.join(data_dir, "X_test.npy")
    Y_test_path  = os.path.join(data_dir, "Y_test.npy")

    # Training hyperparameters
    batch_size = 128
    num_epochs = 500
    learning_rate = 1e-3
    weight_decay = 1e-5  # small weight decay
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # -------------------------------
    # 3) DataLoaders (must be inside main())
    # -------------------------------
    train_dataset = WaveDataset(X_train_path, Y_train_path)
    val_dataset   = WaveDataset(X_val_path,   Y_val_path)
    test_dataset  = WaveDataset(X_test_path,  Y_test_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    # -------------------------------
    # 4) Model, Loss, Optimizer
    # -------------------------------
    # Example: K=6, in_channels=3, hidden_dim=60, num_heads=10, rst_blocks=4 (per paper)
    model = STRWP(input_time_steps=6, input_channels=3, embed_dim=60, num_heads=10, num_blocks=4).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning‐rate scheduler (optional)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    # -------------------------------
    # 5) Optionally Load Checkpoint
    # -------------------------------
    start_epoch = 1
    best_val_loss = float("inf")

    # Find all checkpoint files
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "strwp_best_epoch*.pth"))
    if ckpt_files:
        # Load the latest checkpoint by epoch number
        epochs = [int(os.path.basename(f).split("epoch")[1].split(".pth")[0]) for f in ckpt_files]
        latest_epoch = max(epochs)
        latest_ckpt = os.path.join(checkpoint_dir, f"strwp_best_epoch{latest_epoch}.pth")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["val_loss"]
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, val_loss={best_val_loss:.6f}")

    # -------------------------------
    # 6) Training Loop
    # -------------------------------
    for epoch in range(start_epoch, num_epochs + 1):
        # ---- Training ----
        model.train()
        train_loss_accum = 0.0
        for X_batch_cpu, Y_batch_cpu in train_loader:
            # X_batch: (B, K, 3, H, W)
            # Y_batch: (B, 3, H, W)
            X_batch = X_batch_cpu.to(device, non_blocking=True)
            Y_batch = Y_batch_cpu.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(X_batch)  # (B, 3, H, W)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item() * X_batch.size(0)

        train_loss = train_loss_accum / len(train_dataset)

        # ---- Validation ----
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                val_loss_accum += loss.item() * X_batch.size(0)
        val_loss = val_loss_accum / len(val_dataset)

        # Step the scheduler
        scheduler.step(val_loss)

        print(f"Epoch [{epoch}/{num_epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # ---- Save Checkpoint if Best ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(checkpoint_dir, f"strwp_best_epoch{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"  → Saved new best model: {ckpt_path}")

    # -------------------------------
    # 7) Testing (Once Training Is Done)
    # -------------------------------
    # Load best checkpoint (adapt as needed; assume last one saved is the best)
    best_ckpt = sorted(os.listdir(checkpoint_dir))[-1]
    checkpoint = torch.load(os.path.join(checkpoint_dir, best_ckpt), map_location=device)
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

    # Optionally, compute RMSE on SHWW only:
    # If Y_batch and outputs have shape (B, 3, H, W), channel 2 is SHWW:
    model.eval()
    rmse_shww_sum = 0.0
    count = 0
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            outputs = model(X_batch)  # (B, 3, H, W)
            pred_shww = outputs[:, 2, :, :]   # (B, H, W)
            true_shww = Y_batch[:, 2, :, :]    # (B, H, W)
            rmse_shww_sum += torch.sqrt(((pred_shww - true_shww) ** 2).mean(dim=(1, 2))).sum().item()
            count += pred_shww.size(0)
    rmse_shww = rmse_shww_sum / count
    print(f"Test RMSE on SHWW: {rmse_shww:.6f}")


if __name__ == "__main__":
    main()
