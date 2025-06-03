# main.py

import pathlib
import numpy as np
import torch

from model import STRWP, rolled_out_prediction

def prepare_input_sequence(u10n_arr, v10n_arr, swh_arr, K):
    """
    Given three numpy arrays of shape (T, H, W), extract the first K time‐steps
    and stack them into a tensor of shape (1, K, 3, H, W) ready for the model.
    """
    # Ensure all arrays have the same shape
    assert u10n_arr.shape == v10n_arr.shape == swh_arr.shape, \
        "u10n_arr, v10n_arr, and swh_arr must all have shape (T, H, W)"
    
    T, H, W = u10n_arr.shape
    assert T >= K, f"Need at least K={K} time steps, but got T={T}"

    # Take the first K frames of each variable
    u_window = u10n_arr[0:K]  # shape: (K, H, W)
    v_window = v10n_arr[0:K]  # shape: (K, H, W)
    swh_window = swh_arr[0:K] # shape: (K, H, W)

    # Stack into (K, 3, H, W)
    # Order: channel 0 = u10n, 1 = v10n, 2 = swh
    stacked = np.stack([u_window, v_window, swh_window], axis=1)  
    # Now stacked has shape (K, 3, H, W)

    # Add a batch dimension → (1, K, 3, H, W)
    input_seq = torch.from_numpy(stacked).float().unsqueeze(0)
    return input_seq  # dtype=torch.float32

def main():
    # Path to your .npy files
    data_path = pathlib.Path("./data")
    u10n_arr = np.load(data_path / "u10n_array.npy")          # shape: (T, H, W)
    v10n_arr = np.load(data_path / "v10n_array.npy")          # shape: (T, H, W)
    swh_arr = np.load(data_path / "swh_interp_array.npy")     # shape: (T, H, W)

    # Hyperparameters
    K = 6                # Number of input timesteps
    rollout_steps = 6    # Number of steps to predict ahead
    hidden_dim = 60      # Must match the STRWP definition
    num_heads = 10       # Must match the STRWP definition
    rst_blocks = 4       # Must match the STRWP definition

    # Prepare a single‐batch input (1, K, 3, H, W)
    input_seq = prepare_input_sequence(u10n_arr, v10n_arr, swh_arr, K)

    # Instantiate the model (with the same defaults as in model.py)
    model = STRWP(
        K=K,
        in_channels=3,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        rst_blocks=rst_blocks
    )

    # If you have saved weights, load them here. Otherwise, this uses random init:
    # model.load_state_dict(torch.load("path_to_saved_weights.pth"))

    # Move model + data to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_seq = input_seq.to(device)

    # Run rolled‐out prediction
    preds = rolled_out_prediction(model, input_seq, steps=rollout_steps)
    # preds has shape (1, rollout_steps, 3, H, W)

    # Convert predictions back to numpy
    preds_np = preds.squeeze(0).cpu().numpy()  
    # Now preds_np has shape (rollout_steps, 3, H, W)

    # Save each predicted frame into .npy files
    for t in range(rollout_steps):
        print("u10n:", preds_np[t, 0])
        print("v10n:", preds_np[t, 1])
        print("swh:", preds_np[t, 2])

if __name__ == "__main__":
    main()
