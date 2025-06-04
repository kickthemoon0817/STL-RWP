import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from model import STRWP  # make sure this points to your STRWP definition


START = 612

device = "cuda" if torch.cuda.is_available() else "cpu"
model = STRWP(
    input_time_steps=6,
    input_channels=3,
    embed_dim=60,
    num_heads=10,
    window_size=8,
    mlp_ratio=4.0,
    num_blocks=4
).to(device)

chkpt = torch.load("checkpoints/strwp_best_epoch27.pth", map_location=device)
model.load_state_dict(chkpt["model_state_dict"])
model.eval()


x_test = np.load("data/X_test.npy")
y_test = np.load("data/Y_test.npy")     # (T,3,H,W)

x_test_torch = torch.from_numpy(x_test).float().to(device)  # (T,6,3,H,W)

x_test_temp = x_test_torch[START].unsqueeze(0) # (1,6,3,H,W)

# —————————————
# 3) Roll out for future steps
# —————————————
with torch.no_grad():
    future_steps = 6
    preds = model.roll_out(x_test_temp, steps=future_steps)  # (1,6,3,H,W)

y_preds_temp = preds.cpu().numpy()[0]  # (future_steps,3,H,W)

for t in range(future_steps):
    ch = 2   # channel index
    frame_preds = y_preds_temp[t, ch, :, :]     # shape = (H, W)
    frame_true = y_test[t+START, ch, :, :]            # shape = (H, W)

    H, W = frame_preds.shape
    xx = np.arange(0, W)
    yy = np.arange(0, H)
    X, Y = np.meshgrid(xx, yy)
    Z_preds = frame_preds
    Z_true = frame_true

    print(Z_true)

    Z = Z_preds - Z_true

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="inferno", edgecolor='none')
    ax.set_title(f"Predicted channel {ch} at future t={t}")
    ax.set_xlabel("X (grid col)")
    ax.set_ylabel("Y (grid row)")
    ax.set_zlabel("Value")
    plt.show()
