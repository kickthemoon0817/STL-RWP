import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformerBlock

class ResidualSwinTransformerBlock(nn.Module):
    """
    Residual Swin Transformer block (RST) = two SwinTransformerBlocks (one normal, one shifted)
    + a 3x3 convolution + a residual skip connection over all of them.
    Input & output shape: (B, dim, 40, 40).
    """
    def __init__(self, dim, num_heads, window_size, mlp_ratio=4.0):
        super().__init__()  
        input_resolution = (40, 40)

        # 1st Swin block: no shift
        self.swin1 = SwinTransformerBlock(
            dim=dim,
            input_resolution=input_resolution, 
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,            # no shift
            mlp_ratio=mlp_ratio,
        )
        # 2nd Swin block: shifted by half‐window
        self.swin2 = SwinTransformerBlock(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2,  # shift = half window
            mlp_ratio=mlp_ratio,
        )
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)   

    def forward(self, x):
        # Input shape: (B, dim, 40, 40)
        residual = x

        x = x.permute(0, 2, 3, 1)   # (B, 40, 40, dim)
        x = self.swin1(x)
        x = self.swin2(x)

        x = x.permute(0, 3, 1, 2)   # (B, dim, 40, 40)
        x = self.conv(x)            # (B, dim, 40, 40)
        x = x + residual            # residual skip over the entire RST block
        return x


class STRWP(nn.Module):
    """
    ST-RWP (Swin Transformer for Regional Wave Prediction) model.
    - Input: 6 time steps, each with 3 channels on a 40x40 grid -> (B, 6, 3, 40, 40)
    - Output: next frame with 3 channels on 40x40 -> (B, 3, 40, 40)
    - Rolls out autoregressively if needed.
    """
    def __init__(
        self,
        input_time_steps=6,
        input_channels=3,
        embed_dim=60,
        num_heads=10,
        window_size=8,
        mlp_ratio=4.0,
        num_blocks=4
    ):
        super().__init__()
        self.input_time_steps = input_time_steps
        self.input_channels = input_channels

        # Initial convolution: (6*3)-> embed_dim
        self.conv_in = nn.Conv2d(
            input_time_steps * input_channels,
            embed_dim,
            kernel_size=3,
            padding=1
        )
        # Stack of Residual Swin Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            ResidualSwinTransformerBlock(embed_dim, num_heads, window_size, mlp_ratio)
            for _ in range(num_blocks)
        ])
        # Final convolution back to 3 channels
        self.conv_3 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(embed_dim, input_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Forward pass for one-step prediction.
        x: (B, 6, 3, 40, 40)
        Returns: (B, 3, 40, 40)
        """
        B, T, C, H, W = x.shape

        # Merge time and channel dims => (B, 6*3, 40, 40)
        x = x.view(B, T * C, H, W)
        x = self.conv_in(x)                 # (B, embed_dim, 40, 40)
        residual_1 = x
        for block in self.transformer_blocks:
            x = block(x)                    # (B, embed_dim, 40, 40)
        x = self.conv_3(x)                # (B, 3, 40, 40)
        x = x + residual_1
        x = self.conv_out(x)
        return x

    def roll_out(self, x, steps):
        """
        Autoregressive rollout for multiple future steps.
        x: initial sequence (B, 6, 3, 40, 40)
        steps: number of future frames to generate
        Returns: (B, steps, 3, 40, 40)
        """
        preds = []
        current_input = x
        for _ in range(steps):
            next_frame = self.forward(current_input)     # (B, 3, 40, 40)
            preds.append(next_frame.unsqueeze(1))        # (B, 1, 3, 40, 40)
            # Slide window: drop oldest, append predicted
            current_input = torch.cat([current_input[:, 1:], next_frame.unsqueeze(1)], dim=1)
        return torch.cat(preds, dim=1)  # (B, steps, 3, 40, 40)

import numpy as np

test = np.zeros((1, 6, 3, 40, 40))
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

model.eval()
with torch.no_grad():
    test_result = model(torch.from_numpy(test).float().to(device))
    print(test_result)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = STRWP(
#     input_time_steps=6,
#     input_channels=3,
#     embed_dim=60,
#     num_heads=10,
#     window_size=8,
#     mlp_ratio=4.0,
#     num_blocks=4
# ).to(device)

# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f"Total parameters: {total_params}")
# print(f"Trainable parameters: {trainable_params}")
