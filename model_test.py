import torch
import torch.nn as nn
import torch.nn.functional as F 
from timm.models import SwinTransformer

SwinTransformer()

class STRWP(nn.Module):
    """
    ST-RWP (Swin Transformer for Regional Wave Prediction) model.
    - Input: 6 time steps, each with 3 channels on a 40x40 grid -> (B, 6, 3, 40, 40)
    - Output: next frame with 3 channels on 40x40 -> (B, 3, 40, 40)
    - Rolls out autoregressively if needed.
    """
    def __init__(self,
                 input_time_steps=6,
                 input_channels=3,
                 embed_dim=60,
                 num_heads=10,
                 window_size=8,
                 mlp_ratio=4.0,
                 num_blocks=4):
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
        for block in self.transformer_blocks:
            x = block(x)                    # (B, embed_dim, 40, 40)
        x = self.conv_out(x)                # (B, 3, 40, 40)
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
