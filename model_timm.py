import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformerBlock


class STRWP(nn.Module):
    """
    ST-RWP (Swin Transformer for Regional Wave Prediction) using timm’s SwinTransformerBlock.

    - Input: (B, 6, 3, 40, 40)  →  six time‐steps, each with 3 channels on a 40×40 grid
    - Output: (B, 3, 40, 40)    →  the next frame (3 channels)
    - roll_out(...) can be used to autoregressively generate multiple future frames.
    """
    def __init__(
        self,
        input_time_steps: int = 6,
        input_channels: int   = 3,
        embed_dim: int        = 60,
        num_heads: int        = 10,
        window_size: int      = 8,
        mlp_ratio: float      = 4.0,
        num_blocks: int       = 4
    ):
        super().__init__()
        self.input_time_steps = input_time_steps    # e.g. 6
        self.input_channels   = input_channels      # e.g. 3
        self.embed_dim        = embed_dim           # e.g. 60
        self.num_heads        = num_heads           # e.g. 10
        self.window_size      = window_size         # e.g. 8
        self.mlp_ratio        = mlp_ratio           # e.g. 4.0
        self.num_blocks       = num_blocks          # e.g. 4
        self.H = 40
        self.W = 40

        # --------------------------------------------------------------------
        # 1) conv_in: collapse (6 time‐steps × 3 channels) = 18 → embed_dim
        # --------------------------------------------------------------------
        self.conv_in = nn.Conv2d(
            in_channels  = input_time_steps * input_channels,  # 6×3 = 18
            out_channels = embed_dim,                           # 60
            kernel_size  = 3,
            padding      = 1
        )

        # -----------------------------------------------------------------------------
        # 2) Build a stack of SwinTransformerBlock layers (from timm), alternating shift
        # -----------------------------------------------------------------------------
        # Each block in timm expects:
        #   - dim = embed_dim
        #   - input_resolution = (H, W)  = (40, 40)
        #   - num_heads = num_heads
        #   - window_size = window_size
        #   - shift_size = 0 for even‐indexed block, window_size//2 for odd
        #
        # Internally, each SwinTransformerBlock will:
        #   (a) apply LayerNorm to (B, L, C) where L=H*W
        #   (b) reshape → (B, H, W, C), do window_partition, shift, W-MSA/SW-MSA, etc.
        #   (c) return tokens of shape (B, H*W, C)
        #
        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=embed_dim,
                input_resolution=(self.H, self.W),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(window_size // 2 if (i % 2 == 1) else 0),
                mlp_ratio=mlp_ratio
            )
            for i in range(num_blocks)
        ])

        # --------------------------------------------------------------
        # 3) conv_out: map back from embed_dim → 3 channels (wave variables)
        # --------------------------------------------------------------
        self.conv_out = nn.Conv2d(
            in_channels  = embed_dim,
            out_channels = input_channels,  # 3
            kernel_size  = 3,
            padding      = 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one‐step forecast.
        Args:
            x: (B, 6, 3, 40, 40)
        Returns:
            (B, 3, 40, 40)
        """
        B, T, C, H, W = x.shape
        assert T == self.input_time_steps and C == self.input_channels
        assert H == self.H and W == self.W

        # 1) Merge time & channel dims → (B, 6*3, 40, 40)
        x = x.view(B, T * C, H, W)

        # 2) conv_in → (B, embed_dim, 40, 40)
        x = self.conv_in(x)

        # 3) Flatten for timm’s Swin blocks:
        #    - First reshape from (B, embed_dim, 40, 40) → (B, 40*40, embed_dim)
        #      by doing .flatten(2).transpose(1, 2).
        #    - Then pass through each SwinTransformerBlock, which expects (B, L, C).
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)  where H*W = 1600

        for block in self.swin_blocks:
            # Each block returns (B, H*W, embed_dim)
            x = block(x)

        # 4) Reshape back to (B, embed_dim, 40, 40)
        x = x.transpose(1, 2).view(B, self.embed_dim, H, W)

        # 5) conv_out → (B, 3, 40, 40)
        x = self.conv_out(x)
        return x

    def roll_out(self, x: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Autoregressive rollout for multiple future steps.
        Args:
            x: initial sequence of shape (B, 6, 3, 40, 40)
            steps: how many future frames to generate
        Returns:
            Tensor of shape (B, steps, 3, 40, 40)
        """
        preds = []
        current_input = x  # (B, 6, 3, 40, 40)

        for _ in range(steps):
            next_frame = self.forward(current_input)       # (B, 3, 40, 40)
            preds.append(next_frame.unsqueeze(1))          # (B, 1, 3, 40, 40)
            # Slide the window: drop the oldest time-step, append the predicted
            current_input = torch.cat(
                [current_input[:, 1:], next_frame.unsqueeze(1)], 
                dim=1
            )  # new shape: (B, 6, 3, 40, 40)

        return torch.cat(preds, dim=1)  # (B, steps, 3, 40, 40)

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

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
