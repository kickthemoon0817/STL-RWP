import torch
import torch.nn as nn
import torch.nn.functional as F

class SwinTransformerBlock(nn.Module):
    """
    Single Swin Transformer block with:
      - window-based multi-head self-attention (with optional shift),
      - an MLP feed-forward,
      - LayerNorm before each sub-layer,
      - residual connections for both attention and MLP.
    Input: (B, C, H, W)
    Output: (B, C, H, W) with C = dim.
    """
    def __init__(self, dim, num_heads, window_size, shift=False, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        # If shift=True, we will shift by half-window_size along H and W:
        self.shift_size = window_size // 2 if shift else 0

        # LayerNorm before QKV projection
        self.norm1 = nn.LayerNorm(dim)
        # Combined QKV linear projection
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        # Output projection after attention
        self.proj = nn.Linear(dim, dim)

        # LayerNorm before MLP
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        # Scaling factor for dot-product attention
        self.scale = (dim // num_heads) ** -0.5

    def forward(self, x):
        """
        x: (B, C, H, W), where C=dim
        returns: (B, C, H, W)
        """
        B, C, H, W = x.shape
        # Save input as residual
        shortcut = x

        # ===== Self-Attention Sub-layer =====
        # 1. LayerNorm on channels: convert (B, C, H, W) -> (B, H, W, C) to apply nn.LayerNorm(dim)
        x_ln = self.norm1(x.permute(0, 2, 3, 1))  # (B, H, W, C)
        # 2. If shifting windows, roll the feature map
        if self.shift_size > 0:
            x_ln = torch.roll(
                x_ln, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )  # shift along H, W

        # 3. Partition into windows of size window_size x window_size
        num_win_h = H // self.window_size
        num_win_w = W // self.window_size

        # reshape x_ln -> (B, num_win_h, window_size, num_win_w, window_size, C)
        x_windows = x_ln.view(
            B,
            num_win_h,
            self.window_size,
            num_win_w,
            self.window_size,
            C
        )
        # permute to (B, num_win_h, num_win_w, window_size, window_size, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        # merge windows: (B * num_win_h * num_win_w, window_size*window_size, C)
        x_windows = x_windows.view(
            -1,
            self.window_size * self.window_size,
            C
        )  # call total number of windows = B * nW

        # 4. Compute QKV for all windows at once
        # x_windows shape: (B*nW, N, C), where N = window_size*window_size
        qkv = self.qkv(x_windows)  # (B*nW, N, 3*dim)
        # reshape to (B*nW, N, 3, num_heads, head_dim)
        head_dim = C // self.num_heads
        qkv = qkv.view(-1, qkv.shape[1], 3, self.num_heads, head_dim)
        # permute to (3, B*nW, num_heads, N, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Each of q, k, v now has shape (B*nW, num_heads, N, head_dim)

        # 5. Compute scaled dot-product attention per window
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B*nW, num_heads, N, N)

        # If using shifted windows, we must mask out tokens that cross window boundaries
        if self.shift_size > 0:
            # Build an attention mask for H x W to separate each window's ID
            img_mask = x.new_zeros((1, H, W, 1))  # (1, H, W, 1)
            cnt = 0
            for h in range(0, H, self.window_size):
                for w in range(0, W, self.window_size):
                    img_mask[:, h : h + self.window_size, w : w + self.window_size, :] = cnt
                    cnt += 1
            # roll the mask same as x_ln was rolled
            mask_shifted = torch.roll(
                img_mask, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            # Partition and flatten mask: (1, num_win_h, window_size, num_win_w, window_size, 1)
            mask_windows = mask_shifted.view(
                1,
                num_win_h,
                self.window_size,
                num_win_w,
                self.window_size,
                1
            )
            mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5).contiguous()
            # reshape -> (num_windows, N), where N = window_size*window_size
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            # build boolean mask: (num_windows, N, N): True if two tokens come from different window IDs
            attn_mask = mask_windows.unsqueeze(1) != mask_windows.unsqueeze(2)
            # Expand mask to match batch * num_windows:
            attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1, -1)  # (B, nW, N, N)
            attn_mask = attn_mask.reshape(-1, attn_mask.shape[2], attn_mask.shape[3])  # (B*nW, N, N)
            # Now expand to (B*nW, 1, N, N) so it can broadcast over heads
            attn_mask = attn_mask.unsqueeze(1)
            # Mask out invalid attention scores by setting to large negative
            attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))

        # Softmax normalization
        attn_probs = F.softmax(attn_scores, dim=-1)  # (B*nW, num_heads, N, N)
        # Attend to V
        attn_out = torch.matmul(attn_probs, v)  # (B*nW, num_heads, N, head_dim)

        # ===== Combine heads and reconstruct image shape =====
        # Current shape: (B*nW, num_heads, N, head_dim)
        # Permute to (B*nW, N, num_heads, head_dim)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous()  # <--- fix: used index 3, not 4
        # Merge num_heads and head_dim -> dim
        attn_out = attn_out.view(
            -1, self.window_size * self.window_size, C
        )  # (B*nW, N, dim)

        # Project back to C channels
        attn_out = self.proj(attn_out)  # (B*nW, N, dim)

        # Reshape windows back to (B, num_win_h, num_win_w, window_size, window_size, dim)
        attn_out = attn_out.view(
            B,
            num_win_h,
            num_win_w,
            self.window_size,
            self.window_size,
            C
        )
        # Permute to (B, num_win_h, window_size, num_win_w, window_size, C)
        attn_out = attn_out.permute(0, 1, 3, 2, 4, 5).contiguous()
        # Finally flatten to (B, H, W, C)
        attn_out = attn_out.view(B, H, W, C)

        # Reverse the earlier shift
        if self.shift_size > 0:
            attn_out = torch.roll(attn_out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        # Add residual from the start of this sub-layer (after projecting back to shape)
        x_attn = shortcut.permute(0, 2, 3, 1) + attn_out  # (B, H, W, C)

        # ===== MLP Sub-layer =====
        x_ff = self.norm2(x_attn)  # (B, H, W, C)
        x_ff = self.fc1(x_ff)      # (B, H, W, hidden_dim)
        x_ff = F.gelu(x_ff)
        x_ff = self.fc2(x_ff)      # (B, H, W, C)

        # Add residual again
        x_out = x_attn + x_ff      # (B, H, W, C)

        # Permute back to (B, C, H, W)
        x_out = x_out.permute(0, 3, 1, 2).contiguous()

        return x_out


class ResidualSwinTransformerBlock(nn.Module):
    """
    Residual Swin Transformer block (RST) = two SwinTransformerBlocks (one normal, one shifted)
    + a 3x3 convolution + a residual skip connection over all of them.
    Input & output shape: (B, dim, 40, 40).
    """
    def __init__(self, dim, num_heads, window_size, mlp_ratio=4.0):
        super().__init__()
        # 1st Swin block: no shift
        self.swin1 = SwinTransformerBlock(dim, num_heads, window_size, shift=False, mlp_ratio=mlp_ratio)
        # 2nd Swin block: with shift
        self.swin2 = SwinTransformerBlock(dim, num_heads, window_size, shift=True, mlp_ratio=mlp_ratio)
        # Convolution after the two transformer layers
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        # Input shape: (B, dim, 40, 40)
        residual = x
        x = self.swin1(x)
        x = self.swin2(x)
        x = self.conv(x)           # (B, dim, 40, 40)
        x = x + residual           # residual skip over the entire RST block
        return x


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
