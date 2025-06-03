import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Swin Transformer Block Components (simplified for clarity) ---
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 4)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class RSTBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.swin1 = SwinTransformerBlock(dim, num_heads)
        self.swin2 = SwinTransformerBlock(dim, num_heads)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # B, HW, C
        x = self.swin1(x_flat)
        x = self.swin2(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        return self.conv(x) + x

class STRWP(nn.Module):
    def __init__(self, K=6, in_channels=3, hidden_dim=60, num_heads=10, rst_blocks=4):
        super().__init__()
        self.conv1 = nn.Conv2d(K * in_channels, hidden_dim, kernel_size=3, padding=1)
        self.rst_blocks = nn.Sequential(*[RSTBlock(hidden_dim, num_heads) for _ in range(rst_blocks)])
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(hidden_dim, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B, K, C=3, H, W)
        B, K, C, H, W = x.shape
        x = x.view(B, K * C, H, W)
        low_freq = self.conv1(x)
        x = self.rst_blocks(low_freq)
        x = self.conv2(x) + low_freq
        out = self.conv_out(x)
        return out  # shape: (B, 3, H, W)

# --- Rolled-out Prediction ---
def rolled_out_prediction(model, input_seq, steps=6):
    """
    model: trained ST-RWP model
    input_seq: torch.Tensor of shape (B, K, 3, H, W)
    steps: number of rollout steps (e.g., 6 for 6h prediction)
    """
    model.eval()
    preds = []
    current_input = input_seq.clone()
    with torch.no_grad():
        for _ in range(steps):
            pred = model(current_input)  # shape: (B, 3, H, W)
            preds.append(pred)
            # Prepare next input
            current_input = torch.cat([current_input[:, 1:], pred.unsqueeze(1)], dim=1)
    return torch.stack(preds, dim=1)  # (B, steps, 3, H, W)
