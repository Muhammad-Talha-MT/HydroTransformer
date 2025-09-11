import math
from typing import Optional
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 
import torch.autograd.profiler as profiler
# ---------------------------
# Sinusoidal positional encoding
# ---------------------------
def sinusoidal_position_encoding(L: int, d_model: int, device: torch.device) -> torch.Tensor:
    """(L, d_model) sinusoidal PE; parameter-free and works for variable L."""
    pe = torch.zeros(L, d_model, device=device)
    position = torch.arange(0, L, device=device, dtype=torch.float32).unsqueeze(1)  # (L,1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float32) *
                         (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (L, d)


# ---------------------------
# Spatial encoder (handles any HxW via adaptive pooling)
# ---------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class SpatialEncoder(nn.Module):
    """
    (B*L, C[, +2 coords], H, W) -> (B*L, d_model)
    """
    def __init__(self, in_channels: int, d_model: int = 256, add_coords: bool = False, depth: int = 3):
        super().__init__()
        self.add_coords = add_coords
        c_in = in_channels + (2 if add_coords else 0)

        layers = []
        c = c_in
        for i in range(depth):
            c_out = d_model if i == depth - 1 else max(64, min(d_model // 2, c * 2))
            layers.append(DepthwiseSeparableConv(c, c_out, k=3, s=1, p=1))
            c = c_out
        self.convs = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)

    @staticmethod
    def _make_coords(H: int, W: int, device: torch.device) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, steps=H, device=device)
        xs = torch.linspace(-1.0, 1.0, steps=W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([yy, xx], dim=0)  # (2,H,W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*L, C, H, W)
        BxL, C, H, W = x.shape
        if self.add_coords:
            coords = self._make_coords(H, W, x.device).unsqueeze(0).expand(BxL, -1, -1, -1)
            x = torch.cat([x, coords], dim=1)
        x = self.convs(x)
        x = self.pool(x).squeeze(-1)  # (B*L, d_model)
        return x

# ---------------------------
# Temporal Transformer encoder
# ---------------------------
class TemporalTransformer(nn.Module):
    """
    (B, L, d_model) -> (B, L, d_model)
    """
    def __init__(self, d_model=256, n_heads=8, depth=4, dropout=0.1, norm_first=True):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=norm_first
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x, src_key_padding_mask: Optional[torch.Tensor] = None):
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


# ---------------------------
# Full model: HydroTransformer
# ---------------------------
class HydroTransformer(nn.Module):
    """
    Inputs:
        X: (B, L, C, H, W) climate tensor
    Output:
        y_hat: (B,) — lead-1 streamflow regression (use y[:,0] from your dataset)
    """
    def __init__(self,
                 in_channels: int,
                 d_model: int = 256,
                 spatial_depth: int = 3,
                 temporal_layers: int = 4,
                 n_heads: int = 8,
                 add_coords: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.spatial = SpatialEncoder(in_channels, d_model=d_model, add_coords=add_coords, depth=spatial_depth)
        self.temporal = TemporalTransformer(d_model=d_model, n_heads=n_heads,
                                            depth=temporal_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)  # lead-1 scalar

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     B, L, C, H, W = x.shape

    #     # Profiling spatial encoder
    #     with profiler.profile(use_device='cuda') as prof:
    #         x = x.reshape(B * L, C, H, W)
    #         feats = self.spatial(x)  # (B*L, d)
    #         feats = feats.reshape(B, L, self.d_model)
    #     print("SpatialEncoder profiling:")
    #     print(prof.key_averages().table(sort_by="cpu_time_total"))

    #     # Profiling transformer
    #     with profiler.profile(use_device='cuda') as prof:
    #         pe = sinusoidal_position_encoding(L, self.d_model, feats.device)  # (L,d)
    #         feats = feats + pe.unsqueeze(0)
    #         hidden = self.temporal(self.dropout(feats))  # (B,L,d)
    #     print("Transformer profiling:")
    #     print(prof.key_averages().table(sort_by="cpu_time_total"))

    #     # Profiling head (final prediction)
    #     with profiler.profile(use_device='cuda') as prof:
    #         last = self.norm(hidden[:, -1, :])  # (B,d)
    #         y_hat = self.head(self.dropout(last))  # (B,1)
    #     print("Head profiling:")
    #     print(prof.key_averages().table(sort_by="cpu_time_total"))

    #     return y_hat.squeeze(-1)  # (B,)             # (B,)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C, H, W = x.shape

        # Profiling spatial encoder
        x = x.reshape(B * L, C, H, W)
        feats = self.spatial(x)  # (B*L, d)
        feats = feats.reshape(B, L, self.d_model)

        # Profiling transformer
        pe = sinusoidal_position_encoding(L, self.d_model, feats.device)  # (L,d)
        feats = feats + pe.unsqueeze(0)
        hidden = self.temporal(self.dropout(feats))  # (B,L,d)

        # Profiling head (final prediction)
        last = self.norm(hidden[:, -1, :])  # (B,d)
        y_hat = self.head(self.dropout(last))  # (B,1)

        return y_hat.squeeze(-1)  # (B,)             # (B,)
