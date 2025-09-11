# resnet_transformer.py  — Spatial ResNet + Temporal PatchTST encoder
import os
from typing import Optional, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional torchvision import for ResNet backbone
try:
    from torchvision.models import resnet50, ResNet50_Weights
except Exception:
    resnet50, ResNet50_Weights = None, None

# Optional transformers import for PatchTST
try:
    from transformers import PatchTSTModel, PatchTSTConfig, PatchTSTForPretraining
    _HAS_PATCHTST = True
except Exception:
    PatchTSTModel, PatchTSTConfig, _HAS_PATCHTST = None, None, False


# =========================
# Small utilities
# =========================
def get_base_model(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, nn.DataParallel) else m

def _set_requires_grad_(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


# =========================
# Static Encoders (multi-scale)
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
    def forward(self, x): return self.act(self.norm(self.conv(x)))

class MultiScaleStaticEncoder(nn.Module):
    def __init__(self, d_static: int = 64):
        super().__init__()
        self.d_static = d_static
        self.dem_encoder = nn.Sequential(
            ConvBlock(1, 16, 7, 4, 3), ConvBlock(16, 32, 5, 4, 2), ConvBlock(32, 64, 3, 2, 1),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(64*4, d_static), nn.LayerNorm(d_static)
        )
        self.awc_fc_encoder = nn.Sequential(
            ConvBlock(1, 32, 5, 2, 2), ConvBlock(32, 64, 3, 2, 1),
            nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten(), nn.Linear(64*9, d_static), nn.LayerNorm(d_static)
        )
        self.soil_encoder = nn.Sequential(
            ConvBlock(3, 32, 3, 1, 1), ConvBlock(32, 64, 3, 1, 1),
            nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten(), nn.Linear(64*9, d_static), nn.LayerNorm(d_static)
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_static, d_static*2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_static*2, d_static), nn.LayerNorm(d_static)
        )
    def forward(self, DEM=None, awc=None, fc=None, soil=None):
        feats: List[torch.Tensor] = []
        if DEM is not None: feats.append(self.dem_encoder(DEM))
        if awc is not None: feats.append(self.awc_fc_encoder(awc))
        if fc  is not None: feats.append(self.awc_fc_encoder(fc))
        if soil is not None: feats.append(self.soil_encoder(soil))
        if not feats: return None, False
        fused = torch.stack(feats, dim=0).mean(0)
        fused = self.fusion(fused)
        return fused, True


# =========================
# Spatial Encoder: ResNet-50 backbone
# =========================
class ResNetSpatialEncoder(nn.Module):
    def __init__(self, in_channels: int, d_model: int = 256, pretrained: bool = True, freeze_stages: int = 0):
        super().__init__()
        if resnet50 is None:
            raise ImportError("torchvision is required for ResNetSpatialEncoder. Please install torchvision.")
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        if in_channels != 3:
            orig = self.backbone.conv1
            newc = nn.Conv2d(in_channels, orig.out_channels,
                             kernel_size=orig.kernel_size, stride=orig.stride,
                             padding=orig.padding, bias=False)
            with torch.no_grad():
                if pretrained:
                    mean_w = orig.weight.mean(dim=1, keepdim=True)
                    newc.weight[:] = mean_w.repeat(1, in_channels, 1, 1)
                else:
                    nn.init.kaiming_normal_(newc.weight, mode="fan_out", nonlinearity="relu")
            self.backbone.conv1 = newc
        self.backbone.fc = nn.Identity()
        self.proj = nn.Sequential(nn.Linear(2048, d_model), nn.LayerNorm(d_model))
        self.freeze_up_to(freeze_stages)

    def freeze_up_to(self, stages: int = 0):
        if stages >= 1:
            _set_requires_grad_(self.backbone.conv1, False)
            _set_requires_grad_(self.backbone.bn1, False)
        if stages >= 2: _set_requires_grad_(self.backbone.layer1, False)
        if stages >= 3: _set_requires_grad_(self.backbone.layer2, False)
        if stages >= 4: _set_requires_grad_(self.backbone.layer3, False)
        if stages >= 5: _set_requires_grad_(self.backbone.layer4, False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x); x = self.backbone.bn1(x); x = self.backbone.relu(x); x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x); x = self.backbone.layer2(x); x = self.backbone.layer3(x); x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x); x = torch.flatten(x, 1)
        return self.proj(x)  # (B, d_model)


# =========================
# Temporal Encoder: PatchTST (Transformers)
# =========================
class PatchTSTTemporalEncoder(nn.Module):
    """
    Wraps Hugging Face PatchTST as a temporal encoder that outputs a single vector per sequence.
    - Loads from HF repo; supports 'ForPretraining' checkpoints (IBM ETTh1) by mapping to PatchTSTModel.
    - Projects spatial feature dim -> hub num_input_channels (e.g., 256 -> 7) for compatibility.
    - Optional out_proj to any requested temporal_d_model.
    """
    def __init__(
        self,
        d_in: int,                               # spatial feature dim coming from the spatial encoder
        repo_id: str = "ibm-research/patchtst-etth1-pretrain",
        pool: str = "mean",                      # "mean" | "last"
        out_dim: int | None = None,              # final temporal dim; if None, use checkpoint d_model
        map_location: str | torch.device = "cpu",
        verbose: bool = True,
    ):
        super().__init__()
        self.pretrained_loaded = False
        self.pool = pool

        # Load config from repo (uses model_type="patchtst", context_length=512, num_input_channels=7, etc.)
        # NOTE: This repo’s config declares architectures=["PatchTSTForMaskPretraining"].
        # We still can construct a bare PatchTSTModel with this config.  :contentReference[oaicite:2]{index=2}
        self.config = PatchTSTConfig.from_pretrained(repo_id)

        # Try to get a bare PatchTSTModel with weights; otherwise load pretraining head and map.
        model = None
        try:
            model = PatchTSTModel.from_pretrained(repo_id, torch_dtype=torch.float32)
            self.pretrained_loaded = True
            if verbose: print(f"[PatchTST] Loaded PatchTSTModel from '{repo_id}'.")
        except Exception as e:
            if verbose: print(f"[PatchTST] PatchTSTModel.from_pretrained failed: {e}\n"
                              f"Trying PatchTSTForPretraining → PatchTSTModel...")
            try:
                pre = PatchTSTForPretraining.from_pretrained(repo_id, torch_dtype=torch.float32)
                model = PatchTSTModel(self.config)                       # same config
                missing, unexpected = model.load_state_dict(pre.state_dict(), strict=False)
                if verbose:
                    print(f"[PatchTST] Mapped pretraining weights → bare model. "
                          f"missing={len(missing)} unexpected={len(unexpected)}")
                self.pretrained_loaded = True
            except Exception as e2:
                if verbose: print(f"[PatchTST] Fallback load failed: {e2}\nUsing randomly initialized PatchTSTModel.")
                model = PatchTSTModel(self.config)  # random init

        self.model = model
        # projector from spatial feature dim -> hub num_input_channels (7 for ETTh1 repo)
        self.ch_in = int(self.config.num_input_channels)
        self.ch_proj = nn.Linear(d_in, self.ch_in) if d_in != self.ch_in else nn.Identity()

        # optional projection to requested out_dim
        self.hidden = int(self.config.d_model)
        final_dim = out_dim if (out_dim is not None) else self.hidden
        self.out_proj = nn.Linear(self.hidden, final_dim) if final_dim != self.hidden else nn.Identity()
        self.final_dim = final_dim

        self.dropout = nn.Dropout(getattr(self.config, "head_dropout", 0.0))

    # ---- freezing helpers (match your training script API) ----
    def _set_requires_grad_(self, module: nn.Module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag

    def freeze_all(self, keep_proj: bool = True, keep_norm: bool = True):
        """Freeze the HF encoder; optionally keep our channel/out projections and norms trainable."""
        self._set_requires_grad_(self.model, False)
        if keep_proj:
            self._set_requires_grad_(self.ch_proj, True)
            self._set_requires_grad_(self.out_proj, True)
        if keep_norm:
            for m in self.model.modules():
                if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                    self._set_requires_grad_(m, True)

    def unfreeze_last_n(self, n: int):
        """Unfreeze last n Transformer layers from the HF encoder."""
        if not hasattr(self.model, "encoder") or not hasattr(self.model.encoder, "layers"):
            return
        layers = list(self.model.encoder.layers)
        for layer in layers[-int(n):]:
            self._set_requires_grad_(layer, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_in) → channel-project → HF PatchTST (expects (B, L, num_input_channels)).
        We pool tokens to a single vector per sample, then out_proj to final dim.
        """
        B, L, D = x.shape
        x = self.ch_proj(x)                      # (B, L, ch_in)
        outputs = self.model(past_values=x)
        h = outputs.last_hidden_state            # shape depends on config; flatten tokens

        # h could be (B, num_channels, num_patches, hidden) or (B, tokens, hidden) depending on config.
        if h.dim() == 4:
            B, C, P, H = h.shape
            h = h.reshape(B, C * P, H)          # (B, tokens, hidden)

        if self.pool == "last":
            h = h[:, -1, :]                     # (B, hidden)
        else:
            h = h.mean(dim=1)                   # (B, hidden)

        h = self.out_proj(h)                    # (B, final_dim)
        return self.dropout(h)

# =========================
# Fusion blocks
# =========================
class CrossAttentionFusion(nn.Module):
    def __init__(self, d_dynamic: int, d_static: int, d_out: int, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_dynamic, num_heads, batch_first=True)
        self.static_proj = nn.Linear(d_static, d_dynamic)
        self.out_proj = nn.Linear(d_dynamic, d_out)
        self.norm1 = nn.LayerNorm(d_dynamic)
        self.norm2 = nn.LayerNorm(d_out)
    def forward(self, dynamic_feat, static_feat):
        s = self.static_proj(static_feat).unsqueeze(1)
        q = dynamic_feat.unsqueeze(1)
        attn, _ = self.mha(q, s, s)
        fused = self.norm1(q + attn).squeeze(1)
        return self.norm2(self.out_proj(fused))


# =========================
# Main Model
# =========================
class ImprovedHydroTransformer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 spatial_d_model: int = 256,
                 spatial_pretrained: bool = True,
                 spatial_freeze_stages: int = 0,
                 # temporal
                 temporal_repo: str = "ibm-research/patchtst-etth1-pretrain",
                 temporal_out_dim: int = 512,           # deeper head dim for your downstream head
                 temporal_pool: str = "mean",
                 # static
                 static_d_model: int = 64,
                 # fusion
                 fusion_type: str = "film",
                 # head
                 output_dim: int = 1,
                 head_dropout: float = 0.1):
        super().__init__()

        self.spatial_encoder = ResNetSpatialEncoder(
            in_channels=in_channels,
            d_model=spatial_d_model,
            pretrained=spatial_pretrained,
            freeze_stages=spatial_freeze_stages,
        )

        # >>> PatchTST as temporal encoder
        self.temporal_encoder = PatchTSTTemporalEncoder(
            d_in=spatial_d_model,
            repo_id=temporal_repo,
            pool=temporal_pool,
            out_dim=temporal_out_dim
        )
        t_dim = temporal_out_dim

        self.static_encoder = MultiScaleStaticEncoder(static_d_model)

        self.fusion_type = fusion_type
        if fusion_type == "film":
            self.film_gamma = nn.Linear(static_d_model, t_dim)
            self.film_beta  = nn.Linear(static_d_model, t_dim)
            final_dim = t_dim
        elif fusion_type == "concat":
            self.concat = nn.Sequential(
                nn.Linear(t_dim + static_d_model, t_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.LayerNorm(t_dim),
            )
            final_dim = t_dim
        else:
            self.cross = CrossAttentionFusion(t_dim, static_d_model, t_dim, num_heads=4)
            final_dim = t_dim

        self.output_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(final_dim, output_dim),
        )

    # === keep your optimizer grouping API ===
    def param_groups(self, lr_dyn_trunk: float, lr_stat_trunk: float, lr_other: float, weight_decay: float):
        dyn = [p for p in self.spatial_encoder.parameters() if p.requires_grad]
        stat = [p for p in self.static_encoder.parameters() if p.requires_grad]
        other = []
        other += list(self.temporal_encoder.parameters())
        if self.fusion_type == "film":
            other += list(self.film_gamma.parameters()) + list(self.film_beta.parameters())
        elif self.fusion_type == "concat":
            other += list(self.concat.parameters())
        else:
            other += list(self.cross.parameters())
        other += list(self.output_head.parameters())
        other = [p for p in other if p.requires_grad]
        return [
            {"params": dyn,   "lr": lr_dyn_trunk,  "weight_decay": weight_decay},
            {"params": stat,  "lr": lr_stat_trunk, "weight_decay": weight_decay},
            {"params": other, "lr": lr_other,      "weight_decay": weight_decay},
        ]

    # === wrappers your trainer expects ===
    def freeze_temporal_all(self, keep_proj: bool = True, keep_norm: bool = True):
        self.temporal_encoder.freeze_all(keep_proj=keep_proj, keep_norm=keep_norm)

    def unfreeze_temporal_last_n(self, n: int):
        self.temporal_encoder.unfreeze_last_n(n)

    def forward(self, X, DEM=None, awc=None, fc=None, soil=None):
        B, L, C, H, W = X.shape
        x2d = X.reshape(B * L, C, H, W)
        sfeat = self.spatial_encoder(x2d).reshape(B, L, -1)
        tfeat = self.temporal_encoder(sfeat)

        sstat, has_static = self.static_encoder(DEM=DEM, awc=awc, fc=fc, soil=soil)
        fused = tfeat
        if has_static:
            if sstat.shape[0] == 1 and B > 1:
                sstat = sstat.expand(B, -1)
            if self.fusion_type == "film":
                fused = tfeat * (1 + self.film_gamma(sstat)) + self.film_beta(sstat)
            elif self.fusion_type == "concat":
                fused = self.concat(torch.cat([tfeat, sstat], dim=-1))
            else:
                fused = self.cross(tfeat, sstat)

        out = self.output_head(fused)
        return out.squeeze(-1) if out.shape[-1] == 1 else out
