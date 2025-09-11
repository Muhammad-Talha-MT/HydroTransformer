# resnet_transformer.py
import os
from typing import Optional, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# Optional torchvision import for ResNet backbone
try:
    from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
except Exception:
    resnet18 = resnet50 = ResNet18_Weights = ResNet50_Weights = None



# =========================
# Small utilities
# =========================
def get_base_model(m: nn.Module) -> nn.Module:
    """Unwrap DataParallel for attribute access."""
    return m.module if isinstance(m, nn.DataParallel) else m


def _set_requires_grad_(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


# =========================
# Static Encoders (multi-scale)  [light-weight CNNs]
# =========================
class ConvBlock(nn.Module):
    """Efficient conv block with normalization and activation."""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class MultiScaleStaticEncoder(nn.Module):
    """
    Separate lightweight encoders for DEM (1ch), AWC (1ch), FC (1ch), soil (3ch),
    then fuse by mean + MLP. Inputs are (B,C,H,W). Output: (B, d_static)
    """
    def __init__(self, d_static: int = 64):
        super().__init__()
        self.d_static = d_static

        self.dem_encoder = nn.Sequential(
            ConvBlock(1, 16, 7, 4, 3),  # /4
            ConvBlock(16, 32, 5, 4, 2), # /16
            ConvBlock(32, 64, 3, 2, 1), # /32
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(64 * 4, d_static),
            nn.LayerNorm(d_static)
        )
        self.awc_fc_encoder = nn.Sequential(
            ConvBlock(1, 32, 5, 2, 2),  # /2
            ConvBlock(32, 64, 3, 2, 1), # /4
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(64 * 9, d_static),
            nn.LayerNorm(d_static)
        )
        self.soil_encoder = nn.Sequential(
            ConvBlock(3, 32, 3, 1, 1),
            ConvBlock(32, 64, 3, 1, 1),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(64 * 9, d_static),
            nn.LayerNorm(d_static)
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_static, d_static * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_static * 2, d_static),
            nn.LayerNorm(d_static)
        )

    def forward(self, DEM=None, awc=None, fc=None, soil=None):
        feats: List[torch.Tensor] = []
        if DEM is not None:
            feats.append(self.dem_encoder(DEM))      # (B, d_static)
        if awc is not None:
            feats.append(self.awc_fc_encoder(awc))   # (B, d_static)
        if fc is not None:
            feats.append(self.awc_fc_encoder(fc))    # (B, d_static)
        if soil is not None:
            feats.append(self.soil_encoder(soil))    # (B, d_static)

        if len(feats) == 0:
            return None, False

        fused = torch.stack(feats, dim=0).mean(0)  # (B, d_static)
        fused = self.fusion(fused)                 # (B, d_static)
        return fused, True


# =========================
# Spatial Encoder: ResNet-50 backbone
# =========================
# at top (import both resnets + weights)

class ResNetSpatialEncoder(nn.Module):
    """
    ResNet-18/50 backbone (pretrained) with flexible input channels and projection to d_model.
    Input:  (B*L, C, H, W)  ->  Output: (B*L, d_model)
    """
    def __init__(
        self,
        in_channels: int,
        d_model: int = 256,
        pretrained: bool = True,
        freeze_stages: int = 0,       # 0..5 (0=none, 5=freeze stem + layer1..4)
        arch: str = "resnet18",       # "resnet18" | "resnet50"
    ):
        super().__init__()
        arch = arch.lower()
        if arch == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = resnet18(weights=weights)
            print("Using ResNet-18 backbone")
        elif arch == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = resnet50(weights=weights)
            print("Using ResNet-50 backbone")
        else:
            raise ValueError(f"Unsupported arch: {arch}")

        # Adapt first conv for non-3ch inputs
        if in_channels != 3:
            orig_conv = self.backbone.conv1
            new_conv = nn.Conv2d(in_channels, orig_conv.out_channels,
                                 kernel_size=orig_conv.kernel_size,
                                 stride=orig_conv.stride,
                                 padding=orig_conv.padding, bias=False)
            with torch.no_grad():
                if pretrained:
                    mean_w = orig_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight[:] = mean_w.repeat(1, in_channels, 1, 1)
                else:
                    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            self.backbone.conv1 = new_conv

        # Drop classifier, read feature dim (512 for r18, 2048 for r50)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Projection to spatial_d_model
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # ← this was missing in your runtime class
        self.freeze_up_to(freeze_stages)

    def freeze_up_to(self, stages: int = 0):
        """Freeze up to N stages: 0 none; 1 stem; 2 +layer1; 3 +layer2; 4 +layer3; 5 +layer4."""
        if stages <= 0:
            return
        # stem
        if stages >= 1:
            _set_requires_grad_(self.backbone.conv1, False)
            _set_requires_grad_(self.backbone.bn1,  False)
        # layer1..4
        if stages >= 2: _set_requires_grad_(self.backbone.layer1, False)
        if stages >= 3: _set_requires_grad_(self.backbone.layer2, False)
        if stages >= 4: _set_requires_grad_(self.backbone.layer3, False)
        if stages >= 5: _set_requires_grad_(self.backbone.layer4, False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x); x = self.backbone.bn1(x); x = self.backbone.relu(x); x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x); x = self.backbone.layer2(x); x = self.backbone.layer3(x); x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)     # (B, feat_dim, 1, 1)
        x = torch.flatten(x, 1)          # (B, feat_dim)
        return self.proj(x)              # (B, d_model)

# drop-in for your temporal encoder
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)              # (max_len, d_model)
    def forward(self, x):                           # x: (B,L,d)
        return x + self.pe[:x.size(1)].unsqueeze(0)
    
    
# =========================
# Temporal Encoder (pretrained-friendly, deeper)
# =========================
class SimpleTransformerEncoderPretrained(nn.Module):
    """
    Vanilla TransformerEncoder with:
      - input projection to d_model
      - N encoder layers (norm_first=True)
      - final LayerNorm + Dropout
      - helpers to load a pretrained state_dict (local or HF TimeSeriesTransformer)
      - optional [CLS] token usage (defaults to last-token pooling)

    Expects x: (B, L, d_in) and returns (B, d_model).
    """
    def __init__(
        self,
        d_in: int,
        d_model: int = 512,
        n_heads: int = 8,
        depth: int = 12,
        ff_mult: float = 4.0,
        dropout: float = 0.1,
        norm_first: bool = True,
        use_cls_token: bool = False,
        checkpoint_path: Optional[str] = None,
        map_location: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.depth = depth
        self.use_cls_token = use_cls_token
        self.pretrained_loaded = False
        self.pos_enc = PositionalEncoding(d_model)
        self.proj_in = nn.Linear(d_in, d_model)
        dim_ff = max(32, int(d_model * ff_mult))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        # ----- Try to load weights -----
        if checkpoint_path:
            sd = None
            # Case 1: local file
            if os.path.isfile(str(checkpoint_path)):
                try:
                    sd = torch.load(checkpoint_path, map_location=map_location)
                    if isinstance(sd, dict) and "state_dict" in sd:
                        sd = sd["state_dict"]
                except Exception as e:
                    print(f"[Temporal] Local checkpoint load failed: {e}")

            # Case 2: HF repo id -> raw state dict
            if sd is None and ("/" in str(checkpoint_path)) and (not os.path.exists(str(checkpoint_path))):
                try:
                    from huggingface_hub import hf_hub_download
                    candidates = ("model.safetensors", "pytorch_model.bin", "pytorch_model.pt")
                    for fname in candidates:
                        try:
                            local = hf_hub_download(repo_id=str(checkpoint_path), filename=fname)
                            if fname.endswith(".safetensors"):
                                from safetensors.torch import load_file
                                sd = load_file(local)
                            else:
                                sd = torch.load(local, map_location=map_location)
                            print(f"[Temporal] Downloaded '{fname}' from HF repo '{checkpoint_path}'.")
                            break
                        except Exception:
                            continue
                except Exception as e:
                    print(f"[Temporal] HF download failed: {e}")

            # Try strict=False update for any matching keys
            if isinstance(sd, dict):
                cur = self.state_dict()
                matched = {k: v for k, v in sd.items()
                           if k in cur and hasattr(v, "shape") and cur[k].shape == v.shape}
                if matched:
                    cur.update(matched)
                    missing, unexpected = self.load_state_dict(cur, strict=False)
                    print(f"[Temporal] Loaded {len(matched)} keys; missing:{len(missing)} unexpected:{len(unexpected)}")
                    if len(matched) > 0:
                        self.pretrained_loaded = True

            # Case 3: explicit mapping from HF TimeSeriesTransformer encoder
            if (not self.pretrained_loaded) and ("/" in str(checkpoint_path)):
                ok = self._try_load_hf_tst_encoder(str(checkpoint_path), map_location=map_location)
                if ok:
                    self.pretrained_loaded = True
                    print(f"[Temporal] Loaded pretrained weights from HF TimeSeriesTransformer '{checkpoint_path}'.")
                else:
                    print("[Temporal] No compatible keys found in checkpoint (likely different architecture).")

    # ----- HF TimeSeriesTransformer → nn.TransformerEncoder mapper -----
    def _try_load_hf_tst_encoder(self, repo_id: str, map_location="cpu") -> bool:
        """Download Hugging Face TimeSeriesTransformer checkpoint and map its encoder weights."""
        try:
            import json
            from huggingface_hub import hf_hub_download
            # 1) Read config
            cfg_path = hf_hub_download(repo_id=repo_id, filename="config.json")
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            d_model_hf = int(cfg.get("d_model", self.d_model))
            n_heads_hf = int(cfg.get("encoder_attention_heads", getattr(self.encoder.layers[0].self_attn, "num_heads", -1)))
            depth_hf = int(cfg.get("encoder_layers", self.depth))
            ffn_dim_hf = int(cfg.get("encoder_ffn_dim", int(self.d_model * 4)))

            if (self.d_model != d_model_hf or
                n_heads_hf != self.encoder.layers[0].self_attn.num_heads or
                self.depth != depth_hf or
                self.encoder.layers[0].linear1.out_features != ffn_dim_hf):
                print(f"[Temporal] Size mismatch vs HF config.")
                return False

            # 2) Load HF weights
            pt_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
            sd_hf = torch.load(pt_path, map_location=map_location)
            if not isinstance(sd_hf, dict):
                print("[Temporal] HF state dict not a dict.")
                return False

            # 3) Copy weights into our state dict
            cur = self.state_dict()
            loaded_tensors = 0

            for i in range(min(self.depth, depth_hf)):
                base = f"model.encoder.layers.{i}"

                def get(k):
                    kk = f"{base}.{k}"
                    if kk not in sd_hf:
                        raise KeyError(kk)
                    return sd_hf[kk]

                # Self-attn QKV → in_proj
                q_w, q_b = get("self_attn.q_proj.weight"), get("self_attn.q_proj.bias")
                k_w, k_b = get("self_attn.k_proj.weight"), get("self_attn.k_proj.bias")
                v_w, v_b = get("self_attn.v_proj.weight"), get("self_attn.v_proj.bias")

                cur[f"encoder.layers.{i}.self_attn.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0)
                cur[f"encoder.layers.{i}.self_attn.in_proj_bias"]  = torch.cat([q_b, k_b, v_b], dim=0)

                # Self-attn out
                cur[f"encoder.layers.{i}.self_attn.out_proj.weight"] = get("self_attn.out_proj.weight")
                cur[f"encoder.layers.{i}.self_attn.out_proj.bias"]   = get("self_attn.out_proj.bias")

                # Feed-forward
                cur[f"encoder.layers.{i}.linear1.weight"] = get("fc1.weight")
                cur[f"encoder.layers.{i}.linear1.bias"]   = get("fc1.bias")
                cur[f"encoder.layers.{i}.linear2.weight"] = get("fc2.weight")
                cur[f"encoder.layers.{i}.linear2.bias"]   = get("fc2.bias")

                # Norms
                cur[f"encoder.layers.{i}.norm1.weight"] = get("self_attn_layer_norm.weight")
                cur[f"encoder.layers.{i}.norm1.bias"]   = get("self_attn_layer_norm.bias")
                cur[f"encoder.layers.{i}.norm2.weight"] = get("final_layer_norm.weight")
                cur[f"encoder.layers.{i}.norm2.bias"]   = get("final_layer_norm.bias")

                loaded_tensors += 12  # rough count per layer

            missing, unexpected = self.load_state_dict(cur, strict=False)
            print(f"[Temporal] HF TST load: copied={loaded_tensors} | missing={len(missing)} unexpected={len(unexpected)}")
            return loaded_tensors > 0

        except Exception as e:
            print(f"[Temporal] HF TST mapping failed: {e}")
            return False

    # ----- freezing helpers -----
    def freeze_all(self, keep_proj: bool = True, keep_norm: bool = True):
        """Freeze all encoder layers; optionally keep input projection & final norm trainable."""
        _set_requires_grad_(self.encoder, False)
        _set_requires_grad_(self.proj_in, False)
        _set_requires_grad_(self.norm, False)
        if keep_proj:
            _set_requires_grad_(self.proj_in, True)
        if keep_norm:
            _set_requires_grad_(self.norm, True)

    def unfreeze_last_n(self, n: int):
        """Unfreeze only the last n TransformerEncoder layers."""
        if n <= 0:
            return
        layers = list(self.encoder.layers)
        for layer in layers[-n:]:
            _set_requires_grad_(layer, True)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ----- forward -----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_in) -> (B, d_model)
        If use_cls_token=True, prepends a learned [CLS] and returns its representation.
        Else, returns the last-token representation.
        """
        B, L, _ = x.shape
        x = self.pos_enc(x)                        # (B, L, d_model)
        x = self.proj_in(x)                         # (B, L, d_model)

        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)  # (B,1,d_model)
            x = torch.cat([cls, x], dim=1)          # (B,L+1,d_model)

        x = self.encoder(x)                         # (B, L(+1), d_model)

        if self.use_cls_token:
            x = self.norm(x[:, 0, :])               # [CLS]
        else:
            x = x.mean(dim=1)            # last token

        return self.dropout(x)


# =========================
# Fusion blocks
# =========================
class CrossAttentionFusion(nn.Module):
    """Cross-attention between dynamic and static features."""
    def __init__(self, d_dynamic: int, d_static: int, d_out: int, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_dynamic, num_heads, batch_first=True)
        self.static_proj = nn.Linear(d_static, d_dynamic)
        self.out_proj = nn.Linear(d_dynamic, d_out)
        self.norm1 = nn.LayerNorm(d_dynamic)
        self.norm2 = nn.LayerNorm(d_out)

    def forward(self, dynamic_feat, static_feat):
        # project static and make seq len == 1 for both
        s = self.static_proj(static_feat).unsqueeze(1)  # (B,1,d_dyn)
        q = dynamic_feat.unsqueeze(1)                   # (B,1,d_dyn)
        attn, _ = self.mha(q, s, s)
        fused = self.norm1(q + attn).squeeze(1)         # (B,d_dyn)
        return self.norm2(self.out_proj(fused))


# =========================
# Main Model
# =========================
class ImprovedHydroTransformer(nn.Module):
    """
    End-to-end model:
      - Spatial ResNet-50 per timestep -> (B,L,Ds)
      - Temporal Transformer (deeper) -> (B,Dt)
      - Static encoders -> (B,Ds_static)
      - Fusion (FiLM default) -> (B,Dt)
      - Head -> (B,1)
    """
    def __init__(
        self,
        in_channels: int,                 # spatial inputs C (dynamic climate channels)
        # spatial
        spatial_d_model: int = 256,
        spatial_pretrained: bool = True,
        spatial_freeze_stages: int = 0,   # 0..5
        # temporal (deeper)
        temporal_d_model: int = 512,
        temporal_heads: int = 8,
        temporal_depth: int = 12,
        temporal_ff_mult: float = 4.0,
        temporal_dropout: float = 0.1,
        temporal_norm_first: bool = True,
        temporal_use_cls_token: bool = False,
        temporal_checkpoint_path: Optional[str] = None,  # torch .pth or HF repo id
        # static
        static_d_model: int = 64,
        # fusion
        fusion_type: str = "film",  # "film" | "concat" | "cross_attention"
        # head
        output_dim: int = 1,
        head_dropout: float = 0.1,
        # device map for pretraining
        map_location: Union[str, torch.device] = "cpu",
    ):
        super().__init__()

        # Spatial (B*L,C,H,W)->(B*L,Ds) via ResNet-50
        self.spatial_encoder = ResNetSpatialEncoder(
            in_channels=in_channels,
            d_model=spatial_d_model,
            pretrained=spatial_pretrained,
            freeze_stages=spatial_freeze_stages,
        )

        # Temporal: encoder-only; input dim == spatial_d_model
        self.temporal_encoder = SimpleTransformerEncoderPretrained(
            d_in=spatial_d_model,
            d_model=temporal_d_model,
            n_heads=temporal_heads,
            depth=temporal_depth,
            ff_mult=temporal_ff_mult,
            dropout=temporal_dropout,
            norm_first=temporal_norm_first,
            use_cls_token=temporal_use_cls_token,
            checkpoint_path=temporal_checkpoint_path,
            map_location=map_location,
        )
        t_dim = temporal_d_model

        # Statics
        self.static_encoder = MultiScaleStaticEncoder(static_d_model)

        # Fusion
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
        elif fusion_type == "cross_attention":
            self.cross = CrossAttentionFusion(t_dim, static_d_model, t_dim, num_heads=4)
            final_dim = t_dim
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        # Head
        self.output_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(final_dim, output_dim),
            # nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            # Don't re-init Conv2d (keep pretrained weights)

    # ---- freezing helpers (temporal) ----
    def freeze_temporal_all(self, keep_proj: bool = True, keep_norm: bool = True):
        self.temporal_encoder.freeze_all(keep_proj=keep_proj, keep_norm=keep_norm)

    def unfreeze_temporal_last_n(self, n: int):
        self.temporal_encoder.unfreeze_last_n(n)

    # ---- parameter accounting ----
    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---- optimizer param groups (for training scripts) ----
    def param_groups(self, lr_dyn_trunk: float, lr_stat_trunk: float, lr_other: float, weight_decay: float):
        """
        Return optimizer param groups matching training script call:
        AdamW(base.param_groups(lr_dyn_trunk=..., lr_stat_trunk=..., lr_other=..., weight_decay=...))
        """
        groups = []

        # Spatial trunk (ResNet-50 backbone + its projection)
        dyn_params = [p for p in self.spatial_encoder.parameters() if p.requires_grad]
        groups.append({"params": dyn_params, "lr": lr_dyn_trunk, "weight_decay": weight_decay})

        # Static encoders
        stat_params = [p for p in self.static_encoder.parameters() if p.requires_grad]
        groups.append({"params": stat_params, "lr": lr_stat_trunk, "weight_decay": weight_decay})

        # Everything else: temporal encoder, fusion block, prediction head
        other = []
        other += list(self.temporal_encoder.parameters())
        if self.fusion_type == "film":
            other += list(self.film_gamma.parameters()) + list(self.film_beta.parameters())
        elif self.fusion_type == "concat":
            other += list(self.concat.parameters())
        elif self.fusion_type == "cross_attention":
            other += list(self.cross.parameters())
        other += list(self.output_head.parameters())
        other = [p for p in other if p.requires_grad]

        groups.append({"params": other, "lr": lr_other, "weight_decay": weight_decay})
        return groups

    # ---- forward ----
    def forward(self, X, DEM=None, awc=None, fc=None, soil=None):
        """
        X:   (B, L, C, H, W)
        DEM: (B,1,H,W) or None
        awc: (B,1,H,W) or None
        fc:  (B,1,H,W) or None
        soil:(B,3,H,W) or None
        """
        B, L, C, H, W = X.shape

        # Spatial per timestep
        x2d = X.reshape(B * L, C, H, W)
        sfeat = self.spatial_encoder(x2d).reshape(B, L, -1)  # (B, L, Ds)
        # print("Spatial Layer:", sfeat.shape)
        # Temporal
        tfeat = self.temporal_encoder(sfeat)                  # (B, Dt)
        # print("Tempral Layer:", tfeat.shape)
        # Statics
        sstat, has_static = self.static_encoder(DEM=DEM, awc=awc, fc=fc, soil=soil)
        # print("Static Layer:", sstat.shape)
        # Fusion
        if has_static:
            if sstat.shape[0] == 1 and B > 1:
                sstat = sstat.expand(B, -1)
            if self.fusion_type == "film":
                gamma = self.film_gamma(sstat)
                beta  = self.film_beta(sstat)
                fused = tfeat * (1 + gamma) + beta
            elif self.fusion_type == "concat":
                fused = self.concat(torch.cat([tfeat, sstat], dim=-1))
            else:  # cross_attention
                fused = self.cross(tfeat, sstat)
        else:
            fused = tfeat
        # print("Fused Layer:", fused.shape)
        # exit()
        
        out = self.output_head(fused)  # (B,1) or (B,output_dim)
        return out.squeeze(-1) if out.shape[-1] == 1 else out
