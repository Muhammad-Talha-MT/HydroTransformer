# hydro_transformer_wStatic_pretrained.py
import os
import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Small utility
# =========================
def get_base_model(m: nn.Module) -> nn.Module:
    """Unwrap DataParallel for attribute access."""
    return m.module if isinstance(m, nn.DataParallel) else m


# =========================
# Spatial Encoding (from scratch)
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


class AdaptiveSpatialEncoder(nn.Module):
    """
    Spatial encoder that handles variable input sizes gracefully.
    Uses progressive downsampling with adaptive pooling at the end.
    Input:  (B*L, C, H, W)
    Output: (B*L, d_model)
    """
    def __init__(self, in_channels: int, d_model: int = 128, depth: int = 3):
        super().__init__()
        self.d_model = d_model

        layers = []
        c = in_channels
        # Initial downsample
        layers.append(ConvBlock(c, 64, 7, 2, 3))  # /2
        c = 64

        # Progressive encoding
        channels = [128, 256, d_model]
        for i in range(min(depth, len(channels))):
            c_out = channels[i]
            stride = 2 if i < 2 else 1  # Downsample first two layers
            layers.append(ConvBlock(c, c_out, 3, stride, 1))
            c = c_out

        self.encoder = nn.Sequential(*layers)
        self.adaptive_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(c * 16, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x):
        feat = self.encoder(x)
        return self.adaptive_pool(feat)


# =========================
# Static Encoders (multi-scale)
# =========================
class MultiScaleStaticEncoder(nn.Module):
    """
    Separate light encoders for DEM (1ch), AWC (1ch), FC (1ch), soil (3ch),
    then fuse by mean + MLP.
    All inputs expected as (B,C,H,W) already (your training code expands them).
    Output: (B, d_static)
    """
    def __init__(self, d_static: int = 64):
        super().__init__()
        self.d_static = d_static

        self.dem_encoder = nn.Sequential(
            ConvBlock(1, 16, 7, 4, 3),    # /4
            ConvBlock(16, 32, 5, 4, 2),   # /16
            ConvBlock(32, 64, 3, 2, 1),   # /32
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(64 * 4, d_static),
            nn.LayerNorm(d_static)
        )
        self.awc_fc_encoder = nn.Sequential(
            ConvBlock(1, 32, 5, 2, 2),    # /2
            ConvBlock(32, 64, 3, 2, 1),   # /4
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
            feats.append(self.dem_encoder(DEM))        # (B, d_static)
        if awc is not None:
            feats.append(self.awc_fc_encoder(awc))     # (B, d_static)
        if fc is not None:
            feats.append(self.awc_fc_encoder(fc))      # (B, d_static)
        if soil is not None:
            feats.append(self.soil_encoder(soil))      # (B, d_static)

        if len(feats) == 0:
            return None, False

        fused = torch.stack(feats, dim=0).mean(0)      # (B, d_static)
        fused = self.fusion(fused)                     # (B, d_static)
        return fused, True


# =========================
# Temporal Encoder (pretrained-friendly)
# =========================
class SimpleTransformerEncoderPretrained(nn.Module):
    """
    A vanilla TransformerEncoder with:
      - input projection to d_model
      - N encoder layers (norm_first=True)
      - final LayerNorm + Dropout
      - helpers to load a pretrained state_dict
      - helpers to freeze everything and unfreeze the last N layers
    Expects x: (B, L, d_in) and returns (B, d_model) using the LAST token.

    NEW:
      * If checkpoint_path points to a HF TimeSeriesTransformer repo (e.g.
        'kashif/time-series-transformer-mv-traffic-hourly'), this module will
        auto-download and map the encoder weights into nn.TransformerEncoder.
    """
    def __init__(
        self,
        d_in: int,
        d_model: int = 256,
        n_heads: int = 8,
        depth: int = 6,
        ff_mult: float = 4.0,
        dropout: float = 0.1,
        norm_first: bool = True,
        checkpoint_path: str | None = None,
        map_location: str | torch.device = "cpu",
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.depth = depth
        self.pretrained_loaded = False  # set True only if we actually load something

        self.proj_in = nn.Linear(d_in, d_model)
        dim_ff = int(d_model * ff_mult)  # allow ff_mult like 0.5 → 32 when d_model=64
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=norm_first
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # ----- Try to load weights -----
        if checkpoint_path:
            sd = None
            # Case 1: local file (.pth/.bin/.pt)
            if os.path.isfile(checkpoint_path):
                try:
                    sd = torch.load(checkpoint_path, map_location=map_location)
                    if isinstance(sd, dict) and "state_dict" in sd:
                        sd = sd["state_dict"]
                except Exception as e:
                    print(f"[Temporal] Local checkpoint load failed: {e}")

            # Case 2: treat as Hugging Face repo id -> raw state dict try
            if sd is None and "/" in str(checkpoint_path) and not os.path.exists(checkpoint_path):
                try:
                    from huggingface_hub import hf_hub_download
                    candidates = ("model.safetensors", "pytorch_model.bin", "pytorch_model.pt")
                    for fname in candidates:
                        try:
                            local = hf_hub_download(repo_id=checkpoint_path, filename=fname)
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

            # Try strict=False update for any matching keys (unlikely but cheap)
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

            # Case 3: explicit mapping from HF TimeSeriesTransformer encoder -> our encoder
            if not self.pretrained_loaded and "/" in str(checkpoint_path):
                ok = self._try_load_hf_tst_encoder(checkpoint_path, map_location=map_location)
                if ok:
                    self.pretrained_loaded = True
                else:
                    print("[Temporal] No compatible keys found in checkpoint (likely different architecture).")

    # ----- HF TimeSeriesTransformer → nn.TransformerEncoder mapper -----
    def _try_load_hf_tst_encoder(self, repo_id: str, map_location="cpu") -> bool:
        """
        Download a Hugging Face TimeSeriesTransformer checkpoint and map its
        encoder weights into nn.TransformerEncoderLayer format:
          q/k/v → self_attn.in_proj_(weight|bias)  (stacked [q;k;v])
          out_proj, fc1/fc2, self_attn_layer_norm→norm1, final_layer_norm→norm2
        Returns True on success.
        """
        try:
            import json
            from huggingface_hub import hf_hub_download

            # 1) Read config to sanity-check sizes
            cfg_path = hf_hub_download(repo_id=repo_id, filename="config.json")
            with open(cfg_path, "r") as f:
                cfg = json.load(f)

            d_model_hf = int(cfg.get("d_model", self.d_model))
            n_heads_hf = int(cfg.get("encoder_attention_heads", getattr(self.encoder.layers[0].self_attn, "num_heads", -1)))
            depth_hf   = int(cfg.get("encoder_layers", self.depth))
            ffn_dim_hf = int(cfg.get("encoder_ffn_dim", int(self.d_model * 4)))

            if (self.d_model != d_model_hf or
                n_heads_hf != self.encoder.layers[0].self_attn.num_heads or
                self.depth  != depth_hf or
                self.encoder.layers[0].linear1.out_features != ffn_dim_hf):
                print(f"[Temporal] Size mismatch vs HF config: "
                      f"d_model({self.d_model}!={d_model_hf}) or "
                      f"heads({self.encoder.layers[0].self_attn.num_heads}!={n_heads_hf}) or "
                      f"layers({self.depth}!={depth_hf}) or "
                      f"ffn({self.encoder.layers[0].linear1.out_features}!={ffn_dim_hf}).")
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
                cur[f"encoder.layers.{i}.self_attn.in_proj_bias"]   = torch.cat([q_b, k_b, v_b], dim=0)

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
            print(f"[Temporal] HF TST load: copied={loaded_tensors} | "
                  f"missing={len(missing)} unexpected={len(unexpected)}")
            return loaded_tensors > 0

        except Exception as e:
            print(f"[Temporal] HF TST mapping failed: {e}")
            return False

    # ----- freezing helpers -----
    @staticmethod
    def _set_requires_grad_(module: nn.Module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag

    def freeze_all(self, keep_proj: bool = True, keep_norm: bool = True):
        """Freeze all encoder layers; optionally keep input projection & final norm trainable."""
        self._set_requires_grad_(self.encoder, False)
        self._set_requires_grad_(self.proj_in, False)
        self._set_requires_grad_(self.norm, False)
        if keep_proj:
            self._set_requires_grad_(self.proj_in, True)
        if keep_norm:
            self._set_requires_grad_(self.norm, True)

    def unfreeze_last_n(self, n: int):
        """Unfreeze only the last n TransformerEncoder layers."""
        if n <= 0:
            return
        layers = list(self.encoder.layers)
        for layer in layers[-n:]:
            self._set_requires_grad_(layer, True)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ----- forward -----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_in)
        returns: (B, d_model) using last token
        """
        x = self.proj_in(x)                 # (B, L, d_model)
        x = self.encoder(x)                 # (B, L, d_model)
        x = self.norm(x[:, -1, :])          # (B, d_model)
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
        s = self.static_proj(static_feat).unsqueeze(1)   # (B,1,d_dyn)
        q = dynamic_feat.unsqueeze(1)                    # (B,1,d_dyn)
        attn, _ = self.mha(q, s, s)
        fused = self.norm1(q + attn).squeeze(1)          # (B,d_dyn)
        return self.norm2(self.out_proj(fused))


# =========================
# Main Model
# =========================
class ImprovedHydroTransformer(nn.Module):
    """
    End-to-end model:
      - Spatial CNN per timestep -> (B,L,Ds)
      - Temporal Transformer (pretrained-friendly) -> (B,Dt)
      - Static encoders -> (B,Ds_static)
      - Fusion (FiLM default) -> (B,Dt)
      - Head -> (B,1)
    """
    def __init__(
        self,
        in_channels: int,
        # spatial
        spatial_d_model: int = 128,
        spatial_depth: int = 3,
        # temporal
        temporal_d_model: int = 256,
        temporal_heads: int = 8,
        temporal_depth: int = 6,
        temporal_ff_mult: float = 4.0,
        temporal_dropout: float = 0.1,
        temporal_norm_first: bool = True,
        temporal_checkpoint_path: Optional[str] = None,  # torch .pth or HF repo id
        # static
        static_d_model: int = 64,
        # fusion
        fusion_type: str = "film",  # "film" | "concat" | "cross_attention"
        # head
        output_dim: int = 1,
        head_dropout: float = 0.1,
    ):
        super().__init__()

        # Spatial (B*L,C,H,W)->(B*L,Ds)
        self.spatial_encoder = AdaptiveSpatialEncoder(
            in_channels=in_channels, d_model=spatial_d_model, depth=spatial_depth
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
            checkpoint_path=temporal_checkpoint_path,
            map_location="cpu",
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
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    # ---- delegate freezing to temporal encoder ----
    def freeze_temporal_all(self, keep_proj: bool = True, keep_norm: bool = True):
        self.temporal_encoder.freeze_all(keep_proj=keep_proj, keep_norm=keep_norm)

    def unfreeze_temporal_last_n(self, n: int):
        self.temporal_encoder.unfreeze_last_n(n)

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
        sfeat = self.spatial_encoder(x2d).reshape(B, L, -1)   # (B,L,Ds)

        # Temporal
        tfeat = self.temporal_encoder(sfeat)                  # (B,Dt)

        # Statics
        sstat, has_static = self.static_encoder(DEM=DEM, awc=awc, fc=fc, soil=soil)

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

        out = self.output_head(fused)                         # (B,1) or (B,output_dim)
        return out.squeeze(-1) if out.shape[-1] == 1 else out
