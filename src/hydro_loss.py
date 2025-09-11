# hydro_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HydroLossNorm(nn.Module):
    """
    Single-watershed batch; targets already in [0,1].
    loss = w_huber * Huber(y_hat*, y)           (normalized space)
         + w_nse  * (SSE / max(SST, floor))     (normalized space)
         + w_kge  * (1 - KGE)                   (normalized space)

    y_hat* can be clamped/sigmoided to [0,1] for stability.
    """
    def __init__(self,
                 w_huber: float = 0.7,
                 w_nse: float = 0.2,
                 w_kge: float = 0.1,
                 huber_beta: float = 0.05,
                 sst_floor_per_sample: float = 1e-3,  # since range is 1 in [0,1] space
                 nse_clip: float = 50.0,
                 squash: str = "clamp"  # "clamp" | "sigmoid" | "none"
                 ):
        super().__init__()
        self.w_huber = w_huber
        self.w_nse   = w_nse
        self.w_kge   = w_kge
        self.huber_beta = huber_beta
        self.sst_floor_per_sample = sst_floor_per_sample
        self.nse_clip = nse_clip
        self.squash = squash

    @staticmethod
    def _squash01(x, how: str):
        if how == "clamp":   return x.clamp(0, 1)
        if how == "sigmoid": return torch.sigmoid(x)
        return x  # "none"

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # shapes -> (B,)
        y_hat = y_hat.view(-1)
        y     = y.view(-1)

        # 1) Point loss in normalized space (keep AMP dtype)
        # y_hat_n = self._squash01(y_hat, self.squash)
        hub = F.smooth_l1_loss(y_hat, y, beta=self.huber_beta)

        # 2) Hydrology metrics in fp32 for stability (still normalized space)
        obs  = y.to(torch.float32)
        pred = y_hat.to(torch.float32)
        B = obs.numel()

        # NSE = 1 - SSE/SST  -> minimize SSE/SST
        mu_o = obs.mean()
        sse  = (obs - pred).pow(2).sum()
        sst  = (obs - mu_o).pow(2).sum()
        floor = obs.new_tensor(self.sst_floor_per_sample * B)  # no min/max; range=1 in [0,1]
        nse_ratio = (sse / torch.maximum(sst, floor).clamp_min(1e-8)).clamp_max(self.nse_clip)

        # KGE (on normalized values)
        mu_p  = pred.mean()
        std_o = obs.std(unbiased=False).clamp_min(1e-8)
        std_p = pred.std(unbiased=False).clamp_min(1e-8)
        cov   = ((obs - mu_o) * (pred - mu_p)).mean()
        r     = cov / (std_o * std_p + 1e-8)
        alpha = std_p / (std_o + 1e-8)
        beta  = mu_p / (mu_o + 1e-8)
        kge   = 1.0 - torch.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)
        kge_term = (1.0 - kge)

        # 3) Combine (cast metric terms back to AMP dtype)
        return (self.w_huber * hub
                + self.w_nse * nse_ratio.to(hub.dtype)
                + self.w_kge * kge_term.to(hub.dtype))
