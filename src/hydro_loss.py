# losses.py
import torch, torch.nn as nn, torch.nn.functional as F

class HydroCompositeLoss(nn.Module):
    def __init__(self,
                 w_mse=0.5, w_nse=0.3, w_q_hi=0.2, tau_hi=0.9,
                 w_q_lo=0.0, tau_lo=0.1,
                 var_mode="global",        # 'global' or 'batch'
                 train_var_y=None,         # float or tensor
                 var_floor=1e-3,           # 👈 raise this from 1e-6
                 detach_var=True):
        super().__init__()
        self.w_mse, self.w_nse = w_mse, w_nse
        self.w_q_hi, self.tau_hi = w_q_hi, tau_hi
        self.w_q_lo, self.tau_lo = w_q_lo, tau_lo
        self.var_mode, self.var_floor = var_mode, var_floor
        self.train_var_y = float(train_var_y) if train_var_y is not None else None
        self.detach_var = detach_var

    @staticmethod
    def _pinball(yhat, y, tau):
        e = y - yhat
        return torch.mean(torch.maximum(tau*e, (tau-1.0)*e))

    def forward(self, yhat, y, mask=None):
        yhat = torch.nan_to_num(yhat, 0.0, 1e6, -1e6)
        y    = torch.nan_to_num(y,    0.0, 1e6, -1e6)

        if mask is not None:
            mask = mask.to(y.dtype).to(y.device)
            yhat, y = yhat*mask, y*mask
            denom_count = mask.sum().clamp_min(1.0)
        else:
            denom_count = torch.tensor(1.0, device=y.device, dtype=y.dtype)

        mse = F.mse_loss(yhat, y, reduction='sum') / denom_count

        # ===== NSE surrogate denominator
        if self.var_mode == "global" and self.train_var_y is not None:
            var_y = torch.as_tensor(self.train_var_y, device=y.device, dtype=y.dtype)
        else:
            var_y = torch.var(y, unbiased=False)
            if self.detach_var: var_y = var_y.detach()
        var_y = torch.clamp(var_y, min=self.var_floor)   # 👈 clamp

        nse_term = mse / var_y          # equals (1 - NSÊ)

        q_hi = self._pinball(yhat, y, self.tau_hi)
        q_lo = self._pinball(yhat, y, self.tau_lo) if self.w_q_lo > 0 else y.new_tensor(0.0)

        return self.w_mse*mse + self.w_nse*nse_term + self.w_q_hi*q_hi + self.w_q_lo*q_lo
