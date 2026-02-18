import os
import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from utils.losses import pinball_loss, pinball_loss_expectile
import tempfile
import shap
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, auc, average_precision_score

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, mode = 'revin'):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine == 1:
            self._init_params()
        self.type = mode

    def forward(self, x, mode:str):
            if mode == 'norm':
                self._get_statistics(x)
                x = self._normalize(x)
            elif mode == 'denorm':
                x = self._denormalize(x)
            elif mode == 'denorm_scale':
                x = self._denormalize_scale(x)
            return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        if self.type == 'revin':
            dim2reduce = tuple(range(1, x.ndim-1))
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
            if self.affine :    
                self.mean = self.mean + self.affine_bias
                self.stdev = self.stdev * (torch.relu(self.affine_weight) + self.eps)
        elif self.type == 'robust':
            dim2reduce = tuple(range(1, x.ndim-1))
            self.mean = torch.median(x, dim=1, keepdim=True).values.detach()
            x_mad = torch.median(torch.abs(x-self.mean), dim=1, keepdim = True).values.detach()
            stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
            x_mad_aux = stdev * 0.6744897501960817
            x_mad = x_mad * (x_mad>0) + x_mad_aux * (x_mad==0)
            x_mad[x_mad==0] = 1.0
            x_mad = x_mad + self.eps
            self.stdev = x_mad
            if self.affine :    
                self.mean = self.mean + self.affine_bias
                self.stdev = self.stdev * (torch.relu(self.affine_weight) + self.eps)
    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        return x

    def _denormalize(self, x):
        x = x * self.stdev
        x = x + self.mean
        return x
    
    def _denormalize_scale(self, x, eps = 1e-5):  
        x = x * self.stdev
        return x
    def robust_statistics(self, x, dim=-1, eps=1e-6):
        return None


class BinaryFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.alpha = None if alpha is None else float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

        if pos_weight is not None:
            # register so it moves with .to(device)
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # logits: (B,) or (B,1), y: {0,1} same shape
        if logits.ndim > 1:
            logits = logits.squeeze(-1)
        if y.ndim > 1:
            y = y.squeeze(-1)

        y = y.float()

        # BCE term (stable)
        bce = F.binary_cross_entropy_with_logits(
            logits, y, reduction="none", pos_weight=self.pos_weight
        )

        # p_t = p if y=1 else (1-p)
        p = torch.sigmoid(logits)
        pt = y * p + (1.0 - y) * (1.0 - p)

        focal_factor = (1.0 - pt).pow(self.gamma)

        if self.alpha is None:
            alpha_t = 1.0
        else:
            alpha_t = y * self.alpha + (1.0 - y) * (1.0 - self.alpha)

        loss = alpha_t * focal_factor * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
    
class ShapProbWrapper(nn.Module):
    """Wraps a model that outputs probabilities so SHAP sees an nn.Module."""
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_model(x.float())  # already probabilities
        if out.ndim == 1:
            out = out.unsqueeze(1)        # (B, 1)
        elif out.ndim == 2 and out.shape[1] != 1:
            out = out[:, :1]
        return out


def chebyshev_lobatto_u(J: int, eps: float = 1e-5, device=None):
    # x_j = cos(pi*j/(J-1)) in [-1,1], dense at endpoints
    j = torch.arange(J, device=device, dtype=torch.float32)
    x = torch.cos(torch.pi * j / (J - 1))              # 1..-1
    u = (x + 1.0) / 2.0                                # in [0,1]
    u = torch.flip(u, dims=[0])                        # increasing
    u = u.clamp(eps, 1 - eps)                          # avoid exact 0/1
    return u

def chebyshev_basis(u: torch.Tensor, K: int):
    """
    u: (J,) in (0,1)
    returns T: (J,K) with T_k(2u-1)
    """
    x = 2*u - 1
    T0 = torch.ones_like(x)
    if K == 1:
        return T0.unsqueeze(-1)
    T1 = x
    Ts = [T0, T1]
    for k in range(2, K):
        Ts.append(2*x*Ts[-1] - Ts[-2])
    return torch.stack(Ts[:K], dim=-1)

class ChebyshevQuantile(nn.Module):
    """
    params -> (Q(u_j), q(u_j)) on a fixed u-grid, monotone by construction.
    params shape: (B, H, 2+K) = [b, log_s, a_0..a_{K-1}]
    """
    def __init__(self, K: int, u_grid: torch.Tensor, eps: float = 1e-6, normalize: bool = True):
        super().__init__()
        self.K = K
        self.eps = eps
        self.normalize = normalize

        # buffers move with .to(device)
        self.register_buffer("u", u_grid)                       # (J,)
        self.register_buffer("du", u_grid[1:] - u_grid[:-1])    # (J-1,)
        self.register_buffer("T", chebyshev_basis(u_grid, K))   # (J,K)

        # trapezoid weights on u for ∫_0^1 f(u) du
        du = self.du
        wu = torch.zeros_like(u_grid)
        wu[0] = du[0] / 2
        wu[-1] = du[-1] / 2
        wu[1:-1] = (du[:-1] + du[1:]) / 2
        self.register_buffer("wu", wu)                          # (J,)

    def forward(self, params: torch.Tensor):
        """
        returns:
          Q: (B,H,J)
          q: (B,H,J)  (quantile density)
        """
        b = params[..., 0]                           # (B,H)
        s = F.softplus(params[..., 1]) + self.eps    # (B,H)
        a = params[..., 2:]                          # (B,H,K)

        # g(u) = sum_k a_k T_k(u)
        g = torch.einsum("bhk,jk->bhj", a, self.T)    # (B,H,J)

        # q(u) > 0
        q = F.softplus(g) + self.eps                 # (B,H,J)

        # integrate q to get Q on the u grid (cumulative trapezoid)
        q_mid = 0.5 * (q[..., 1:] + q[..., :-1])     # (B,H,J-1)
        integ = torch.cumsum(q_mid * self.du, dim=-1) # (B,H,J-1)
        integ = torch.cat([torch.zeros_like(b)[..., None], integ], dim=-1)  # (B,H,J)

        if self.normalize:
            # normalize integral so that Q(0)=b and Q(1)=b+s
            total = integ[..., -1] + self.eps        # (B,H)
            Q = b[..., None] + s[..., None] * integ / total[..., None]
            q = s[..., None] * q / total[..., None]  # adjust density consistently
        else:
            Q = b[..., None] + integ

        return Q, q
       
def cdf_from_quantile_on_grid(Q: torch.Tensor, u: torch.Tensor, z_grid: torch.Tensor, eps: float = 1e-8):
    """
    Q:      (B,H,J) monotone increasing in J
    u:      (J,)
    z_grid: (Z,)
    returns:
      F: (B,H,Z) with F(z_grid)
    """
    assert Q.ndim == 3
    B, H, J = Q.shape
    Z = z_grid.numel()

    # Expand u so we can gather with batched indices
    u_expand = u.view(1, 1, J).expand(B, H, J)  # (B,H,J)

    # idx_raw in [0..J], shape (B,H,Z)
    idx_raw = torch.searchsorted(Q, z_grid.view(1, 1, Z).expand(B, H, Z), right=True)

    # Masks for out-of-range z
    below = (idx_raw == 0)
    above = (idx_raw >= J)

    # Clamp to [1, J-1] so we can take idx-1 and idx safely
    idx = idx_raw.clamp(1, J - 1)

    idx0 = idx - 1
    idx1 = idx

    Q0 = Q.gather(-1, idx0)
    Q1 = Q.gather(-1, idx1)
    u0 = u_expand.gather(-1, idx0)
    u1 = u_expand.gather(-1, idx1)

    z = z_grid.view(1, 1, Z).expand(B, H, Z)
    t = (z - Q0) / (Q1 - Q0 + eps)
    F = u0 + t * (u1 - u0)

    # Set exact tails
    F = torch.where(below, torch.zeros_like(F), F)
    F = torch.where(above, torch.ones_like(F), F)
    return F


class CRPSFromQuantiles(nn.Module):
    """
    CRPS(F,y) = ∫ (F(z) - 1{y<=z})^2 dz
              = ∫_0^1 (u - 1{y<=Q(u)})^2 q(u) du
    """
    def __init__(self, u: torch.Tensor, wu: torch.Tensor):
        super().__init__()
        self.register_buffer("u", u)    # (J,)
        self.register_buffer("wu", wu)  # (J,)

    def forward(self, Q: torch.Tensor, q: torch.Tensor, y: torch.Tensor):
        # Q,q: (B,H,J), y: (B,H)
        I = (y.unsqueeze(-1) <= Q).float()
        u = self.u.view(1, 1, -1)
        integrand = (u - I).pow(2) * q
        loss_bh = torch.sum(integrand * self.wu.view(1, 1, -1), dim=-1)
        return loss_bh.mean()


class ThresholdWeightedCRPSFromQuantiles(nn.Module):
    def __init__(self, u: torch.Tensor, wu: torch.Tensor,
                 threshold_low: float, threshold_high: float,
                 side: str = "two_sided", smooth_h: float = 0.0):
        super().__init__()
        self.register_buffer("u", u)
        self.register_buffer("wu", wu)

        self.side = side
        self.smooth_h = smooth_h

        if side == "two_sided":
            self.threshold = (threshold_low, threshold_high)
        elif side == "below":
            self.threshold = float(threshold_low)
        elif side == "above":
            self.threshold = float(threshold_high)
        else:
            raise ValueError("side must be one of: below, above, two_sided")

    def weight(self, z: torch.Tensor):
        h = self.smooth_h
        if self.side == "two_sided":
            rL, rU = self.threshold
            if h and h > 0:
                wL = torch.sigmoid((rL - z) / h)  # ~1 if z<=rL
                wU = torch.sigmoid((z - rU) / h)  # ~1 if z>=rU
                return wL + wU
            return (z <= rL).float() + (z >= rU).float()

        r = self.threshold
        if h and h > 0:
            if self.side == "below":
                return torch.sigmoid((r - z) / h)
            return torch.sigmoid((z - r) / h)

        if self.side == "below":
            return (z <= r).float()
        return (z >= r).float()

    def forward(self, Q: torch.Tensor, q: torch.Tensor, y: torch.Tensor):
        I = (y.unsqueeze(-1) <= Q).float()
        wQ = self.weight(Q)
        u = self.u.view(1, 1, -1)
        integrand = wQ * (u - I).pow(2) * q
        loss_bh = torch.sum(integrand * self.wu.view(1, 1, -1), dim=-1)
        return loss_bh.mean()
   
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_quantile_cdf_pdf(u_grid, Q_i_h, q_i_h, z_grid, cdf_i_h,
                          thr_low=None, thr_high=None, side="two_sided",
                          title_prefix="", out_path_prefix="dist"):
    """
    u_grid: (J,)
    Q_i_h:  (J,)
    q_i_h:  (J,)
    z_grid: (Z,)
    cdf_i_h:(Z,)
    """
    # 1) Quantile function
    plt.figure(figsize=(7,4))
    plt.plot(u_grid, Q_i_h, lw=2)
    plt.xlabel("u")
    plt.ylabel("Q(u)  [depeg bps]")
    plt.title(f"{title_prefix} Quantile function")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path_prefix + "_quantile.png", dpi=150)
    plt.close()

    # 2) CDF
    plt.figure(figsize=(7,4))
    plt.plot(z_grid, cdf_i_h, lw=2)
    if side in ["two_sided", "below"] and thr_low is not None:
        plt.axvline(thr_low, color="black", ls="--", lw=1, label="thr_low")
    if side in ["two_sided", "above"] and thr_high is not None:
        plt.axvline(thr_high, color="black", ls="--", lw=1, label="thr_high")
    plt.xlabel("z  [depeg bps]")
    plt.ylabel("F(z)")
    plt.title(f"{title_prefix} CDF")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path_prefix + "_cdf.png", dpi=150)
    plt.close()

    # 3) Implied PDF at quantile grid: f(Q(u)) ≈ 1 / q(u)
    pdf_at_Q = 1.0 / np.maximum(q_i_h, 1e-8)
    plt.figure(figsize=(7,4))
    plt.plot(Q_i_h, pdf_at_Q, lw=2)
    if side in ["two_sided", "below"] and thr_low is not None:
        plt.axvline(thr_low, color="black", ls="--", lw=1)
    if side in ["two_sided", "above"] and thr_high is not None:
        plt.axvline(thr_high, color="black", ls="--", lw=1)
    plt.xlabel("z = Q(u)  [depeg bps]")
    plt.ylabel("pdf(z)  (approx)")
    plt.title(f"{title_prefix} Implied PDF")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path_prefix + "_pdf.png", dpi=150)
    plt.close()


def plot_fan_chart(u_grid, Q_i_allH, y_true_i_allH,
                   qs=(0.01, 0.05, 0.5, 0.95, 0.99),
                   thr_low=None, thr_high=None, side="two_sided",
                   title_prefix="", out_path="fan.png"):
    """
    Q_i_allH: (H,J)
    y_true_i_allH: (H,)
    """
    H, J = Q_i_allH.shape
    u_grid = np.asarray(u_grid)

    def Q_at_p(Q_h, p):
        return np.interp(p, u_grid, Q_h)

    bands = {p: np.array([Q_at_p(Q_i_allH[h], p) for h in range(H)]) for p in qs}
    x = np.arange(H)

    plt.figure(figsize=(9,4))
    # widest band first
    if 0.01 in bands and 0.99 in bands:
        plt.fill_between(x, bands[0.01], bands[0.99], alpha=0.12, label="98% PI")
    if 0.05 in bands and 0.95 in bands:
        plt.fill_between(x, bands[0.05], bands[0.95], alpha=0.20, label="90% PI")

    if 0.5 in bands:
        plt.plot(x, bands[0.5], lw=2, label="median")

    plt.plot(x, y_true_i_allH, lw=1.5, color="red", label="true")

    if side in ["two_sided", "below"] and thr_low is not None:
        plt.axhline(thr_low, color="black", ls="--", lw=1)
    if side in ["two_sided", "above"] and thr_high is not None:
        plt.axhline(thr_high, color="black", ls="--", lw=1)

    plt.xlabel("horizon step")
    plt.ylabel("depeg bps")
    plt.title(f"{title_prefix} Fan chart over horizon")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pit_hist(u_grid, Q_all, y_true, horizon=0, bins=20,
                  title_prefix="", out_path="pit.png"):
    """
    PIT for a fixed horizon: PIT = F(y). We approximate via inverting Q:
      PIT = interp(y, Q(u), u)
    Q_all:  (B,H,J)
    y_true: (B,H)
    """
    u_grid = np.asarray(u_grid)
    Q_h = Q_all[:, horizon, :]      # (B,J)
    y_h = y_true[:, horizon]        # (B,)

    pits = []
    for i in range(Q_h.shape[0]):
        Qi = Q_h[i]
        yi = y_h[i]
        # clip to support to avoid NaNs
        if yi <= Qi[0]:
            pit = 0.0
        elif yi >= Qi[-1]:
            pit = 1.0
        else:
            pit = float(np.interp(yi, Qi, u_grid))
        pits.append(pit)

    pits = np.asarray(pits)

    plt.figure(figsize=(7,4))
    plt.hist(pits, bins=bins, range=(0,1), density=True, alpha=0.7, edgecolor="black")
    plt.axhline(1.0, color="red", lw=1.5, label="uniform density")
    plt.xlabel("PIT = F(y)")
    plt.ylabel("density")
    plt.title(f"{title_prefix} PIT histogram (h={horizon})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()



class Baseclass_forecast(L.LightningModule):
    def __init__(self,
                batch_size, test_batch_size,
                learning_rate, method,
                forecast_task, dist_side, tau_pinball,
                n_cheb, twcrps_threshold_low, twcrps_threshold_high, twcrps_side, twcrps_smooth_h, u_grid_size, dist_loss,
                cdf_grid_size=512, cdf_grid_min=None, cdf_grid_max=None,
                **kwargs
                ):
        super().__init__()
        # Save hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.cdf_grid_size = cdf_grid_size
        self.cdf_grid_min = cdf_grid_min
        self.cdf_grid_max = cdf_grid_max
        self.forecast_task = forecast_task
        if self.forecast_task == 'distribution':
            u = chebyshev_lobatto_u(u_grid_size)
            self.quantile = ChebyshevQuantile(K=n_cheb, u_grid=u, normalize=True)
            if dist_loss == 'crps':
                self.criterion = CRPSFromQuantiles(self.quantile.u, self.quantile.wu)
            else:
                self.criterion = ThresholdWeightedCRPSFromQuantiles(
                    u=self.quantile.u, wu=self.quantile.wu,
                    threshold_low=twcrps_threshold_low,
                    threshold_high=twcrps_threshold_high,
                    side=twcrps_side,
                    smooth_h=twcrps_smooth_h
                )
        else:
            self.criterion= self.get_criterion(forecast_task, dist_side, tau_pinball)
        self.method = method
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        outputs = self.model(batch_x)

        if self.forecast_task == 'distribution':
            Q, q = self.quantile(outputs)
            loss = self.criterion(Q, q, batch_y)
        else:
            loss = self.criterion(outputs, batch_y)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        outputs = self.model(batch_x)

        if self.forecast_task == 'distribution':
            Q, q = self.quantile(outputs)
            loss = self.criterion(Q, q, batch_y)
        else:
            loss = self.criterion(outputs, batch_y)

        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        outputs = self.model(batch_x)

        if self.forecast_task == 'distribution':
            Q, q = self.quantile(outputs)
            test_loss = self.criterion(Q, q, batch_y)
        else:
            test_loss = self.criterion(outputs, batch_y)

        self.log('test_loss', test_loss)

        # store raw outputs (params) for distribution; store point/quantiles otherwise
        if batch_idx == 0:
            self.x_test = batch_x.detach().cpu()
            self.y_test = batch_y.detach().cpu()
            self.y_pred = outputs.detach().cpu()
        else:
            self.x_test = torch.cat((self.x_test, batch_x.detach().cpu()), dim=0)
            self.y_test = torch.cat((self.y_test, batch_y.detach().cpu()), dim=0)
            self.y_pred = torch.cat((self.y_pred, outputs.detach().cpu()), dim=0)
        
    def on_test_epoch_end(self):

        run_dir = f'./{self.logger.experiment_id}/{self.logger.run_id}'
        plots_dir = os.path.join(run_dir, "plots")
        _ensure_dir(run_dir)
        _ensure_dir(plots_dir)

        A = {
            'true': np.array(self.y_test),
            'pred': np.array(self.y_pred),
            'seq':  np.array(self.x_test),
        }

        # ---- compute Q,q and CDF if distribution ----
        if self.forecast_task == 'distribution':
            with torch.no_grad():
                params = torch.tensor(A['pred'], dtype=torch.float32, device=self.device)  # (B,H,2+K)
                Q_t, q_t = self.quantile(params)  # (B,H,J)

                # CDF grid in bps space
                y_np = A['true']
                zmin = self.cdf_grid_min if self.cdf_grid_min is not None else float(np.nanmin(y_np) - 10.0)
                zmax = self.cdf_grid_max if self.cdf_grid_max is not None else float(np.nanmax(y_np) + 10.0)
                z_grid_t = torch.linspace(zmin, zmax, self.cdf_grid_size, device=self.device)

                Fz_t = cdf_from_quantile_on_grid(Q_t, self.quantile.u, z_grid_t)

            # move to CPU numpy for saving/plotting
            u_grid = self.quantile.u.detach().cpu().numpy()
            Q = Q_t.detach().cpu().numpy()
            q = q_t.detach().cpu().numpy()
            z_grid = z_grid_t.detach().cpu().numpy()
            cdf = Fz_t.detach().cpu().numpy()

            A['u_grid'] = u_grid
            A['Q']      = Q
            A['q']      = q
            A['z_grid'] = z_grid
            A['cdf']    = cdf

            # ---- choose a few representative examples to plot ----
            y_true = A['true']  # (B,H)
            B, H = y_true.shape

            # pick "most extreme" samples by max |depeg| over horizon
            score = np.max(np.abs(y_true), axis=1)
            topk = np.argsort(-score)[:3]  # top 3
            randk = np.random.choice(np.arange(B), size=min(2, B), replace=False)
            idxs = list(dict.fromkeys(list(topk) + list(randk)))  # unique, keep order

            thr_low = self.hparams.twcrps_threshold_low
            thr_high = self.hparams.twcrps_threshold_high
            side = self.hparams.twcrps_side

            # plot per-sample at a selected horizon (e.g. h=0)
            h0 = 0
            for j, i in enumerate(idxs):
                prefix = os.path.join(plots_dir, f"s{i}_h{h0}")
                title = f"sample {i}, h={h0}"

                plot_quantile_cdf_pdf(
                    u_grid=u_grid,
                    Q_i_h=Q[i, h0],
                    q_i_h=q[i, h0],
                    z_grid=z_grid,
                    cdf_i_h=cdf[i, h0],
                    thr_low=thr_low, thr_high=thr_high, side=side,
                    title_prefix=title,
                    out_path_prefix=prefix
                )

                # fan chart across all horizons for that sample
                plot_fan_chart(
                    u_grid=u_grid,
                    Q_i_allH=Q[i],               # (H,J)
                    y_true_i_allH=y_true[i],     # (H,)
                    thr_low=thr_low, thr_high=thr_high, side=side,
                    title_prefix=f"sample {i}",
                    out_path=os.path.join(plots_dir, f"s{i}_fan.png")
                )

            # PIT histogram (calibration) for horizon 0
            plot_pit_hist(
                u_grid=u_grid,
                Q_all=Q,
                y_true=y_true,
                horizon=0,
                bins=20,
                title_prefix="Test set",
                out_path=os.path.join(plots_dir, "pit_h0.png")
            )

        # ---- save pickle ----
        out_path = os.path.join(run_dir, "preds_test_set.pkl")
        pkl.dump(A, open(out_path, 'wb'))

        # ---- log artifacts ----
        self.logger.experiment.log_artifact(self.logger.run_id, out_path)

        # log plots directory if created
        if os.path.isdir(plots_dir):
            # log individual files (simple and reliable)
            for fn in os.listdir(plots_dir):
                if fn.endswith(".png"):
                    self.logger.experiment.log_artifact(self.logger.run_id, os.path.join(plots_dir, fn))

    def predict_step(self, batch, batch_idx):
        return self.model(batch)
    
    def forward(self, batch, batch_idx):
        return self.model(batch)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def get_criterion(self, forecast_task, dist_side, tau_pinball):
        if forecast_task == 'point':
            criterion = nn.MSELoss()
        elif forecast_task == 'quantile':
            criterion = lambda pred, target: pinball_loss(pred, target, tau_pinball, dist_side)
        elif forecast_task == 'expectile':
            criterion = lambda pred, target: pinball_loss_expectile(pred, target, tau_pinball, dist_side)
        else:
            raise ValueError("distribution criterion is created in __init__")
        return criterion
    
    @staticmethod
    def add_task_specific_args(parent_parser):
        class_parser = parent_parser.add_argument_group('Base class arguments')
        class_parser.add_argument('--forecast_task', type=str, default = 'quantile', choices=['quantile', 'point', 'expectile', 'distribution'], help='quantile, expectile or point forecasting')
        class_parser.add_argument('--dist_side', type=str, default='both', choices=['both', 'up', 'down'], help='side of the distribution to be predicted (for quantile/expectile forecasting)')
        class_parser.add_argument('--tau_pinball', type=float, help='tau parameter for pinball loss (quantile/expectile regression)', default=0.05)
        class_parser.add_argument('--n_cheb', type=int, default=2, help='number of Chebyshev polynomials for distribution forecasting')
        
        class_parser.add_argument('--dist_loss', type=str, default='twcrps', choices=['crps','twcrps'])
        class_parser.add_argument('--twcrps_threshold_low', type=float, default=-10.0)  # for price target
        class_parser.add_argument('--twcrps_threshold_high', type=float, default=10.0)  # for price target
        class_parser.add_argument('--twcrps_side', type=str, default='two_sided', choices=['below','above', 'two_sided'])
        class_parser.add_argument('--twcrps_smooth_h', type=float, default=2)
        class_parser.add_argument('--u_grid_size', type=int, default=256)
        return parent_parser


class Baseclass_earlywarning(L.LightningModule):
    def __init__(self,
                batch_size, test_batch_size,learning_rate,
                class_loss,
                compute_shap, shap_background_size, shap_test_samples,
                focal_alpha, focal_gamma, pos_weight = None,
                **kwargs
                ):
        super().__init__()
        # Save hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.pos_weight = pos_weight
        if self.pos_weight is None: 
            pos_weight = torch.tensor([1.0], dtype=torch.float32)
    
        self.class_loss = class_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.threshold = 0.5 # for validation confusion matrix ; arbitrary
        self.val_probs, self.val_true = [], []
        self.test_probs, self.test_true, self.test_seq = [], [], []
        self.criterion= self.get_criterion(class_loss)
        self.compute_shap = compute_shap
        self.shap_background_size = shap_background_size
        self.shap_test_samples = shap_test_samples
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch    
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        outputs = self.model(batch_x)
        logits = outputs.squeeze(-1)  # (B,)
        loss = self.criterion(logits, batch_y)
        self.log('train_loss', loss)
        return loss
    
    def on_validation_epoch_start(self):
        self.val_probs, self.val_true = [], []

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()  

        outputs = self.model(batch_x)                  
        logits = outputs.squeeze(-1)  # (B,)

        loss = self.criterion(logits, batch_y)
        prob =torch.sigmoid(logits)  # (B,)

        self.val_probs.append(prob.detach().cpu())
        self.val_true.append(batch_y.detach().cpu())

        self.log("val_loss", loss, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        if len(self.val_true) == 0:
            return

        y_true = torch.cat(self.val_true).numpy()
        y_prob = torch.cat(self.val_probs).numpy()

        auc, auprc = self._safe_auc_auprc(y_true, y_prob)
        self.log("val_auc", auc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_auprc", auprc, prog_bar=True, on_step=False, on_epoch=True)

        y_pred = (y_prob >= self.threshold).astype(int)

        with tempfile.TemporaryDirectory() as td:
            cm_path = os.path.join(td, "val_confusion_matrix.png")
            self._plot_confusion_matrix(
                y_true=y_true.astype(int),
                y_pred=y_pred,
                title=f"Val Confusion Matrix (thr={self.threshold:.2f})",
                out_path=cm_path
            )
            self._mlflow_log_artifact(cm_path, artifact_path="validation")

    def _shap_forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x.float())  # already probabilities
        if out.ndim == 1:
            out = out.unsqueeze(1)
        elif out.ndim == 2 and out.shape[1] != 1:
            # if ever multi-class, you'd handle differently
            out = out[:, :1]
        return out
    
    def _log_shap_on_test(self):
        if not self.trainer.is_global_zero:
            return
        if len(getattr(self, "test_seq", [])) == 0:
            return

        X_test = torch.cat(self.test_seq, dim=0)  # (N, seq_len, n_features)
        n_test = X_test.shape[0]

        n_eval = min(self.shap_test_samples, n_test)
        n_bg   = min(self.shap_background_size, n_test)

        idx = torch.randperm(n_test)[:n_eval]
        bg_idx = torch.randperm(n_test)[:n_bg]

        X_eval = X_test[idx]
        X_bg   = X_test[bg_idx]

        device = next(self.model.parameters()).device
        X_eval = X_eval.to(device)
        X_bg   = X_bg.to(device)

        shap_model = ShapProbWrapper(self.model).to(device)
        shap_model.eval()

        # IMPORTANT: ensure autograd works even if Lightning uses inference_mode=True
        with torch.inference_mode(False), torch.enable_grad():
            explainer = shap.GradientExplainer(shap_model, X_bg)
            shap_values = explainer.shap_values(X_eval)

        # shap_values can be either an array or a list of arrays (one per output)
        if isinstance(shap_values, list):
            shap_arr = shap_values[0]
        else:
            shap_arr = shap_values

        X_eval_cpu = X_eval.detach().cpu().numpy()
        idx_cpu = idx.detach().cpu().numpy()

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)

            npz_path = td / "shap_test_subset.npz"
            np.savez_compressed(
                npz_path,
                shap_values=shap_arr,   # typically (B, seq_len, n_features)
                x_eval=X_eval_cpu,
                test_indices=idx_cpu
            )
            self._mlflow_log_artifact(str(npz_path), artifact_path="test/shap")

            # Optional: summary plot (aggregate over time -> (B, n_features))
            shap_agg = np.abs(shap_arr).mean(axis=1)  # mean over seq_len
            x_agg = X_eval_cpu.mean(axis=1)

            plt.figure(figsize=(8, 4), dpi=150)
            shap.summary_plot(shap_agg, features=x_agg, show=False)
            fig_path = td / "shap_summary_mean_over_time.png"
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()

            self._mlflow_log_artifact(str(fig_path), artifact_path="test/shap")

    def on_test_epoch_start(self):
        self.test_probs, self.test_true, self.test_seq = [], [], []
        self.test_price_next = []

    def test_step(self, batch, batch_idx):
        batch_x, batch_y2 = batch
        batch_x = batch_x.float()
        y_target = batch_y2[:,0].float()
        price_next = batch_y2[:,1].float()

        outputs = self.model(batch_x)                  # probs
        logits = outputs.squeeze(-1)  # (B,)
        prob = torch.sigmoid(logits)  # (B,)
        loss = self.criterion(logits, y_target)

        self.test_probs.append(prob.detach().cpu())
        self.test_true.append(y_target.detach().cpu())
        self.test_seq.append(batch_x.detach().cpu())   
        self.test_price_next.append(price_next.detach().cpu())

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        if len(self.test_true) == 0:
            return

        y_true = torch.cat(self.test_true).numpy()
        y_prob = torch.cat(self.test_probs).numpy()
        x_seq = torch.cat(self.test_seq, dim=0).numpy()
        price_next = torch.cat(self.test_price_next).numpy()
        best_thr = self._best_threshold_from_roc(y_true, y_prob, default=self.threshold)
        self.log("test_best_threshold_roc", best_thr, prog_bar=True, on_step=False, on_epoch=True)

        y_pred = (y_prob >= best_thr).astype(int)

        auc, auprc = self._safe_auc_auprc(y_true, y_prob)
        self.log("test_auc", auc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_auprc", auprc, prog_bar=True, on_step=False, on_epoch=True)


        payload = {
            "true": y_true,
            "prob": y_prob,
            "pred": y_pred,
            "seq": x_seq,  
            "price_next": price_next,
            "threshold": best_thr,
        }

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)

            pkl_path = td / "preds_test_set.pkl"
            with open(pkl_path, "wb") as f:
                pkl.dump(payload, f)

            cm_path = td / "test_confusion_matrix.png"
            self._plot_confusion_matrix(
                y_true=y_true.astype(int),
                y_pred=y_pred,
                title=f"Test Confusion Matrix (thr={best_thr:.2f})",
                out_path=str(cm_path)
            )

            self._mlflow_log_artifact(str(pkl_path), artifact_path="test")
            self._mlflow_log_artifact(str(cm_path), artifact_path="test")
            
            timeline_path = td / "test_prob_next_price_timeline.png"
            self._plot_test_prob_price_through_time(
                y_prob=y_prob,
                y_true=y_true,
                price_next=price_next,
                threshold = best_thr,
                out_path=str(timeline_path),
            )

            roc_path = td / "test_roc.png"
            pr_path  = td / "test_precision_recall.png"
            self._plot_roc_pr_curves(y_true, y_prob, str(roc_path), str(pr_path))

            self._mlflow_log_artifact(str(timeline_path), artifact_path="test")
            self._mlflow_log_artifact(str(roc_path), artifact_path="test")
            self._mlflow_log_artifact(str(pr_path), artifact_path="test")
        
        if self.compute_shap:
            try:
                self._log_shap_on_test()
            except Exception as e:
                print(f"----- Error computing SHAP values on test set: {e} -----")
    
    def predict_step(self, batch, batch_idx):
        return self.model(batch)
    
    def forward(self, batch, batch_idx):
        return self.model(batch)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def get_criterion(self, class_loss):
        if class_loss == 'bce':
            criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        elif class_loss == 'focal':
            alpha = self.focal_alpha
            gamma = self.focal_gamma
            return BinaryFocalLoss(alpha=alpha, gamma=gamma, pos_weight=self.pos_weight, reduction="mean")
        else:
            raise ValueError(f"Unknown class_loss: {class_loss}")

        return criterion
    
    def _ensure_1d_prob_and_target(self, outputs: torch.Tensor, batch_y: torch.Tensor):
        """
        outputs: probs from model, expected shape (B,) or (B,1)
        batch_y: shape (B,)
        returns: (prob_1d, y_1d) both float tensors on same device
        """
        prob = outputs
        if prob.ndim > 1:
            prob = prob.squeeze(-1)
        prob = prob.float().clamp(0.0, 1.0)

        y = batch_y
        if y.ndim > 1:
            y = y.squeeze(-1)
        y = y.float()
        return prob, y


    def _safe_auc_auprc(self, y_true: np.ndarray, y_prob: np.ndarray):
        # AUC/AUPRC undefined if only one class present
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = np.nan
        try:
            auprc = average_precision_score(y_true, y_prob)
        except Exception:
            auprc = np.nan
        return auc, auprc


    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: str):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=["0", "1"], yticklabels=["0", "1"],
            xlabel="Predicted", ylabel="True", title=title
        )

        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    
    def _best_threshold_from_roc(self, y_true: np.ndarray, y_prob: np.ndarray, default: float = 0.5) -> float:
        try:
            fpr, tpr, thr = roc_curve(y_true.astype(int), y_prob)

            # roc_curve often includes thr[0] = inf; exclude non-finite thresholds
            m = np.isfinite(thr)
            if m.sum() == 0:
                return float(default)

            fpr, tpr, thr = fpr[m], tpr[m], thr[m]
            j = tpr - fpr
            best_idx = int(np.argmax(j))
            return float(thr[best_idx])
        except Exception:
            return float(default)
        
    def _plot_test_prob_price_through_time(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        price_next: np.ndarray,
        out_path: str,
        threshold: float,
        max_points: int = 5000,
    ):
        N = len(y_prob)
        if N == 0:
            return

        # downsample if needed
        if N > max_points:
            idx = np.linspace(0, N - 1, max_points).astype(int)
            t = idx
            y_prob = y_prob[idx]
            y_true = y_true[idx]
            price_next = price_next[idx]
        else:
            t = np.arange(N)

        fig, ax1 = plt.subplots(figsize=(12, 4), dpi=150)

        ax1.plot(t, y_prob, color="royalblue", lw=1.5, label="Pred prob")
        ax1.axhline(threshold, color="cornflowerblue", ls="--", lw=1, alpha=0.6,
                    label=f"thr={self.threshold:.2f}")
        ax1.set_ylim(-0.02, 1.02)
        ax1.set_ylabel("P(event)", color="royalblue")
        ax1.tick_params(axis="y", labelcolor="royalblue")

        mask = (y_true.astype(int) == 1)
        if mask.any():
            ax1.scatter(t[mask], y_prob[mask], s=18, color="red", zorder=5, label="True = 1")
            for tt in t[mask]:
                ax1.axvline(tt, color="red", alpha=0.08, lw=1)

        ax1.set_xlabel("Test sample index (order in dataloader)")
        ax1.set_title("Test predicted probability through time + NEXT price overlay")

        ax2 = ax1.twinx()
        ax2.plot(t, price_next, color="black", alpha=0.35, lw=1.0, label="Next price")
        ax2.set_ylabel("Next price", color="black")
        ax2.tick_params(axis="y", labelcolor="black")

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


    def _plot_roc_pr_curves(self, y_true: np.ndarray, y_prob: np.ndarray, roc_path: str, pr_path: str):
        y_true = y_true.astype(int)

        # ROC
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            fig = plt.figure(figsize=(5, 4), dpi=150)
            plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], "k--", lw=1)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC (test)")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(roc_path)
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] ROC curve not plotted: {e}")

        # Precision-Recall
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            fig = plt.figure(figsize=(5, 4), dpi=150)
            plt.plot(recall, precision, lw=2, label=f"AP={ap:.3f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision–Recall (test)")
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.savefig(pr_path)
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] PR curve not plotted: {e}")

    def _mlflow_log_artifact(self, local_path: str, artifact_path: str):
        # only log once in DDP
        if not self.trainer.is_global_zero:
            return
        self.logger.experiment.log_artifact(self.logger.run_id, local_path, artifact_path=artifact_path)

    @staticmethod
    def add_task_specific_args(parent_parser):
        early_warning = parent_parser.add_argument_group('Base early warning class arguments')
        early_warning.add_argument('--class_loss', type=str, default='bce', choices=['bce', 'focal'], help='loss function for classification task')
        early_warning.add_argument('--focal_alpha', type=float, default=0.25, help='alpha parameter for focal loss function for classification task')
        early_warning.add_argument('--focal_gamma', type=float, default=2.0, help='gamma parameter for focal loss function for classification task')
        early_warning.add_argument('--compute_shap', type=int, choices=[0,1], default=0, help='whether to compute SHAP values at test time')
        early_warning.add_argument('--shap_background_size', type=int, default=64, help='number of background samples for SHAP')
        early_warning.add_argument('--shap_test_samples', type=int, default=256, help='number of test samples to compute SHAP values for (keep small; SHAP can be expensive)')
        return parent_parser
