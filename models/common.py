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
            self.criterion= self.get_criterion(method, forecast_task, dist_side, tau_pinball)
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
    
    def get_criterion(self, method, forecast_task, dist_side, tau_pinball):
        if forecast_task == 'point':
            criterion = nn.MSELoss()
        elif forecast_task == 'quantile':
            criterion = lambda pred, target: pinball_loss(pred, target, tau_pinball, dist_side)
        elif forecast_task == 'expectile':
            criterion = lambda pred, target: pinball_loss_expectile(pred, target, tau_pinball, dist_side)
        else:
            raise ValueError("distribution criterion is created in __init__")
    
    @staticmethod
    def add_task_specific_args(parent_parser):
        # Embedding
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