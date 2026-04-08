import os
import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.distributions import Normal
import pickle as pkl
from utils.losses import pinball_loss, pinball_loss_expectile
import tempfile
import shap
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, auc, average_precision_score
from models.utils import (
    chebyshev_lobatto_u, uniform_u, cdf_from_quantile_on_grid, bps_to_logprice,
    _ensure_dir, _batched_range, plot_quantile_cdf_pdf, plot_fan_chart, 
    plot_pit_hist, logit_u, power_tails_u, plot_pit_ecdf, 
    plot_tail_exceedance_calibration, plot_tail_exceedance_ratio,
    plot_pit_hist_from_values, plot_pit_ecdf_from_values, compute_spliced_pit_batched,
    plot_es_diagnostics, plot_var_es_timeline, compute_spliced_var_es_batched, build_spliced_tail_plot_grid, plot_tail_cdf_survival,
    plot_cross_attention_sample, plot_cross_attention_over_test, plot_gate_open_rate_bar, plot_topk_variable_traces_over_test, plot_variable_heatmap_over_test
    )
from models.helper_classes import RevIN, ChebyshevQuantile, ISplineQuantile, SplicedGPDQuantile, ShapProbWrapper
from models.losses import GaussianCRPS, GaussianTWCRPS, CRPSFromQuantiles, ThresholdWeightedCRPSFromQuantiles, BinaryFocalLoss, ChainingFunction


class Baseclass_forecast(L.LightningModule):
    def __init__(self,
                batch_size, test_batch_size,
                learning_rate, method,
                forecast_task, dist_side, tau_pinball,
                n_cheb, twcrps_threshold_low, twcrps_threshold_high, twcrps_side, twcrps_smooth_h, u_grid_size, dist_loss, grid_density, revin_type, 
                quantile_decomp, spline_degree, knot_kind, knot_p, 
                tail_model, gpd_u_low, gpd_u_high, gpd_xi_min, gpd_xi_max,
                cdf_grid_size=512, cdf_grid_min=None, cdf_grid_max=None, 
                l0_lambda = 0.0, save_test_diagnostics=False, diag_top_k_vars=20, diag_max_plot_samples=1000,use_log_price = False,
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
        self.grid_density = grid_density
        self.revin_type = revin_type
        self.quantile_decomp = quantile_decomp
        self.spline_degree = spline_degree
        self.knot_kind = knot_kind
        self.tail_model = tail_model
        self.gpd_u_low = gpd_u_low
        self.gpd_u_high = gpd_u_high
        self.gpd_xi_min = gpd_xi_min
        self.gpd_xi_max = gpd_xi_max
        self.l0_lambda = l0_lambda
        self.save_test_diagnostics = bool(save_test_diagnostics)
        self.diag_top_k_vars = int(diag_top_k_vars)
        self.diag_max_plot_samples = int(diag_max_plot_samples)
        self.knot_p = knot_p  
        # Convert TWCRPS thresholds and smooth_h from bps to log-price for internal use 
        if use_log_price:
            twcrps_threshold_low  = bps_to_logprice(twcrps_threshold_low)
            twcrps_threshold_high = bps_to_logprice(twcrps_threshold_high)
            twcrps_smooth_h = bps_to_logprice(twcrps_smooth_h)

        if self.forecast_task == 'distribution':
            if self.grid_density == 'chebyshev':
                u = chebyshev_lobatto_u(u_grid_size)
            elif self.grid_density == 'uniform':
                u = uniform_u(u_grid_size)
            elif self.grid_density == 'power-tail':
                u = power_tails_u(u_grid_size)
            elif self.grid_density == 'logit':
                u = logit_u(u_grid_size)

            # IMPORTANT for spliced tails: do not include exact 0 or 1
            u = u.float().clamp(min=1e-6, max=1.0 - 1e-6)

            if self.quantile_decomp == "chebyshev":
                body_quantile = ChebyshevQuantile(
                    K=n_cheb, u_grid=u, normalize=True, revin_type=self.revin_type
                )
            elif self.quantile_decomp == "spline":
                body_quantile = ISplineQuantile(
                    K=n_cheb, u_grid=u, normalize=True, revin_type=self.revin_type,
                    degree=self.spline_degree, knot_kind=self.knot_kind, knot_p=self.knot_p
                )

            if self.tail_model == "gpd":
                self.quantile = SplicedGPDQuantile(
                    body_quantile=body_quantile,
                    body_param_dim=2 + n_cheb,
                    u_low=self.gpd_u_low,
                    u_high=self.gpd_u_high,
                    xi_min=self.gpd_xi_min,
                    xi_max=self.gpd_xi_max,
                    eps=1e-6,
                )
            else:
                self.quantile = body_quantile

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
        elif self.forecast_task == "gaussian":
            if self.grid_density == 'chebyshev':
                u = chebyshev_lobatto_u(u_grid_size)
            elif self.grid_density == 'uniform':
                u = uniform_u(u_grid_size)
            elif self.grid_density == 'power-tail':
                u = power_tails_u(u_grid_size)
            elif self.grid_density == 'logit':
                u = logit_u(u_grid_size)
            else:
                raise ValueError(f"Unknown grid_density: {self.grid_density}")
            if dist_loss == 'crps':
                self.criterion = GaussianCRPS(u_grid=u, 
                                            crps_convention=True
                )
            elif dist_loss == 'twcrps':
                self.criterion = GaussianTWCRPS(
                    u_grid=u,
                    threshold_low=twcrps_threshold_low,
                    threshold_high=twcrps_threshold_high,
                    side=twcrps_side,
                    smooth_h=twcrps_smooth_h
                )
        else:
            self.criterion= self.get_criterion(forecast_task, dist_side, tau_pinball)
        self.method = method
        self.save_hyperparameters()

    def _compute_pred_loss(self, outputs, batch_y):
        if self.forecast_task == 'distribution':
            Q, q = self.quantile(outputs)
            pred_loss = self.criterion(Q, q, batch_y)
        elif self.forecast_task == 'gaussian':
            pred_loss = self.criterion(outputs, batch_y)
        else:
            pred_loss = self.criterion(outputs, batch_y)
        return pred_loss
        
    def _get_covariate_names_for_plots(self):
        """
        Returns names for the covariate branch only.
        Expected length = number of non-target input channels.
        """
        if hasattr(self, "var_names") and self.var_names is not None:
            return [str(x) for x in self.var_names]

        if hasattr(self, "model") and hasattr(self.model, "var_names") and self.model.var_names is not None:
            return [str(x) for x in self.model.var_names]

        return None


    def _get_target_name_for_plots(self):
        if hasattr(self, "target_name") and self.target_name is not None:
            return str(self.target_name)

        if hasattr(self, "model") and hasattr(self.model, "target_name") and self.model.target_name is not None:
            return str(self.model.target_name)

        return None

    def _append_test_diagnostics(self):
        aux = getattr(self.model, "latest_aux", None)
        if aux is None or not isinstance(aux, dict):
            return

        for key in ["selection_weights", "hard_gates", "expected_open"]:
            val = aux.get(key, None)
            if torch.is_tensor(val):
                self.test_diag_buffers[key].append(val.detach().cpu())

        cross_maps = aux.get("cross_attn_maps", None)
        if isinstance(cross_maps, list):
            cross_maps = [m.detach().cpu() for m in cross_maps if torch.is_tensor(m)]
            if len(cross_maps) > 0:
                self.test_diag_buffers["cross_attn_last"].append(cross_maps[-1])
                self.test_diag_buffers["cross_attn_mean_layers"].append(
                    torch.stack(cross_maps, dim=0).mean(dim=0)
                )

    def _finalize_test_diagnostics(self):
        diag = {}
        for key, vals in self.test_diag_buffers.items():
            if len(vals) == 0:
                continue
            diag[key] = torch.cat(vals, dim=0).numpy()
        return diag

    def _get_auxiliary_losses(self, device):
        zero = torch.zeros((), device=device)

        if not hasattr(self.model, "get_auxiliary_losses"):
            return {"l0_penalty": zero}

        aux = self.model.get_auxiliary_losses()
        if aux is None:
            return {"l0_penalty": zero}

        out = {}
        for k, v in aux.items():
            if v is None:
                out[k] = zero
            elif torch.is_tensor(v):
                out[k] = v
            else:
                out[k] = torch.tensor(float(v), device=device)

        if "l0_penalty" not in out:
            out["l0_penalty"] = zero

        return out


    def _compute_total_loss(self, outputs, batch_y):
        pred_loss = self._compute_pred_loss(outputs, batch_y)
        aux_losses = self._get_auxiliary_losses(device=batch_y.device)
        l0_penalty = aux_losses.get("l0_penalty", torch.zeros((), device=batch_y.device))
        total_loss = pred_loss + self.l0_lambda * l0_penalty
        return pred_loss, l0_penalty, total_loss

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        outputs = self.model(batch_x)
        pred_loss, l0_penalty, total_loss = self._compute_total_loss(outputs, batch_y)

        self.log('train_pred_loss', pred_loss, on_epoch=True, prog_bar=False)
        self.log('train_l0_penalty', l0_penalty, on_epoch=True, prog_bar=False)
        self.log('train_loss', total_loss, on_epoch=True, prog_bar=True)

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        outputs = self.model(batch_x)
        pred_loss, l0_penalty, total_loss = self._compute_total_loss(outputs, batch_y)

        self.log('val_loss', pred_loss, prog_bar=True)
        self.log('val_l0_penalty', l0_penalty, prog_bar=False)
        self.log('val_loss_total', total_loss, prog_bar=False)

    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        outputs = self.model(batch_x)
        pred_loss, l0_penalty, total_loss = self._compute_total_loss(outputs, batch_y)

        self.log('test_loss', pred_loss)
        self.log('test_l0_penalty', l0_penalty)
        self.log('test_loss_total', total_loss)

        # store raw outputs
        if batch_idx == 0:
            self.x_test = batch_x.detach().cpu()
            self.y_test = batch_y.detach().cpu()
            self.y_pred = outputs.detach().cpu()
        else:
            self.x_test = torch.cat((self.x_test, batch_x.detach().cpu()), dim=0)
            self.y_test = torch.cat((self.y_test, batch_y.detach().cpu()), dim=0)
            self.y_pred = torch.cat((self.y_pred, outputs.detach().cpu()), dim=0)

        if self.save_test_diagnostics:
            self._append_test_diagnostics()

    def on_test_epoch_start(self):
        self.x_test = None
        self.y_test = None
        self.y_pred = None

        self.test_diag_buffers = {
            "selection_weights": [],
            "hard_gates": [],
            "expected_open": [],
            "cross_attn_last": [],
            "cross_attn_mean_layers": [],
        }

    def on_test_epoch_end(self):
        run_dir = f'./{self.logger.experiment_id}/{self.logger.run_id}'
        plots_dir = os.path.join(run_dir, "plots")
        _ensure_dir(run_dir)
        _ensure_dir(plots_dir)

        # Everything starts on CPU
        A = {
            'true': np.array(self.y_test),
            'pred': np.array(self.y_pred),
            'seq':  np.array(self.x_test),
        }
        var_names = self._get_covariate_names_for_plots()
        target_name = self._get_target_name_for_plots()

        # if var_names is not None:
        #     A["covariate_names"] = list(var_names)

        # if target_name is not None:
        #     A["target_name"] = target_name

        if self.forecast_task == 'distribution':
            params_np = A['pred']                      # (B,H,P)
            y_true = A['true']                         # (B,H)
            B, H, P = params_np.shape
            J = int(self.quantile.u.numel())

            is_spliced_gpd = isinstance(self.quantile, SplicedGPDQuantile)

            # Choose a few representative samples (CPU)
            score = np.max(np.abs(y_true), axis=1)
            topk = np.argsort(-score)[:3]
            randk = np.random.choice(np.arange(B), size=min(2, B), replace=False)
            idxs = list(dict.fromkeys(list(topk) + list(randk)))  # unique

            thr_low = self.hparams.twcrps_threshold_low
            thr_high = self.hparams.twcrps_threshold_high
            side = self.hparams.twcrps_side

            # u-grid for plots (CPU)
            u_grid = self.quantile.u.detach().cpu().numpy()
            A['u_grid'] = u_grid

            # Pre-allocate CPU storage for Q,q
            Q_cpu = np.empty((B, H, J), dtype=np.float32)
            q_cpu = np.empty((B, H, J), dtype=np.float32)

            bs = getattr(self.hparams, "test_quantile_batch_size", 128)
            device = self.device
            self.quantile.eval()

            # -------------------------
            # Chunked quantile eval
            # -------------------------
            with torch.inference_mode():
                for b0, b1 in _batched_range(B, bs):
                    params_t = torch.from_numpy(params_np[b0:b1]).to(device=device, dtype=torch.float32)
                    Q_t, q_t = self.quantile(params_t)  # works for both plain and spliced models

                    Q_cpu[b0:b1] = Q_t.detach().cpu().numpy()
                    q_cpu[b0:b1] = q_t.detach().cpu().numpy()

                    del params_t, Q_t, q_t

            A['Q'] = Q_cpu
            A['q'] = q_cpu

            # -------------------------
            # z-grid for CDF/PDF plots
            # -------------------------
            y_np = y_true
            zmin = self.cdf_grid_min if self.cdf_grid_min is not None else float(np.nanmin(y_np) - 10.0)
            zmax = self.cdf_grid_max if self.cdf_grid_max is not None else float(np.nanmax(y_np) + 10.0)
            z_grid = np.linspace(zmin, zmax, self.cdf_grid_size, dtype=np.float32)
            A['z_grid'] = z_grid

            h0 = 0
            cdf_sel = {}
            pdf_sel = {}

            # -------------------------
            # Selected-example CDF/PDF
            # -------------------------
            with torch.inference_mode():
                z_grid_t = torch.from_numpy(z_grid).to(device=device, dtype=torch.float32)

                if is_spliced_gpd:
                    # For spliced tails, use a sample-specific plotting grid built from
                    # extreme predicted quantiles so the extrapolated GPD tails are visible.
                    z_grid_sel = {}
                    tail_plot_info = {}

                    for i in idxs:
                        params_i_t = torch.from_numpy(params_np[i:i+1, h0:h0+1]).to(device=device, dtype=torch.float32)  # (1,1,P)

                        # Build wide sample-specific grid from extreme predicted quantiles
                        z_grid_i_t, q_ext_i = build_spliced_tail_plot_grid(
                            spliced_quantile=self.quantile,
                            params_i_h=params_i_t,
                            alpha_plot_low=1e-5,     # adjust if you want even rarer tail display
                            alpha_plot_high=1e-5,
                            n_grid=self.cdf_grid_size,
                            pad_frac=0.05,
                        )

                        Fz_i_t = self.quantile.cdf_on_grid(params_i_t, z_grid_i_t)  # (1,1,Z)
                        fz_i_t = self.quantile.pdf_on_grid(params_i_t, z_grid_i_t)  # (1,1,Z)

                        z_grid_sel[int(i)] = z_grid_i_t.detach().cpu().numpy().astype(np.float32)
                        cdf_sel[int(i)] = Fz_i_t[0, 0].detach().cpu().numpy().astype(np.float32)
                        pdf_sel[int(i)] = fz_i_t[0, 0].detach().cpu().numpy().astype(np.float32)
                        tail_plot_info[int(i)] = q_ext_i

                        del params_i_t, z_grid_i_t, Fz_i_t, fz_i_t

                    A["z_grid_sel_h0"] = z_grid_sel
                    A["cdf_sel_h0"] = cdf_sel
                    A["pdf_sel_h0"] = pdf_sel
                    A["tail_plot_info_h0"] = tail_plot_info

                else:
                    # Original interior-grid CDF approximation
                    Q_sel_t = torch.from_numpy(Q_cpu[idxs, h0:h0+1]).to(device=device, dtype=torch.float32)  # (n,1,J)
                    Fz_sel_t = cdf_from_quantile_on_grid(Q_sel_t, self.quantile.u, z_grid_t)  # (n,1,Z)
                    Fz_sel = Fz_sel_t.detach().cpu().numpy()

                    for k, i in enumerate(idxs):
                        cdf_sel[int(i)] = Fz_sel[k, 0]

                    del Q_sel_t, Fz_sel_t

                del z_grid_t

            A['cdf_sel_h0'] = cdf_sel
            if is_spliced_gpd:
                A['pdf_sel_h0'] = pdf_sel

            # -------------------------
            # Optional: save splice/tail params for debugging
            # -------------------------
            if is_spliced_gpd:
                tail_info = {
                    "u_low": self.quantile.u_low,
                    "u_high": self.quantile.u_high,
                    "xi_min": self.quantile.xi_min,
                    "xi_max": self.quantile.xi_max,
                }

                with torch.inference_mode():
                    for b0, b1 in _batched_range(B, bs):
                        params_t = torch.from_numpy(params_np[b0:b1]).to(device=device, dtype=torch.float32)

                        body_params = params_t[..., :self.quantile.body_param_dim]
                        raw_xi_low = params_t[..., self.quantile.body_param_dim]
                        raw_xi_high = params_t[..., self.quantile.body_param_dim + 1]

                        _, _, xL, qL, xU, qU, xiL, xiU, betaL, betaU = self.quantile._tail_params_from_body(
                            body_params, raw_xi_low, raw_xi_high
                        )

                        if b0 == 0:
                            xL_all = xL.detach().cpu().numpy()
                            xU_all = xU.detach().cpu().numpy()
                            xiL_all = xiL.detach().cpu().numpy()
                            xiU_all = xiU.detach().cpu().numpy()
                            betaL_all = betaL.detach().cpu().numpy()
                            betaU_all = betaU.detach().cpu().numpy()
                        else:
                            xL_all = np.concatenate([xL_all, xL.detach().cpu().numpy()], axis=0)
                            xU_all = np.concatenate([xU_all, xU.detach().cpu().numpy()], axis=0)
                            xiL_all = np.concatenate([xiL_all, xiL.detach().cpu().numpy()], axis=0)
                            xiU_all = np.concatenate([xiU_all, xiU.detach().cpu().numpy()], axis=0)
                            betaL_all = np.concatenate([betaL_all, betaL.detach().cpu().numpy()], axis=0)
                            betaU_all = np.concatenate([betaU_all, betaU.detach().cpu().numpy()], axis=0)

                        del params_t, body_params, raw_xi_low, raw_xi_high
                        del xL, qL, xU, qU, xiL, xiU, betaL, betaU

                tail_info["xL"] = xL_all
                tail_info["xU"] = xU_all
                tail_info["xiL"] = xiL_all
                tail_info["xiU"] = xiU_all
                tail_info["betaL"] = betaL_all
                tail_info["betaU"] = betaU_all
                A["gpd_tail_info"] = tail_info

            # -------------------------
            # Plotting
            # -------------------------
            
            for i in idxs:
                prefix = os.path.join(plots_dir, f"s{i}_h{h0}")
                title = f"sample {i}, h={h0}"

                if is_spliced_gpd:
                    z_grid_i = A["z_grid_sel_h0"][int(i)]
                else:
                    z_grid_i = z_grid

                plot_quantile_cdf_pdf(
                    u_grid=u_grid,
                    Q_i_h=Q_cpu[i, h0],                 # (J,)
                    q_i_h=q_cpu[i, h0],                 # (J,)
                    z_grid=z_grid_i,                    # sample-specific for spliced tails
                    cdf_i_h=cdf_sel[int(i)],            # (Z,)
                    thr_low=thr_low, thr_high=thr_high, side=side,
                    title_prefix=title,
                    out_path_prefix=prefix
                )

                plot_fan_chart(
                    u_grid=u_grid,
                    Q_i_allH=Q_cpu[i],                  # (H,J)
                    y_true_i_allH=y_true[i],            # (H,)
                    thr_low=thr_low, thr_high=thr_high, side=side,
                    title_prefix=f"sample {i}",
                    out_path=os.path.join(plots_dir, f"s{i}_fan.png")
                )

                if is_spliced_gpd:
                    tail_info_i = A["tail_plot_info_h0"][int(i)]
                    plot_tail_cdf_survival(
                        z_grid=z_grid_i,
                        cdf_i_h=cdf_sel[int(i)],
                        xL=tail_info_i["q_lo_splice"],
                        xU=tail_info_i["q_hi_splice"],
                        title_prefix=title,
                        out_path=os.path.join(plots_dir, f"s{i}_h{h0}_tail_logcdf.png"),
                    )

            # -------------------------
            # PIT / ECDF / tail-calibration diagnostics
            # -------------------------
            if is_spliced_gpd:
                pit_vals = compute_spliced_pit_batched(
                    spliced_quantile=self.quantile,
                    params_np=params_np,
                    y_true_np=y_true,
                    horizon=0,
                    batch_size=bs,
                    device=device,
                )

                pit_vals, pit_stats = plot_pit_hist_from_values(
                    pits=pit_vals,
                    bins=20,
                    title_prefix="Test set (exact tail-aware)",
                    out_path=os.path.join(plots_dir, "pit_h0.png"),
                )

                _, pit_ecdf_stats = plot_pit_ecdf_from_values(
                    pits=pit_vals,
                    title_prefix="Test set (exact tail-aware)",
                    out_path=os.path.join(plots_dir, "pit_ecdf_h0.png"),
                )
            else:
                pit_vals, pit_stats = plot_pit_hist(
                    u_grid=u_grid,
                    Q_all=Q_cpu,
                    y_true=y_true,
                    horizon=0,
                    bins=20,
                    title_prefix="Test set",
                    out_path=os.path.join(plots_dir, "pit_h0.png")
                )

                _, pit_ecdf_stats = plot_pit_ecdf(
                    u_grid=u_grid,
                    Q_all=Q_cpu,
                    y_true=y_true,
                    horizon=0,
                    title_prefix="Test set",
                    out_path=os.path.join(plots_dir, "pit_ecdf_h0.png"),
                )

            tail_summary = plot_tail_exceedance_calibration(
                u_grid=u_grid,
                Q_all=Q_cpu,
                y_true=y_true,
                horizon=0,
                alphas=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
                title_prefix="Test set",
                out_path=os.path.join(plots_dir, "tail_exceedance_h0.png"),
            )

            plot_tail_exceedance_ratio(
                summary=tail_summary,
                title_prefix="Test set",
                out_path=os.path.join(plots_dir, "tail_exceedance_ratio_h0.png"),
            )

            A["pit_h0"] = pit_vals
            A["pit_stats_h0"] = pit_stats
            A["pit_ecdf_stats_h0"] = pit_ecdf_stats
            A["tail_exceedance_h0"] = tail_summary
            
            # -------------------------
            # Exact VaR / ES diagnostics for spliced GPD tails
            # -------------------------
            if is_spliced_gpd:
                # choose levels that really stress the tails
                es_alphas = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05], dtype=np.float32)

                # lower tail
                varL_h0, esL_h0 = compute_spliced_var_es_batched(
                    spliced_quantile=self.quantile,
                    params_np=params_np,
                    alphas=es_alphas,
                    horizon=0,
                    side="lower",
                    batch_size=bs,
                    n_int=256,
                    device=device,
                )

                # upper tail
                varU_h0, esU_h0 = compute_spliced_var_es_batched(
                    spliced_quantile=self.quantile,
                    params_np=params_np,
                    alphas=es_alphas,
                    horizon=0,
                    side="upper",
                    batch_size=bs,
                    n_int=256,
                    device=device,
                )

                es_lower_summary = plot_es_diagnostics(
                    y_true=y_true[:, 0],
                    var_pred=varL_h0,
                    es_pred=esL_h0,
                    alphas=es_alphas,
                    side="lower",
                    title_prefix="Test set h=0",
                    out_path=os.path.join(plots_dir, "es_diagnostics_lower_h0.png"),
                    log_x=True,
                )

                es_upper_summary = plot_es_diagnostics(
                    y_true=y_true[:, 0],
                    var_pred=varU_h0,
                    es_pred=esU_h0,
                    alphas=es_alphas,
                    side="upper",
                    title_prefix="Test set h=0",
                    out_path=os.path.join(plots_dir, "es_diagnostics_upper_h0.png"),
                    log_x=True,
                )

                # timeline plot at 1% if available, otherwise nearest level
                k_es = int(np.argmin(np.abs(es_alphas - 0.01)))

                plot_var_es_timeline(
                    y_true=y_true[:, 0],
                    var_alpha=varL_h0[:, k_es],
                    es_alpha=esL_h0[:, k_es],
                    alpha=float(es_alphas[k_es]),
                    side="lower",
                    title_prefix="Test set h=0",
                    out_path=os.path.join(plots_dir, "var_es_timeline_lower_h0.png"),
                )

                plot_var_es_timeline(
                    y_true=y_true[:, 0],
                    var_alpha=varU_h0[:, k_es],
                    es_alpha=esU_h0[:, k_es],
                    alpha=float(es_alphas[k_es]),
                    side="upper",
                    title_prefix="Test set h=0",
                    out_path=os.path.join(plots_dir, "var_es_timeline_upper_h0.png"),
                )

                A["es_alphas_h0"] = es_alphas
                A["var_lower_h0"] = varL_h0
                A["es_lower_h0"] = esL_h0
                A["var_upper_h0"] = varU_h0
                A["es_upper_h0"] = esU_h0
                A["es_lower_summary_h0"] = es_lower_summary
                A["es_upper_summary_h0"] = es_upper_summary

        elif self.forecast_task == 'gaussian':
            # -------------------------
            # Basic CPU arrays / shapes
            # -------------------------
            params_np = A['pred']                      # (B,H,2) = [mu, raw_scale]
            y_true = A['true']                         # (B,H)
            B, H, P = params_np.shape
            assert P == 2, f"Gaussian mode expects output shape (B,H,2), got last dim={P}"

            # representative samples, same policy as distribution branch
            score = np.max(np.abs(y_true), axis=1)
            topk = np.argsort(-score)[:3]
            randk = np.random.choice(np.arange(B), size=min(2, B), replace=False)
            idxs = list(dict.fromkeys(list(topk) + list(randk)))  # unique

            thr_low = self.hparams.twcrps_threshold_low
            thr_high = self.hparams.twcrps_threshold_high
            side = self.hparams.twcrps_side

            # -------------------------
            # Build u-grid for quantile plots
            # -------------------------
            if self.grid_density == 'chebyshev':
                u = chebyshev_lobatto_u(self.hparams.u_grid_size)
            elif self.grid_density == 'uniform':
                u = uniform_u(self.hparams.u_grid_size)
            elif self.grid_density == 'power-tail':
                u = power_tails_u(self.hparams.u_grid_size)
            elif self.grid_density == 'logit':
                u = logit_u(self.hparams.u_grid_size)
            else:
                raise ValueError(f"Unknown grid_density: {self.grid_density}")

            # Gaussian quantile / quantile-density explode at exact 0 and 1,
            # so clamp to an open interval.
            eps = float(getattr(self, "gaussian_eps", 1e-6))
            u = u.float().clamp(min=eps, max=1.0 - eps).contiguous()
            u_grid = u.detach().cpu().numpy()  # (J,)
            A['u_grid'] = u.detach().cpu().numpy()
            J = int(u.numel())

            # -------------------------
            # Decode mu, sigma
            # -------------------------
            mu_np = params_np[..., 0].astype(np.float32)  # (B,H)
            sigma_np = params_np[..., 1].astype(np.float32)


            A['mu'] = mu_np
            A['sigma'] = sigma_np

            # -------------------------
            # Precompute standard-normal objects on u-grid
            # -------------------------
            # z_u = Phi^{-1}(u), phi(z_u) = standard-normal pdf at z_u
            normal = Normal(
                loc=torch.tensor(0.0, dtype=u.dtype, device=u.device),
                scale=torch.tensor(1.0, dtype=u.dtype, device=u.device),
            )

            z_u = normal.icdf(u)  # (J,)
            phi_z_u = torch.exp(-0.5 * z_u**2) / math.sqrt(2.0 * math.pi)  # (J,)

            # q(u) = dQ/du = sigma / phi(Phi^{-1}(u))
            denom_q = phi_z_u.clamp_min(eps)

            # -------------------------
            # Compute Q(u), q(u) in chunks
            # -------------------------
            Q_cpu = np.empty((B, H, J), dtype=np.float32)
            q_cpu = np.empty((B, H, J), dtype=np.float32)

            bs = getattr(self.hparams, "test_quantile_batch_size", 128)

            with torch.inference_mode():
                for b0, b1 in _batched_range(B, bs):
                    mu_t = torch.from_numpy(mu_np[b0:b1]).float()        # (b,H)
                    sigma_t = torch.from_numpy(sigma_np[b0:b1]).float()  # (b,H)

                    Q_t = mu_t.unsqueeze(-1) + sigma_t.unsqueeze(-1) * z_u.view(1, 1, -1)   # (b,H,J)
                    q_t = sigma_t.unsqueeze(-1) / denom_q.view(1, 1, -1)                     # (b,H,J)

                    Q_cpu[b0:b1] = Q_t.detach().cpu().numpy().astype(np.float32)
                    q_cpu[b0:b1] = q_t.detach().cpu().numpy().astype(np.float32)

                    del mu_t, sigma_t, Q_t, q_t

            A['Q'] = Q_cpu
            A['q'] = q_cpu

            # -------------------------
            # z-grid for CDF/PDF plots
            # -------------------------
            zmin = self.cdf_grid_min if self.cdf_grid_min is not None else float(np.nanmin(y_true) - 10.0)
            zmax = self.cdf_grid_max if self.cdf_grid_max is not None else float(np.nanmax(y_true) + 10.0)
            z_grid = np.linspace(zmin, zmax, self.cdf_grid_size, dtype=np.float32)
            A['z_grid'] = z_grid

            # selected CDF/PDF only for plotting
            h0 = 0
            cdf_sel = {}
            pdf_sel = {}

            z_grid_t = torch.from_numpy(z_grid).float()
            sqrt_2 = math.sqrt(2.0)
            sqrt_2pi = math.sqrt(2.0 * math.pi)

            with torch.inference_mode():
                for i in idxs:
                    mu_i = torch.tensor(mu_np[i, h0], dtype=torch.float32)
                    sigma_i = torch.tensor(sigma_np[i, h0], dtype=torch.float32)

                    zz = (z_grid_t - mu_i) / sigma_i
                    cdf_i = 0.5 * (1.0 + torch.erf(zz / sqrt_2))
                    pdf_i = torch.exp(-0.5 * zz**2) / (sigma_i * sqrt_2pi)

                    cdf_sel[int(i)] = cdf_i.detach().cpu().numpy().astype(np.float32)
                    pdf_sel[int(i)] = pdf_i.detach().cpu().numpy().astype(np.float32)

                    del mu_i, sigma_i, zz, cdf_i, pdf_i

            A['cdf_sel_h0'] = cdf_sel
            A['pdf_sel_h0'] = pdf_sel

            # -------------------------
            # Plotting, same artifacts as distribution branch
            # -------------------------
            for i in idxs:
                prefix = os.path.join(plots_dir, f"s{i}_h{h0}")
                title = f"sample {i}, h={h0}"

                plot_quantile_cdf_pdf(
                    u_grid=A['u_grid'],                   # (J,)
                    Q_i_h=Q_cpu[i, h0],                  # (J,)
                    q_i_h=q_cpu[i, h0],                  # (J,)
                    z_grid=z_grid,                       # (Z,)
                    cdf_i_h=cdf_sel[int(i)],             # (Z,)
                    thr_low=thr_low, thr_high=thr_high, side=side,
                    title_prefix=title,
                    out_path_prefix=prefix
                )

                plot_fan_chart(
                    u_grid=A['u_grid'],
                    Q_i_allH=Q_cpu[i],                   # (H,J)
                    y_true_i_allH=y_true[i],             # (H,)
                    thr_low=thr_low, thr_high=thr_high, side=side,
                    title_prefix=f"sample {i}",
                    out_path=os.path.join(plots_dir, f"s{i}_fan.png")
                )

            pit_vals, pit_stats = plot_pit_hist(
                u_grid=u_grid,
                Q_all=Q_cpu,
                y_true=y_true,
                horizon=0,
                bins=20,
                title_prefix="Test set",
                out_path=os.path.join(plots_dir, "pit_h0.png")
            )
            pit_vals, pit_ecdf_stats = plot_pit_ecdf(
                u_grid=u_grid,
                Q_all=Q_cpu,
                y_true=y_true,
                horizon=0,
                title_prefix="Test set",
                out_path=os.path.join(plots_dir, "pit_ecdf_h0.png"),
            )

            tail_summary = plot_tail_exceedance_calibration(
                u_grid=u_grid,
                Q_all=Q_cpu,
                y_true=y_true,
                horizon=0,
                alphas=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
                title_prefix="Test set",
                out_path=os.path.join(plots_dir, "tail_exceedance_h0.png"),
            )

            plot_tail_exceedance_ratio(
                summary=tail_summary,
                title_prefix="Test set",
                out_path=os.path.join(plots_dir, "tail_exceedance_ratio_h0.png"),
            )

            A["pit_ecdf_stats_h0"] = pit_ecdf_stats
            A["tail_exceedance_h0"] = tail_summary
            A["pit_h0"] = pit_vals
            A["pit_stats_h0"] = pit_stats

        if self.save_test_diagnostics:
            diagnostics = self._finalize_test_diagnostics()
            A["diagnostics"] = diagnostics
            diag = A["diagnostics"]


            if "selection_weights" in diag:
                plot_variable_heatmap_over_test(
                    diag["selection_weights"],
                    out_path=os.path.join(plots_dir, "selection_weights_heatmap.png"),
                    var_names=var_names,
                    title="Selection weights over test windows",
                    top_k=self.diag_top_k_vars,
                    max_rows=self.diag_max_plot_samples,
                )
                plot_topk_variable_traces_over_test(
                    diag["selection_weights"],
                    out_path=os.path.join(plots_dir, "selection_weights_traces.png"),
                    var_names=var_names,
                    title="Top selection weights over test windows",
                    top_k=min(8, self.diag_top_k_vars),
                    smooth=11,
                )

            if "hard_gates" in diag:
                plot_variable_heatmap_over_test(
                    diag["hard_gates"],
                    out_path=os.path.join(plots_dir, "hard_gates_heatmap.png"),
                    var_names=var_names,
                    title="Hard gates over test windows",
                    top_k=self.diag_top_k_vars,
                    max_rows=self.diag_max_plot_samples,
                )
                plot_gate_open_rate_bar(
                    diag["hard_gates"],
                    out_path=os.path.join(plots_dir, "hard_gates_open_rate.png"),
                    var_names=var_names,
                    title="Mean hard-gate openness",
                    top_k=self.diag_top_k_vars,
                )

            if "cross_attn_mean_layers" in diag:
                plot_cross_attention_over_test(
                    diag["cross_attn_mean_layers"],
                    out_path=os.path.join(plots_dir, "cross_attention_over_test_meanH.png"),
                    var_names=var_names,
                    title="Cross-attention over test windows",
                    horizon_reduce="mean",
                    top_k=self.diag_top_k_vars,
                    max_rows=self.diag_max_plot_samples,
                )

                # representative sample: highest |target| event
                y_true_np = np.array(self.y_test)
                sample_idx = int(np.argmax(np.max(np.abs(y_true_np), axis=1))) if y_true_np.ndim == 2 else 0

                plot_cross_attention_sample(
                    diag["cross_attn_mean_layers"][sample_idx],
                    out_path=os.path.join(plots_dir, f"cross_attention_sample_{sample_idx}.png"),
                    var_names=var_names,
                    title=f"Cross-attention sample {sample_idx}",
                    top_k=self.diag_top_k_vars,
                )

        # ---- save pickle ----
        out_path = os.path.join(run_dir, "preds_test_set.pkl")
        with open(out_path, 'wb') as f:
            pkl.dump(A, f)

        # ---- log artifacts ----
        self.logger.experiment.log_artifact(self.logger.run_id, out_path)

        if os.path.isdir(plots_dir):
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
        class_parser.add_argument('--forecast_task', type=str, default = 'quantile', choices=['quantile', 'point', 'expectile', 'distribution', 'gaussian'], help='quantile, expectile or point forecasting')
        class_parser.add_argument('--dist_side', type=str, default='both', choices=['both', 'up', 'down'], help='side of the distribution to be predicted (for quantile/expectile forecasting)')

        loss_parser = parent_parser.add_argument_group('Training loss arguments')
        loss_parser.add_argument('--dist_loss', type=str, default='twcrps', choices=['crps','twcrps'], help='which distributional loss to use for distribution forecasting')
        loss_parser.add_argument('--twcrps_threshold_low', type=float, default=-10.0, help='lower threshold for twCRPS loss (for price target), in depeg bps (conversion to log is done internally)')
        loss_parser.add_argument('--twcrps_threshold_high', type=float, default=10.0, help='upper threshold for twCRPS loss (for price target), in depeg bps (conversion to log is done internally)')
        loss_parser.add_argument('--twcrps_side', type=str, default='two_sided', choices=['below','above', 'two_sided'], help='side of the distribution to consider for twCRPS loss')
        loss_parser.add_argument('--twcrps_smooth_h', type=float, default=2, help='smoothing parameter for twCRPS loss, in depeg bps (conversion to log is done internally)')
        loss_parser.add_argument('--tau_pinball', type=float, help='tau parameter for pinball loss (quantile/expectile regression)', default=0.05)
        
        distribution_parser = parent_parser.add_argument_group('Non parametric quantile function arguments (for distribution forecasting)')
        distribution_parser.add_argument('--u_grid_size', type=int, default=128, help='number of points in the Chebyshev grid for distribution forecasting')
        distribution_parser.add_argument('--grid_density', type=str, default='uniform', choices=['chebyshev', 'uniform', 'logit', 'power-tail'], help='type of grid for distribution forecasting')
        distribution_parser.add_argument('--quantile_decomp', type=str, default='chebyshev', choices=['chebyshev', 'spline'], help='type of quantile decomposition for distribution forecasting')
        distribution_parser.add_argument('--spline_degree', type=int, default=3, help='degree of I-spline basis (if quantile_decomp is spline)')
        distribution_parser.add_argument('--knot_kind', type=str, default='uniform', choices = ['power_tails', 'uniform'], help='kind of knot placement for I-splines (if quantile_decomp is spline)')
        distribution_parser.add_argument('--knot_p', type=float, default=3.0, help='power parameter for power-tail knot placement (if quantile_decomp is spline and knot_kind is power_tails)')
        distribution_parser.add_argument('--n_cheb', type=int, default=2, help='number of Chebyshev polynomials for distribution forecasting')
        
        tail_parser = parent_parser.add_argument_group('Spliced tail model arguments (for distribution forecasting)')
        tail_parser.add_argument('--tail_model', type=str, default='none', choices=['none', 'gpd'], help='whether to use a spliced tail model for distribution forecasting; if gpd, splice a GPD tail model onto the interior quantile function')
        tail_parser.add_argument('--gpd_u_low', type=float, default=0.01, help='lower splice point for GPD tail model (as quantile level)')
        tail_parser.add_argument('--gpd_u_high', type=float, default=0.99, help='upper splice point for GPD tail model (as quantile level)')
        tail_parser.add_argument('--gpd_xi_min', type=float, default=-0.25, help='minimum shape parameter for GPD tail model')
        tail_parser.add_argument('--gpd_xi_max', type=float, default=0.50, help='maximum shape parameter for GPD tail model')

        diag_parser = parent_parser.add_argument_group('Model regularization / diagnostics')
        diag_parser.add_argument('--l0_lambda', type=float, default=0.0,
                                help='weight for hard-concrete L0 penalty')
        diag_parser.add_argument('--save_test_diagnostics', type=int, choices=[0,1], default=0,
                                help='whether to save model diagnostics on test set')
        diag_parser.add_argument('--diag_top_k_vars', type=int, default=20,
                                help='top-k variables to show in diagnostic plots')
        diag_parser.add_argument('--diag_max_plot_samples', type=int, default=2000,
                                help='max number of test windows to show in heatmap diagnostics')
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
        self.log('train_loss', loss, on_epoch=True)
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
