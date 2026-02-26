import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
import lightning as L
import pickle as pkl
import os
import shap
import argparse
from models.common import RevIN, ShapProbWrapper, Baseclass_forecast, Baseclass_earlywarning    
from einops import rearrange
from torch.distributions import Normal, StudentT
from neuralforecast.losses.pytorch import DistributionLoss, sCRPS
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.losses import pinball_loss, pinball_loss_expectile
import tempfile
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt


class Model(nn.Module):

    def __init__(self, seq_len, pred_len, d_model, d_ff, dropout, e_layers, activation, embed, freq, n_heads, factor, enc_in,
                method = 'forecast', forecast_task = 'quantile', dist_side = 'both',
                affine = True, revin_type= 'revin',
                n_cheb = 2):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.method = method
        self.n_cheb = n_cheb
        self.enc_in = enc_in
        self.revin = RevIN(self.enc_in, affine = affine, mode=revin_type)
        self.forecast_task = forecast_task
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, embed, freq,
                                                    dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        if dist_side == 'both' and forecast_task in ['quantile', 'expectile']:
            self.projection = nn.Linear(d_model, self.pred_len * 2)
        elif (dist_side in ['up', 'down'] and forecast_task in ['quantile', 'expectile']) or forecast_task == 'point':
            self.projection = nn.Linear(d_model, self.pred_len) 
        elif forecast_task == 'distribution':
            self.projection = nn.Linear(d_model, self.pred_len * (2 + self.n_cheb))
        self.classify = nn.Linear(d_model, 1)


    def forecast(self, x_enc):

        _, _, N = x_enc.shape
        
        x_enc = self.revin(x_enc, 'norm')

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        dec_out = self.projection(enc_out).permute(0, 2, 1)
        if self.forecast_task == 'distribution':
            dec_out, std_out, cheb_out = torch.split(enc_out, [self.pred_len, self.pred_len, self.pred_len * self.n_cheb], dim=-2)
            dec_out = self.revin(dec_out, 'denorm')
            std_out = self.revin(std_out, 'denorm_scale')
            dec_out = dec_out[:,:,-1]
            std_out = std_out[:,:,-1]
            cheb_out = cheb_out[:,:,-1].view(cheb_out.shape[0], self.pred_len, self.n_cheb)
            dec_out = torch.cat([dec_out.unsqueeze(-1), std_out.unsqueeze(-1), cheb_out], dim= -1)
        else:
            dec_out = self.revin(dec_out, 'denorm')
            dec_out = dec_out[:,:,-1]
        return dec_out
    
    def earlywarning(self, x_enc):
        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        class_out = self.classify(enc_out[:, -1, :])
        return class_out

    def forward(self, x_enc):
        if self.method == 'forecast':
            dec_out = self.forecast(x_enc)
            if self.forecast_task == 'distribution':
                return dec_out
            else:
                return torch.squeeze(dec_out, dim=-1)
                     
        if self.method == 'earlywarning':
            class_out = self.earlywarning(x_enc)
            class_out = torch.squeeze(class_out, dim=-1)
            return class_out
        

class iTransformer_forecast(Baseclass_forecast):
    def __init__(self, 
                seq_len, pred_len, d_model, dropout,
                e_layers, activation, embed, freq, n_heads, factor, d_ff,
                enc_in, method, batch_size, test_batch_size, affine, revin_type,
                forecast_task, dist_side, tau_pinball,
                n_cheb, twcrps_threshold_low, twcrps_threshold_high, twcrps_side, 
                twcrps_smooth_h, u_grid_size, dist_loss, grid_density, quantile_decomp, degree, knot_kind, knot_p,
                **kwargs
                ):
        super(iTransformer_forecast, self).__init__(
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            learning_rate=0.0001,
            method=method,
            forecast_task=forecast_task,
            dist_side=dist_side,
            tau_pinball=tau_pinball,
            n_cheb=n_cheb,
            twcrps_threshold_low=twcrps_threshold_low,
            twcrps_threshold_high=twcrps_threshold_high,
            twcrps_side=twcrps_side,
            twcrps_smooth_h=twcrps_smooth_h,
            u_grid_size=u_grid_size,
            grid_density=grid_density,
            dist_loss=dist_loss,
            revin_type=revin_type,
            quantile_decomp=quantile_decomp,
            degree=degree,
            knot_kind=knot_kind,   
            knot_p=knot_p,
        )
        self.model = Model(seq_len, pred_len, d_model, d_ff, dropout, e_layers, 
                           activation, embed, freq, n_heads, factor, enc_in, method, 
                           forecast_task, dist_side, affine, revin_type,
                           n_cheb
                           )
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = parent_parser.add_argument_group('Model-specific arguments')
        model_parser.add_argument('--embed', type=str, default='fixed')
        model_parser.add_argument('--freq', type=str, default='h')

        # Properties of Attn layer
        model_parser.add_argument('--d_model', type=int, default=2048)
        model_parser.add_argument('--factor', type=int, default=3)
        model_parser.add_argument('--n_heads', type=int, default=3)
        model_parser.add_argument('--d_ff', type=int, default=1024)
        model_parser.add_argument('--activation', type=str, default='gelu')
        model_parser.add_argument('--dropout', type=float,default=0.1)
        model_parser.add_argument('--e_layers', type=int, default=2)
        model_parser.add_argument('--scaler', type=str,default='revin')
        model_parser.add_argument('--affine', type=int, choices = [0,1], default=1)
        Baseclass_forecast.add_task_specific_args(parent_parser)
        return parent_parser


class iTransformer_earlywarning(Baseclass_earlywarning):
    def __init__(self, 
                seq_len, d_model, dropout, learning_rate,
                e_layers, activation, embed, freq, n_heads, factor, d_ff,
                enc_in, method, batch_size,
                class_loss, compute_shap, shap_background_size, shap_test_samples, pos_weight, focal_alpha, focal_gamma,        
                **kwargs
                ):
        super(iTransformer_earlywarning, self).__init__(
            batch_size,
            batch_size,
            learning_rate,
            class_loss,
            compute_shap,
            shap_background_size,
            shap_test_samples,
            focal_alpha,
            focal_gamma,
            pos_weight
        )
        self.model = Model(seq_len, pred_len=1, d_model=d_model, d_ff=d_ff, dropout=dropout, e_layers=e_layers, 
                        activation=activation, embed=embed, freq=freq, n_heads=n_heads, factor=factor, enc_in=enc_in, method=method, 
                        affine=0, scaler=None,
                        )
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = parent_parser.add_argument_group('Model-specific arguments')
        model_parser.add_argument('--embed', type=str, default='fixed')
        model_parser.add_argument('--freq', type=str, default='h')

        # Properties of Attn layer
        model_parser.add_argument('--d_model', type=int, default=2048)
        model_parser.add_argument('--factor', type=int, default=3)
        model_parser.add_argument('--n_heads', type=int, default=3)
        model_parser.add_argument('--d_ff', type=int, default=1024)
        model_parser.add_argument('--activation', type=str, default='gelu')
        model_parser.add_argument('--dropout', type=float,default=0.1)
        model_parser.add_argument('--e_layers', type=int, default=2)
        Baseclass_earlywarning.add_task_specific_args(parent_parser)
        return parent_parser
        






# class iTransformer_forecast(L.LightningModule):
#     def __init__(self,
#                 seq_len, pred_len, batch_size, test_batch_size,
#                 learning_rate, method,
#                 embed, freq, dropout, e_layers,
#                 d_model, factor, n_heads, d_ff, activation, enc_in,
#                 affine, scaler, forecast_task, dist_side, tau_pinball,
#                 n_cheb, twcrps_threshold_low, twcrps_threshold_high, twcrps_side, twcrps_smooth_h, u_grid_size, dist_loss,
#                 cdf_grid_size=512, cdf_grid_min=None, cdf_grid_max=None,
#                 **kwargs
#                 ):
#         super().__init__()
#         # Save hyperparameters
#         self.learning_rate = learning_rate
#         self.batch_size = batch_size
#         self.test_batch_size = test_batch_size
#         self.model = Model(seq_len, pred_len, d_model, d_ff, dropout, e_layers, 
#                            activation, embed, freq, n_heads, factor, enc_in, method, 
#                            forecast_task, dist_side, affine, scaler, 
#                            n_cheb, twcrps_threshold_low, twcrps_threshold_high, twcrps_side, twcrps_smooth_h, u_grid_size
#                            )
#         self.cdf_grid_size = cdf_grid_size
#         self.cdf_grid_min = cdf_grid_min
#         self.cdf_grid_max = cdf_grid_max
#         self.forecast_task = forecast_task
#         if self.forecast_task == 'distribution':
#             u = chebyshev_lobatto_u(u_grid_size)
#             self.quantile = ChebyshevQuantile(K=n_cheb, u_grid=u, normalize=True)
#             if dist_loss == 'crps':
#                 self.criterion = CRPSFromQuantiles(self.quantile.u, self.quantile.wu)
#             else:
#                 self.criterion = ThresholdWeightedCRPSFromQuantiles(
#                     u=self.quantile.u, wu=self.quantile.wu,
#                     threshold_low=twcrps_threshold_low,
#                     threshold_high=twcrps_threshold_high,
#                     side=twcrps_side,
#                     smooth_h=twcrps_smooth_h
#                 )
#         else:
#             self.criterion= self.get_criterion(method, forecast_task, dist_side, tau_pinball)
#         self.method = method
#         self.save_hyperparameters()

#     def training_step(self, batch, batch_idx):
#         batch_x, batch_y = batch
#         batch_x = batch_x.float()
#         batch_y = batch_y.float()

#         outputs = self.model(batch_x)

#         if self.forecast_task == 'distribution':
#             Q, q = self.quantile(outputs)
#             loss = self.criterion(Q, q, batch_y)
#         else:
#             loss = self.criterion(outputs, batch_y)

#         self.log('train_loss', loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         batch_x, batch_y = batch
#         batch_x = batch_x.float()
#         batch_y = batch_y.float()

#         outputs = self.model(batch_x)

#         if self.forecast_task == 'distribution':
#             Q, q = self.quantile(outputs)
#             loss = self.criterion(Q, q, batch_y)
#         else:
#             loss = self.criterion(outputs, batch_y)

#         self.log('val_loss', loss)

#     def test_step(self, batch, batch_idx):
#         batch_x, batch_y = batch
#         batch_x = batch_x.float()
#         batch_y = batch_y.float()

#         outputs = self.model(batch_x)

#         if self.forecast_task == 'distribution':
#             Q, q = self.quantile(outputs)
#             test_loss = self.criterion(Q, q, batch_y)
#         else:
#             test_loss = self.criterion(outputs, batch_y)

#         self.log('test_loss', test_loss)

#         # store raw outputs (params) for distribution; store point/quantiles otherwise
#         if batch_idx == 0:
#             self.x_test = batch_x.detach().cpu()
#             self.y_test = batch_y.detach().cpu()
#             self.y_pred = outputs.detach().cpu()
#         else:
#             self.x_test = torch.cat((self.x_test, batch_x.detach().cpu()), dim=0)
#             self.y_test = torch.cat((self.y_test, batch_y.detach().cpu()), dim=0)
#             self.y_pred = torch.cat((self.y_pred, outputs.detach().cpu()), dim=0)
        
#     def on_test_epoch_end(self):

#         run_dir = f'./{self.logger.experiment_id}/{self.logger.run_id}'
#         plots_dir = os.path.join(run_dir, "plots")
#         _ensure_dir(run_dir)
#         _ensure_dir(plots_dir)

#         A = {
#             'true': np.array(self.y_test),
#             'pred': np.array(self.y_pred),
#             'seq':  np.array(self.x_test),
#         }

#         # ---- compute Q,q and CDF if distribution ----
#         if self.forecast_task == 'distribution':
#             with torch.no_grad():
#                 params = torch.tensor(A['pred'], dtype=torch.float32, device=self.device)  # (B,H,2+K)
#                 Q_t, q_t = self.quantile(params)  # (B,H,J)

#                 # CDF grid in bps space
#                 y_np = A['true']
#                 zmin = self.cdf_grid_min if self.cdf_grid_min is not None else float(np.nanmin(y_np) - 10.0)
#                 zmax = self.cdf_grid_max if self.cdf_grid_max is not None else float(np.nanmax(y_np) + 10.0)
#                 z_grid_t = torch.linspace(zmin, zmax, self.cdf_grid_size, device=self.device)

#                 Fz_t = cdf_from_quantile_on_grid(Q_t, self.quantile.u, z_grid_t)

#             # move to CPU numpy for saving/plotting
#             u_grid = self.quantile.u.detach().cpu().numpy()
#             Q = Q_t.detach().cpu().numpy()
#             q = q_t.detach().cpu().numpy()
#             z_grid = z_grid_t.detach().cpu().numpy()
#             cdf = Fz_t.detach().cpu().numpy()

#             A['u_grid'] = u_grid
#             A['Q']      = Q
#             A['q']      = q
#             A['z_grid'] = z_grid
#             A['cdf']    = cdf

#             # ---- choose a few representative examples to plot ----
#             y_true = A['true']  # (B,H)
#             B, H = y_true.shape

#             # pick "most extreme" samples by max |depeg| over horizon
#             score = np.max(np.abs(y_true), axis=1)
#             topk = np.argsort(-score)[:3]  # top 3
#             randk = np.random.choice(np.arange(B), size=min(2, B), replace=False)
#             idxs = list(dict.fromkeys(list(topk) + list(randk)))  # unique, keep order

#             thr_low = self.hparams.twcrps_threshold_low
#             thr_high = self.hparams.twcrps_threshold_high
#             side = self.hparams.twcrps_side

#             # plot per-sample at a selected horizon (e.g. h=0)
#             h0 = 0
#             for j, i in enumerate(idxs):
#                 prefix = os.path.join(plots_dir, f"s{i}_h{h0}")
#                 title = f"sample {i}, h={h0}"

#                 plot_quantile_cdf_pdf(
#                     u_grid=u_grid,
#                     Q_i_h=Q[i, h0],
#                     q_i_h=q[i, h0],
#                     z_grid=z_grid,
#                     cdf_i_h=cdf[i, h0],
#                     thr_low=thr_low, thr_high=thr_high, side=side,
#                     title_prefix=title,
#                     out_path_prefix=prefix
#                 )

#                 # fan chart across all horizons for that sample
#                 plot_fan_chart(
#                     u_grid=u_grid,
#                     Q_i_allH=Q[i],               # (H,J)
#                     y_true_i_allH=y_true[i],     # (H,)
#                     thr_low=thr_low, thr_high=thr_high, side=side,
#                     title_prefix=f"sample {i}",
#                     out_path=os.path.join(plots_dir, f"s{i}_fan.png")
#                 )

#             # PIT histogram (calibration) for horizon 0
#             plot_pit_hist(
#                 u_grid=u_grid,
#                 Q_all=Q,
#                 y_true=y_true,
#                 horizon=0,
#                 bins=20,
#                 title_prefix="Test set",
#                 out_path=os.path.join(plots_dir, "pit_h0.png")
#             )

#         # ---- save pickle ----
#         out_path = os.path.join(run_dir, "preds_test_set.pkl")
#         pkl.dump(A, open(out_path, 'wb'))

#         # ---- log artifacts ----
#         self.logger.experiment.log_artifact(self.logger.run_id, out_path)

#         # log plots directory if created
#         if os.path.isdir(plots_dir):
#             # log individual files (simple and reliable)
#             for fn in os.listdir(plots_dir):
#                 if fn.endswith(".png"):
#                     self.logger.experiment.log_artifact(self.logger.run_id, os.path.join(plots_dir, fn))

#     def predict_step(self, batch, batch_idx):
#         return self.model(batch)
    
#     def forward(self, batch, batch_idx):
#         return self.model(batch)
    
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
#         return optimizer
    
#     def get_criterion(self, method, forecast_task, dist_side, tau_pinball):
#         if forecast_task == 'point':
#             criterion = nn.MSELoss()
#         elif forecast_task == 'quantile':
#             criterion = lambda pred, target: pinball_loss(pred, target, tau_pinball, dist_side)
#         elif forecast_task == 'expectile':
#             criterion = lambda pred, target: pinball_loss_expectile(pred, target, tau_pinball, dist_side)
#         else:
#             raise ValueError("distribution criterion is created in __init__")
    
#     @staticmethod
#     def add_model_specific_args(parent_parser):
#         model_parser = parent_parser.add_argument_group('iTransformer')
#         # Embedding
#         model_parser.add_argument('--embed', type=str, default='fixed')
#         model_parser.add_argument('--freq', type=str, default='h')

        
#         # Properties of Attn layer
#         model_parser.add_argument('--d_model', type=int, default=2048)
#         model_parser.add_argument('--factor', type=int, default=3)
#         model_parser.add_argument('--n_heads', type=int, default=3)
#         model_parser.add_argument('--d_ff', type=int, default=1024)
#         model_parser.add_argument('--activation', type=str, default='gelu')

#         model_parser.add_argument('--forecast_task', type=str, default = 'quantile', choices=['quantile', 'point', 'expectile', 'distribution'], help='quantile, expectile or point forecasting')
#         model_parser.add_argument('--dist_side', type=str, default='both', choices=['both', 'up', 'down'], help='side of the distribution to be predicted (for quantile/expectile forecasting)')
#         model_parser.add_argument('--tau_pinball', type=float, help='tau parameter for pinball loss (quantile/expectile regression)', default=0.05)
#         model_parser.add_argument('--n_cheb', type=int, default=2, help='number of Chebyshev polynomials for distribution forecasting')
        
#         model_parser.add_argument('--dist_loss', type=str, default='twcrps', choices=['crps','twcrps'])
#         model_parser.add_argument('--twcrps_threshold_low', type=float, default=-10.0)  # for price target
#         model_parser.add_argument('--twcrps_threshold_high', type=float, default=10.0)  # for price target
#         model_parser.add_argument('--twcrps_side', type=str, default='two_sided', choices=['below','above', 'two_sided'])
#         model_parser.add_argument('--twcrps_smooth_h', type=float, default=2)
#         model_parser.add_argument('--u_grid_size', type=int, default=256)

#         model_parser.add_argument('--dropout', type=float,default=0.1)
#         model_parser.add_argument('--e_layers', type=int, default=2)

#         model_parser.add_argument('--scaler', type=str,default='revin')
#         model_parser.add_argument('--affine', type=int, choices = [0,1], default=1)

#         model_parser.add_argument('--learning_rate', type=float,default=0.0001)
#         return parent_parser

