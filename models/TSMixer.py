import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from models.common import RevIN, Baseclass_forecast, Baseclass_earlywarning

class TemporalMixing(nn.Module):
    """
    TemporalMixing
    """

    def __init__(self, n_series, input_size, dropout):
        super().__init__()
        self.temporal_norm = nn.BatchNorm1d(
            num_features=n_series * input_size, eps=0.001, momentum=0.01
        )
        self.temporal_lin = nn.Linear(input_size, input_size)
        self.temporal_drop = nn.Dropout(dropout)

    def forward(self, input):
        # Get shapes
        batch_size = input.shape[0]
        input_size = input.shape[1]
        n_series = input.shape[2]

        # Temporal MLP
        x = input.permute(0, 2, 1)  # [B, L, N] -> [B, N, L]
        x = x.reshape(batch_size, -1)  # [B, N, L] -> [B, N * L]
        x = self.temporal_norm(x)  # [B, N * L] -> [B, N * L]
        x = x.reshape(batch_size, n_series, input_size)  # [B, N * L] -> [B, N, L]
        x = F.relu(self.temporal_lin(x))  # [B, N, L] -> [B, N, L]
        x = x.permute(0, 2, 1)  # [B, N, L] -> [B, L, N]
        x = self.temporal_drop(x)  # [B, L, N] -> [B, L, N]

        return x + input


class FeatureMixing(nn.Module):
    """
    FeatureMixing
    """

    def __init__(self, n_series, input_size, dropout, ff_dim):
        super().__init__()
        self.feature_norm = nn.BatchNorm1d(
            num_features=n_series * input_size, eps=0.001, momentum=0.01
        )
        self.feature_lin_1 = nn.Linear(n_series, ff_dim)
        self.feature_lin_2 = nn.Linear(ff_dim, n_series)
        self.feature_drop_1 = nn.Dropout(dropout)
        self.feature_drop_2 = nn.Dropout(dropout)

    def forward(self, input):
        # Get shapes
        batch_size = input.shape[0]
        input_size = input.shape[1]
        n_series = input.shape[2]

        # Feature MLP
        x = input.reshape(batch_size, -1)  # [B, L, N] -> [B, L * N]
        x = self.feature_norm(x)  # [B, L * N] -> [B, L * N]
        x = x.reshape(batch_size, input_size, n_series)  # [B, L * N] -> [B, L, N]
        x = F.relu(self.feature_lin_1(x))  # [B, L, N] -> [B, L, ff_dim]
        x = self.feature_drop_1(x)  # [B, L, ff_dim] -> [B, L, ff_dim]
        x = self.feature_lin_2(x)  # [B, L, ff_dim] -> [B, L, N]
        x = self.feature_drop_2(x)  # [B, L, N] -> [B, L, N]

        return x + input


class MixingLayer(nn.Module):
    """
    MixingLayer
    """

    def __init__(self, n_series, input_size, dropout, ff_dim):
        super().__init__()
        # Mixing layer consists of a temporal and feature mixer
        self.temporal_mixer = TemporalMixing(n_series, input_size, dropout)
        self.feature_mixer = FeatureMixing(n_series, input_size, dropout, ff_dim)

    def forward(self, input):
        x = self.temporal_mixer(input)
        x = self.feature_mixer(x)
        return x

class Model(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, dropout, n_layers,
                method, forecast_task = None, dist_side = None,
                enc_in = None,
                affine = True, scaler = 'revin', n_cheb =2):
        super(Model, self).__init__()
        self.n_layers = n_layers
        self.enc_in = enc_in
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.method = method
        self.revin = RevIN(self.enc_in, affine = affine, mode=scaler)
        mixing_layers = [
            MixingLayer(
                n_series=self.enc_in, input_size=self.seq_len, dropout=dropout, ff_dim=d_model
            )
            for _ in range(self.n_layers)
        ]
        self.mixing_layers = nn.Sequential(*mixing_layers)
        if dist_side == 'both' and forecast_task in ['quantile', 'expectile']:
            self.projection = nn.Linear(self.seq_len, self.pred_len * 2)
        elif (dist_side in ['up', 'down'] and forecast_task in ['quantile', 'expectile']) or forecast_task == 'point':
            self.projection = nn.Linear(self.seq_len, self.pred_len)
        elif forecast_task == 'distribution':
            self.projection = nn.Linear(self.seq_len, self.pred_len * (2+n_cheb))
        self.classify_time = nn.Linear(seq_len, 1)
        self.classify_features = nn.Linear(enc_in, 1)

    def forecast(self, x_enc):
        x_enc = self.revin(x_enc, 'norm')

        x_enc = self.mixing_layers(x_enc)

        enc_out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)
        dec_out = self.revin(enc_out, 'denorm')
        dec_out = dec_out[:,:,-1]
        return dec_out
    
    def earlywarning(self, x_enc):
        x_enc = self.mixing_layers(x_enc)
        enc_out = self.classify_time(x_enc.transpose(1, 2)).transpose(1, 2)
        dec_out = self.classify_features(enc_out)
        # dec_out = torch.sigmoid(dec_out)
        return dec_out
    
    def forward(self, x_enc):
        if self.method == 'forecast':
            dec_out = self.forecast(x_enc)
            dec_out = torch.squeeze(dec_out.view(dec_out.shape[0], self.pred_len, -1), dim = -1)
            return dec_out
        elif self.method == 'earlywarning':
            classify_out = self.earlywarning(x_enc).squeeze(dim = -1)
            return classify_out
  

class TSMixer_forecast(Baseclass_forecast):
    def __init__(self, 
                seq_len, pred_len, d_model, dropout,
                n_layers,
                enc_in, method, batch_size, test_batch_size, affine, scaler,
                forecast_task, dist_side, tau_pinball,
                n_cheb, twcrps_threshold_low, twcrps_threshold_high, twcrps_side, 
                twcrps_smooth_h, u_grid_size, dist_loss,
                **kwargs
                ):
        super(TSMixer_forecast, self).__init__(
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
            dist_loss=dist_loss,
        )
        self.model = Model(seq_len, pred_len, d_model, dropout, n_layers, method, forecast_task, dist_side, 
                           enc_in, affine, scaler, n_cheb,
                           )
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = parent_parser.add_argument_group('Model-specific arguments')
        model_parser.add_argument('--d_model', type=int, default=2048)
        model_parser.add_argument('--dropout', type=float,default=0.1)
        model_parser.add_argument('--n_layers', type=int, default=3)
        model_parser.add_argument('--scaler', type=str,default='revin')
        model_parser.add_argument('--affine', type=int, choices = [0,1], default=1)
        Baseclass_forecast.add_task_specific_args(parent_parser)
        return parent_parser
        
class TSMixer_earlywarning(Baseclass_earlywarning):
    def __init__(self, 
                seq_len, pred_len, d_model, dropout,
                n_layers, learning_rate,
                enc_in, method, batch_size, test_batch_size, affine, scaler, 
                compute_shap, shap_background_size, shap_test_samples,
                class_loss, focal_alpha, focal_gamma, pos_weight,
                **kwargs
                ):
        super(TSMixer_earlywarning, self).__init__(
            batch_size= batch_size,
            test_batch_size= test_batch_size,
            learning_rate = learning_rate,
            class_loss= class_loss,
            compute_shap=compute_shap,
            shap_background_size=shap_background_size,
            shap_test_samples=shap_test_samples,
            focal_alpha= focal_alpha,
            focal_gamma=focal_gamma,
            pos_weight=pos_weight,
        )
        self.model = Model(seq_len, pred_len, d_model, dropout, n_layers, method, forecast_task = None, dist_side = None,
                           enc_in=enc_in, affine=affine, scaler=scaler
                           )
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = parent_parser.add_argument_group('Model-specific arguments')
        model_parser.add_argument('--d_model', type=int, default=2048)
        model_parser.add_argument('--dropout', type=float,default=0.1)
        model_parser.add_argument('--n_layers', type=int, default=3)
        model_parser.add_argument('--scaler', type=str,default='revin')
        model_parser.add_argument('--affine', type=int, choices = [0,1], default=1)
        Baseclass_earlywarning.add_task_specific_args(parent_parser)
        return parent_parser
        
