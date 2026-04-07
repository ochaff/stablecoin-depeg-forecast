from models.common import RevIN, Baseclass_forecast
import torch 
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import lightning as L


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
        

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
            return x


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Model(nn.Module):

    def __init__(self, seq_len, pred_len, patch_len, enc_in, d_model, d_ff,
                dropout, embed, freq, factor, n_heads, activation, e_layers,forecast_task = None, dist_side = None,
                affine = True, revin_type = 'revin', n_cheb =2):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.patch_num = int(seq_len // patch_len)
        self.n_vars = 1
        self.forecast_task = forecast_task
        self.dist_side = dist_side
        self.n_cheb = n_cheb
        self.std_activ = nn.Softplus()
        self.revin = RevIN(self.n_vars, affine = affine, mode=revin_type)
        # Embedding
        self.en_embedding = EnEmbedding(self.n_vars, d_model, self.patch_len, dropout)

        self.ex_embedding = DataEmbedding_inverted(seq_len, d_model, embed, freq,
                                                   dropout)
        

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout= dropout,
                                      output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.head_nf = d_model * (self.patch_num + 1)
        if dist_side == 'both' and forecast_task in ['quantile', 'expectile']:
            self.head = FlattenHead(enc_in, self.head_nf, pred_len * 2,
                                    head_dropout= dropout)
        elif (dist_side in ['up', 'down'] and forecast_task in ['quantile', 'expectile']) or forecast_task == 'point':
            self.head = FlattenHead(enc_in, self.head_nf, pred_len,
                                    head_dropout= dropout)
        elif forecast_task == 'distribution' :
            self.head = FlattenHead(enc_in, self.head_nf, pred_len * (2+n_cheb),
                                    head_dropout= dropout)
        elif forecast_task == 'gaussian' :
            self.head = FlattenHead(enc_in, self.head_nf, pred_len * 2,
                                    head_dropout= dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = self.revin(x_enc, mode='norm')

        en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        enc_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        enc_out = enc_out.transpose(1, 2)  # z: [bs x target_window x nvars]
        if self.forecast_task == "distribution":
            dec_out, std_out, cheb_out = torch.split(enc_out, [self.pred_len, self.pred_len, self.pred_len * self.n_cheb], dim=-2)
            dec_out = self.revin(dec_out, 'denorm')
            std_out = self.revin(std_out, 'denorm_scale')
            dec_out = dec_out[:,:,-1]
            std_out = std_out[:,:,-1]
            cheb_out = cheb_out[:,:,-1].view(cheb_out.shape[0], self.pred_len, self.n_cheb)

            std_out = self.std_activ(std_out)
            dec_out = torch.cat([dec_out.unsqueeze(-1), std_out.unsqueeze(-1), cheb_out], dim= -1)
        elif self.forecast_task == "gaussian":
            dec_out, std_out = torch.split(enc_out, [self.pred_len, self.pred_len], dim=-2)
            dec_out = self.revin(dec_out, 'denorm')
            std_out = self.revin(std_out, 'denorm_scale')
            dec_out = dec_out[:,:,-1]
            std_out = std_out[:,:,-1]
            std_out = self.std_activ(std_out)
            dec_out = torch.cat([dec_out.unsqueeze(-1), std_out.unsqueeze(-1)], dim= -1)
        else:
            dec_out = self.revin(enc_out, 'denorm')
            dec_out = dec_out[:,:,-1]
        return dec_out

    def forward(self, x_enc, x_mark_enc = None, x_dec = None, x_mark_dec = None, mask=None):   
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out
    


class TimeXer_forecast(Baseclass_forecast):
    def __init__(self, 
                seq_len, pred_len, d_model, dropout, patch_len, n_layers,
                enc_in, method, batch_size, test_batch_size, affine, revin_type,
                factor, n_heads,
                forecast_task, dist_side, tau_pinball,
                n_cheb, twcrps_threshold_low, twcrps_threshold_high, twcrps_side,
                twcrps_smooth_h, u_grid_size, dist_loss, grid_density, quantile_decomp, spline_degree, knot_kind, knot_p,
                **kwargs
                ):
        super(TimeXer_forecast, self).__init__(
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
            spline_degree=spline_degree,
            knot_kind=knot_kind,   
            knot_p=knot_p,
        )
        self.model = Model(seq_len=seq_len, pred_len=pred_len,patch_len= patch_len,
                        dropout=dropout, embed='fixed', freq='h', factor=factor, n_heads=n_heads, activation='gelu',
                        enc_in=enc_in, d_model=d_model, d_ff=d_model*4,
                        e_layers=n_layers, forecast_task=forecast_task, dist_side=dist_side, 
                        affine=affine, revin_type=revin_type, n_cheb=n_cheb
                        )
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = parent_parser.add_argument_group('Model-specific arguments')
        model_parser.add_argument('--d_model', type=int, default=2048)
        model_parser.add_argument('--dropout', type=float,default=0.1)
        model_parser.add_argument('--patch_len', type=int, default=12)
        model_parser.add_argument('--n_layers', type=int, default=3)
        model_parser.add_argument('--revin_type', type=str, choices = ['revin', 'robust'],default='revin')
        model_parser.add_argument('--affine', type=int, choices = [0,1], default=1)
        model_parser.add_argument('--factor', type=int, default=5)
        model_parser.add_argument('--n_heads', type=int, default=8)
        Baseclass_forecast.add_task_specific_args(parent_parser)
        return parent_parser
        