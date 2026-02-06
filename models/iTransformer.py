import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
import lightning as L
import pickle as pkl
import argparse
from models.common import RevIN
from einops import rearrange
from torch.distributions import Normal, StudentT
from neuralforecast.losses.pytorch import DistributionLoss, sCRPS
from torch.optim.lr_scheduler import CosineAnnealingLR



class Model(nn.Module):

    def __init__(self, seq_len, pred_len, d_model, d_ff, dropout, e_layers, activation, embed, freq, n_heads, factor, enc_in, exogenous,
                method = 'forecast', forecast_task = 'quantile', dist_side = 'both',
                affine = True, scaler= 'revin'):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.exogenous = exogenous
        self.method = method
        self.enc_in = enc_in
        self.revin = RevIN(self.enc_in, affine = affine, mode=scaler)
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
        else:
            self.projection = nn.Linear(d_model, self.pred_len)   
        self.classify = nn.Linear(d_model, 1)


    def forecast(self, x_enc):

        _, _, N = x_enc.shape
        if self.exogenous > 0:
            x_exo = x_enc[:,:,self.enc_in:]
            x_enc = x_enc[:,:,:self.enc_in]
            _, _, N = x_enc.shape
            x_enc = self.revin(x_enc, 'norm')
            self.ms = x_exo.mean(1, keepdim=True).detach()
            x_exo = x_exo - self.ms
            self.sv = torch.sqrt(torch.var(x_exo, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_exo /= self.sv
            x_enc = torch.concat((x_enc, x_exo), dim = -1)
        else:
            x_enc = self.revin(x_enc, 'norm')

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, self.enc_in - 1]
        dec_out = dec_out.view(dec_out.size(0), self.pred_len, -1)
        return dec_out
    
    def earlywarning(self, x_enc):
        _, _, N = x_enc.shape
        if self.exogenous > 0:
            x_exo = x_enc[:,:,self.enc_in:]
            x_enc = x_enc[:,:,:self.enc_in]
            _, _, N = x_enc.shape
            x_enc = self.revin(x_enc, 'norm')
            self.ms = x_exo.mean(1, keepdim=True).detach()
            x_exo = x_exo - self.ms
            self.sv = torch.sqrt(torch.var(x_exo, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_exo /= self.sv
            x_enc = torch.concat((x_enc, x_exo), dim = -1)
        else:
            x_enc = self.revin(x_enc, 'norm')

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        class_out = torch.sigmoid(self.classify(enc_out[:, -1, :]))
        return class_out

    def forward(self, x_enc):
        if self.method == 'forecast':
            dec_out = self.forecast(x_enc)
            dec_out = dec_out[:, -self.pred_len:, :]  
            return dec_out
        if self.method == 'earlywarning':
            class_out = self.earlywarning(x_enc)
            return class_out
        

class iTransformer(L.LightningModule):
    def __init__(self,
                seq_len, pred_len, batch_size, test_batch_size,
                learning_rate, method,
                embed, freq, dropout, e_layers,
                d_model, factor, n_heads, d_ff, activation, enc_in,
                exogenous, affine, scaler, forecast_task, dist_side, tau_pinball,
                **kwargs
                ):
        super().__init__()
        # Save hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.model = Model(seq_len, pred_len, d_model, d_ff, dropout, e_layers, activation, embed, freq, n_heads, factor, enc_in, exogenous, method, forecast_task, dist_side, affine, scaler)
        
        self.criterion= self.get_criterion(method, forecast_task, dist_side, tau_pinball)
        self.method = method
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch    
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        outputs = self.model(batch_x)
    
        loss = self.criterion(outputs, batch_y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
            
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        outputs = self.model(batch_x)
        
        loss = self.criterion(outputs, batch_y)
        self.log('val_loss', loss)
    
    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        outputs = self.model(batch_x)
        test_loss = self.criterion(outputs, batch_y)
        self.log('test_loss', test_loss)

        if batch_idx == 0:
            self.x_test = batch_x.detach().cpu()
            self.y_test = batch_y.detach().cpu()
            self.y_pred = outputs.detach().cpu()
        else:
            self.x_test =  torch.cat((self.x_test,batch_x.detach().cpu()), dim=0)
            self.y_test =  torch.cat((self.y_test,batch_y.detach().cpu()), dim=0)
            self.y_pred = torch.cat((self.y_pred,outputs.detach().cpu()), dim=0)
    
    def on_test_epoch_end(self):
        '''
        Saves predictions in mlflow artifacts
        '''
        A = {'true' : np.array(self.y_test), 'pred' : np.array(self.y_pred), 'seq' : np.array(self.x_test)}
        pkl.dump(A, open(f'./{self.logger.experiment_id}/{self.logger.run_id}/preds_test_set.pkl', 'wb'))
        self.logger.experiment.log_artifact(self.logger.run_id, f'./{self.logger.experiment_id}/{self.logger.run_id}/preds_test_set.pkl')

    def predict_step(self, batch, batch_idx):
        return self.model(batch)
    
    def forward(self, batch, batch_idx):
        return self.model(batch)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def get_criterion(self, method, forecast_task, dist_side, tau_pinball):
        if method == 'forecast':
            if forecast_task == 'point':
                criterion = nn.MSELoss()
            elif forecast_task == 'quantile':
                criterion = lambda pred, target: self.pinball_loss(pred, target, tau_pinball, dist_side)
            elif forecast_task == 'expectile':
                criterion = lambda pred, target: self.pinball_loss_expectile(pred, target, tau_pinball, dist_side)
        elif method == 'earlywarning':
            criterion = nn.BCELoss()
        return criterion
    def pinball_loss_expectile(self, pred, target, tau, dist_side):
        if dist_side == 'up':
            r = (target - pred)
            w = torch.where(r >= 0, torch.as_tensor(tau, device=pred.device, dtype=pred.dtype),
                                torch.as_tensor(1.0 - tau, device=pred.device, dtype=pred.dtype))
            loss = w * r.pow(2)
            loss = torch.mean(loss)
        elif dist_side == 'down':
            r = (pred - target)
            w = torch.where(r >= 0, torch.as_tensor(tau, device=pred.device, dtype=pred.dtype),
                                torch.as_tensor(1.0 - tau, device=pred.device, dtype=pred.dtype))
            loss = w * r.pow(2)
            loss = torch.mean(loss)
        elif dist_side == 'both':
            pred_down = pred[:, :, 0]
            pred_up = pred[:, :, 1]
            r_down = (pred_down - target)
            w_down = torch.where(r_down >= 0, torch.as_tensor(tau, device=pred.device, dtype=pred.dtype),
                                torch.as_tensor(1.0 - tau, device=pred.device, dtype=pred.dtype))
            loss_down = w_down * r_down.pow(2)
            r_up = (target - pred_up)
            w_up = torch.where(r_up >= 0, torch.as_tensor(tau, device=pred.device, dtype=pred.dtype),
                                torch.as_tensor(1.0 - tau, device=pred.device, dtype=pred.dtype))
            loss_up = w_up * r_up.pow(2)
            loss = torch.mean(loss_down + loss_up)
        return loss
    
    def pinball_loss(self, pred, target, tau, dist_side):
        if dist_side == 'both':
            pred_down = pred[:, :, 0]
            pred_up = pred[:, :, 1]
            loss_up = torch.max((tau * (target - pred_up)), ((tau - 1) * (target - pred_up)))
            loss_down = torch.max((tau * (pred_down - target)), ((tau - 1) * (pred_down - target)))
            loss = torch.mean(loss_up + loss_down)
        elif dist_side == 'up':
            loss = torch.max((tau * (target - pred)), ((tau - 1) * (target - pred)))
            loss = torch.mean(loss)
        elif dist_side == 'down':
            loss = torch.max((tau * (pred - target)), ((tau - 1) * (pred - target)))
            loss = torch.mean(loss)
        return loss




    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = parent_parser.add_argument_group('iTransformer')
        # Embedding
        model_parser.add_argument('--embed', type=str, default='fixed')
        model_parser.add_argument('--freq', type=str, default='M')

        
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

        model_parser.add_argument('--learning_rate', type=float,default=0.0001)
        model_parser.add_argument('--quantile', type=float,default=0)
        return parent_parser