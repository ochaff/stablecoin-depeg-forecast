from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import RevIN

class IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, out_features: int = 1):
        super().__init__()
        self.out_features = out_features
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        backcast = theta[:, : self.backcast_size]
        forecast = theta[:, self.backcast_size :]
        forecast = forecast.reshape(len(forecast), -1, self.out_features)
        return backcast, forecast


class TrendBasis(nn.Module):
    def __init__(
        self,
        degree_of_polynomial: int,
        backcast_size: int,
        forecast_size: int,
        out_features: int = 1,
    ):
        super().__init__()
        self.out_features = out_features
        polynomial_size = degree_of_polynomial + 1
        self.backcast_basis = nn.Parameter(
            torch.tensor(
                np.concatenate(
                    [
                        np.power(
                            np.arange(backcast_size, dtype=float) / backcast_size, i
                        )[None, :]
                        for i in range(polynomial_size)
                    ]
                ),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.forecast_basis = nn.Parameter(
            torch.tensor(
                np.concatenate(
                    [
                        np.power(
                            np.arange(forecast_size, dtype=float) / forecast_size, i
                        )[None, :]
                        for i in range(polynomial_size)
                    ]
                ),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        polynomial_size = self.forecast_basis.shape[0]  # [polynomial_size, L+H]
        backcast_theta = theta[:, :polynomial_size]
        forecast_theta = theta[:, polynomial_size:]
        forecast_theta = forecast_theta.reshape(
            len(forecast_theta), polynomial_size, -1
        )
        backcast = torch.einsum("bp,pt->bt", backcast_theta, self.backcast_basis)
        forecast = torch.einsum("bpq,pt->btq", forecast_theta, self.forecast_basis)
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    def __init__(
        self,
        harmonics: int,
        backcast_size: int,
        forecast_size: int,
        out_features: int = 1,
    ):
        super().__init__()
        self.out_features = out_features
        frequency = np.append(
            np.zeros(1, dtype=float),
            np.arange(harmonics, harmonics / 2 * forecast_size, dtype=float)
            / harmonics,
        )[None, :]
        backcast_grid = (
            -2
            * np.pi
            * (np.arange(backcast_size, dtype=float)[:, None] / forecast_size)
            * frequency
        )
        forecast_grid = (
            2
            * np.pi
            * (np.arange(forecast_size, dtype=float)[:, None] / forecast_size)
            * frequency
        )

        backcast_cos_template = torch.tensor(
            np.transpose(np.cos(backcast_grid)), dtype=torch.float32
        )
        backcast_sin_template = torch.tensor(
            np.transpose(np.sin(backcast_grid)), dtype=torch.float32
        )
        backcast_template = torch.cat(
            [backcast_cos_template, backcast_sin_template], dim=0
        )

        forecast_cos_template = torch.tensor(
            np.transpose(np.cos(forecast_grid)), dtype=torch.float32
        )
        forecast_sin_template = torch.tensor(
            np.transpose(np.sin(forecast_grid)), dtype=torch.float32
        )
        forecast_template = torch.cat(
            [forecast_cos_template, forecast_sin_template], dim=0
        )

        self.backcast_basis = nn.Parameter(backcast_template, requires_grad=False)
        self.forecast_basis = nn.Parameter(forecast_template, requires_grad=False)

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        harmonic_size = self.forecast_basis.shape[0]  # [harmonic_size, L+H]
        backcast_theta = theta[:, :harmonic_size]
        forecast_theta = theta[:, harmonic_size:]
        forecast_theta = forecast_theta.reshape(len(forecast_theta), harmonic_size, -1)
        backcast = torch.einsum("bp,pt->bt", backcast_theta, self.backcast_basis)
        forecast = torch.einsum("bpq,pt->btq", forecast_theta, self.forecast_basis)
        return backcast, forecast

ACTIVATIONS = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]


class NBEATSBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """

    def __init__(
        self,
        input_size: int,
        n_theta: int,
        mlp_units: list,
        basis: nn.Module,
        dropout_prob: float,
        activation: str,
    ):
        """ """
        super().__init__()

        self.dropout_prob = dropout_prob

        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"
        activ = getattr(nn, activation)()

        hidden_layers = [
            nn.Linear(in_features=input_size, out_features=mlp_units[0][0])
        ]
        for layer in mlp_units:
            hidden_layers.append(nn.Linear(in_features=layer[0], out_features=layer[1]))
            hidden_layers.append(activ)

            if self.dropout_prob > 0:
                raise NotImplementedError("dropout")
                # hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        output_layer = [nn.Linear(in_features=mlp_units[-1][1], out_features=n_theta)]
        layers = hidden_layers + output_layer
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(self, insample_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute local projection weights and projection
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta)
        return backcast, forecast

class NBEATS(nn.Module):

    def __init__(
        self,
        h,
        input_size,
        n_harmonics: int = 2,
        n_polynomials: int = 2,
        stack_types: list = ["identity", "trend", "seasonality"],
        n_blocks: list = [1, 1, 1],
        mlp_units: list = 3 * [[512, 512]],
        dropout_prob_theta: float = 0.0,
        activation: str = "ReLU",
        shared_weights: bool = False,
        affine: bool = True,
        scaler: str = 'revin',
        enc_in: int = 26,
        method : str = 'HierE2E',
        S_mat = None,
        n_components : int = 5,
        n_high : int = 26,
        num_blocks : int = 5,
        n_samples : int = 200,
        criterion : str = " ",
        decompose : bool = False,
    ):
        super().__init__()
        self.enc_in = enc_in
        self.revin = RevIN(self.enc_in, affine = affine, mode=scaler)
        self.outputsize_multiplier = 2
            
        self.h = h
        self.S_mat = S_mat
        self.input_size = input_size
        self.method = method
        # Architecture
        blocks = self.create_stack(
            h=h,
            input_size=input_size,
            stack_types=stack_types,
            n_blocks=n_blocks,
            mlp_units=mlp_units,
            dropout_prob_theta=dropout_prob_theta,
            activation=activation,
            shared_weights=shared_weights,
            n_polynomials=n_polynomials,
            n_harmonics=n_harmonics,
        )
        self.blocks = torch.nn.ModuleList(blocks)
        self.decompose_forecast = decompose
        self.std_activ = nn.Softplus()
    def create_stack(
        self,
        stack_types,
        n_blocks,
        input_size,
        h,
        mlp_units,
        dropout_prob_theta,
        activation,
        shared_weights,
        n_polynomials,
        n_harmonics,
    ):

        block_list = []
        for i in range(len(stack_types)):
            for block_id in range(n_blocks[i]):
                if shared_weights and block_id > 0:
                    nbeats_block = block_list[-1]
                else:
                    if stack_types[i] == "seasonality":
                        n_theta = (
                            2
                            * (self.outputsize_multiplier + 1)
                            * int(np.ceil(n_harmonics / 2 * h) - (n_harmonics - 1))
                        )
                        basis = SeasonalityBasis(
                            harmonics=n_harmonics,
                            backcast_size=input_size,
                            forecast_size=h,
                            out_features=self.outputsize_multiplier,
                        )

                    elif stack_types[i] == "trend":
                        n_theta = (self.outputsize_multiplier + 1) * (
                            n_polynomials + 1
                        )
                        basis = TrendBasis(
                            degree_of_polynomial=n_polynomials,
                            backcast_size=input_size,
                            forecast_size=h,
                            out_features=self.outputsize_multiplier,
                        )

                    elif stack_types[i] == "identity":
                        n_theta = input_size + self.outputsize_multiplier * h
                        basis = IdentityBasis(
                            backcast_size=input_size,
                            forecast_size=h,
                            out_features=self.outputsize_multiplier,
                        )
                    else:
                        raise ValueError(f"Block type {stack_types[i]} not found!")

                    nbeats_block = NBEATSBlock(
                        input_size=input_size,
                        n_theta=n_theta,
                        mlp_units=mlp_units,
                        basis=basis,
                        dropout_prob=dropout_prob_theta,
                        activation=activation,
                    )

                # Select type of evaluation and apply it to all layers of block
                block_list.append(nbeats_block)

        return block_list

    def forward(self, batch_x):
        batch_x = self.revin(batch_x, 'norm')
        insample_y = torch.permute(batch_x, (0,2,1))
        batch_size = insample_y.shape[0]
        enc_in = insample_y.shape[1]
        insample_y = torch.reshape(insample_y, (batch_size*enc_in, -1))
        assert insample_y.shape[1] == self.input_size
        insample_mask = torch.ones_like(insample_y)

        # NBEATS' forward
        residuals = insample_y.flip(dims=(-1,))  # backcast init
        insample_mask = insample_mask.flip(dims=(-1,))

        forecast = insample_y[:, -1:, None]  # Level with Naive1
        block_forecasts = [forecast.repeat(1, self.h, 1)]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(insample_y=residuals)

            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast

            if self.decompose_forecast:
                block_forecasts.append(block_forecast)

        if self.decompose_forecast:
            # (n_batch, n_blocks, h, out_features)
            block_forecasts = torch.stack(block_forecasts)
            block_forecasts = block_forecasts.permute(1, 0, 2, 3)
            block_forecasts = block_forecasts.squeeze(-1)  # univariate output
            return block_forecasts
        else:
            forecast = torch.reshape(forecast, (batch_size, enc_in, self.h, self.outputsize_multiplier))
            forecast = torch.permute(forecast, (0,2,1,3))
            mu_out = forecast[:,:,:,0]
            if self.method != 'CNF':
                std_out = self.std_activ(forecast[:,:,:,1])
            if self.method == 'PROFHIT':
                mu_out, std_out = self.refine(mu_out, std_out)
                mu_2 = torch.einsum('iv ,blv->bli', self.S_mat, mu_out)
                sig_2 = torch.einsum('iv ,blv->bli', self.S_mat, torch.square(std_out))
                JFD = (1/2)*((torch.square(std_out) + torch.square(mu_out - mu_2))/(2*sig_2) + (sig_2 + torch.square(mu_out-mu_2))/(2*torch.square(std_out)) - 1 )
                mu_out = self.revin(mu_out, 'denorm')
                std_out = self.revin(std_out, 'denorm_scale')
                return mu_out, std_out, JFD
            elif self.method == 'HierE2E':
                mu_out = self.revin(mu_out, 'denorm')
                std_out = self.revin(std_out, 'denorm_scale')
                return mu_out, std_out
            elif self.method == 'DPMN':
                mu_out = forecast[:,:,:,self.n_components:]
                std_out = self.std_activ(forecast[:,:,:,:self.n_components])
                mu_out = mu_out.reshape(mu_out.shape[0], -1, self.enc_in).contiguous()
                mu_out = self.revin(mu_out, 'denorm')
                mu_out = mu_out.reshape(mu_out.shape[0], self.h,self.enc_in, self.n_components).contiguous()

                std_out = std_out.reshape(std_out.shape[0], -1, self.enc_in).contiguous()
                std_out = self.revin(std_out, 'denorm_scale')
                std_out = std_out.reshape(std_out.shape[0], self.h,self.enc_in, self.n_components).contiguous()
                return mu_out, std_out
            
            elif self.method == 'CLOVER':
                mu_out = self.revin(mu_out, 'denorm')
                std_out = self.revin(std_out, 'denorm_scale')
                factor_vec = forecast[:,:,:,2:]
                factor_vec = factor_vec.reshape(factor_vec.shape[0], -1, self.enc_in).contiguous()
                factor_vec = self.revin(factor_vec, 'denorm_scale')
                factor_vec = factor_vec.reshape(factor_vec.shape[0], self.h,self.enc_in, self.n_components).contiguous()
                factor_vec = factor_vec / torch.sqrt(torch.tensor(self.n_components))
                return mu_out, std_out, factor_vec
            
            elif self.method == 'CNF':
                forecast = torch.reshape(mu_out, (batch_size * self.h, enc_in)).repeat(self.n_samples, 1,1)
                rec_forecasts = self.CNF.sample(self.n_samples, 
                                                noise=Normal(torch.zeros(batch_size*self.h, enc_in - self.n_high) ,1.0).rsample((self.n_samples,)),
                                                cond_inputs = forecast) 
                
                rec_forecasts = rec_forecasts.reshape(self.n_samples, batch_size, self.h, enc_in - self.n_high)
                rec_forecasts = torch.concat((torch.zeros_like(rec_forecasts)[:,:,:,:self.n_high], rec_forecasts), dim = -1)
                rec_forecasts = self.revin(rec_forecasts,'denorm')
                rec_forecasts = rec_forecasts[:,:,:,self.n_high:]
                return rec_forecasts
