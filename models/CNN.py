import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from models.common import RevIN, Baseclass_forecast, Baseclass_earlywarning

class CNN(nn.Module):
    def __init__(self, seq_len, enc_in, d_ff, kernel_size, dropout=0.1,method='earlywarning', scaler ='revin', activation='relu'):
        super(CNN, self).__init__()
        self.method = method
        self.scaler = scaler
        self.conv1 = nn.Conv1d(in_channels = enc_in, out_channels = d_ff, kernel_size = kernel_size)
        self.avgpool1 = nn.AvgPool1d(kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels = d_ff, out_channels = d_ff // 2, kernel_size = kernel_size//2)
        self.avgpool2 = nn.AvgPool1d(kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels = d_ff // 2, out_channels = d_ff // 4, kernel_size = kernel_size//4)

        if activation == 'relu':
            self.activ = nn.ReLU()
        elif activation == 'gelu':
            self.activ = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

        # Fully connected layer
        self.fc_input_size = d_ff // 4 * (((seq_len - kernel_size + 1) // 4 - kernel_size//2 + 1) // 3 - kernel_size//4 + 1)
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1) 

    def classify(self, x):
        x = x.permute(0, 2, 1)  
        x = self.conv1(x)
        x = self.activ(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.activ(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.activ(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # x = self.sigmoid(x)
        return x

    def forward(self, x):
        if self.method == 'earlywarning':
            return self.classify(x)
        elif self.method == 'forecast':
            raise NotImplementedError("CNN is not implemented for forecasting yet.")
    

class CNN_earlywarning(Baseclass_earlywarning):
    def __init__(self,
                seq_len,
                enc_in,
                batch_size,
                test_batch_size,
                learning_rate,
                class_loss,
                compute_shap,
                shap_background_size,
                shap_test_samples,
                focal_alpha,
                focal_gamma,
                d_ff,
                kernel_size,
                activation,
                pos_weight = None,
                dropout=0.1,
                scaler='revin',
                **kwargs):
        super(CNN_earlywarning, self).__init__(batch_size=batch_size, test_batch_size=test_batch_size, learning_rate=learning_rate, class_loss=class_loss,
                                               compute_shap=compute_shap, shap_background_size=shap_background_size, shap_test_samples=shap_test_samples,
                                               focal_alpha=focal_alpha, focal_gamma=focal_gamma, pos_weight=pos_weight
                                               )            
        self.model = CNN(seq_len=seq_len, enc_in=enc_in, d_ff=d_ff, kernel_size=kernel_size, dropout=dropout, method='earlywarning', scaler=scaler, activation=activation)
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = parent_parser.add_argument_group('Model-specific arguments')
        model_parser.add_argument('--d_ff', type=int, default=128)
        model_parser.add_argument('--kernel_size', type=int, default=12)
        model_parser.add_argument('--dropout', type=float, default=0.2)
        model_parser.add_argument('--scaler', type=str, default='revin')
        model_parser.add_argument('--activation', type=str, choices=['relu', 'gelu'], default='gelu')
        Baseclass_earlywarning.add_task_specific_args(parent_parser)
        return parent_parser