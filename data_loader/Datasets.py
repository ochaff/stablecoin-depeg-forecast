import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

class Dataset_forecast(Dataset) :
    def __init__(self, path, flag, size, split, splitval, scaler):
        super().__init__()
        self.path = path
        self.flag = flag
        self.seq_len = size[0]
        self.pred_len = size[1]
        self.label_len = size[2]
        self.split = split
        self.scaler = scaler
        self.splitval = splitval
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.__read_data__()

    def __read_data__(self):
        df = pd.read_parquet(self.path)

        if self.split < 1 :   
            self.split = int(self.split*df.shape[0])
            self.splitval = int(self.splitval*df.shape[0])
        else :
            self.split = int(self.split)
            self.splitval = int(self.splitval)
        
        if self.set_type == 0 :
            df = df.iloc[:self.splitval]
            if self.scaler != None :
                df = self.scaler.fit_transform(df)
        elif self.set_type == 1:
            df = df.iloc[self.splitval-self.seq_len:self.split]
            if self.scaler != None :
                df = self.scaler.fit_transform(df)
        elif self.set_type == 2 :
            df = df.iloc[self.split-self.seq_len:]
            if self.scaler != None :
                df = self.scaler.fit_transform(df)
        
        self.data_x = df.values 
        self.data_y = df['poolTick'].values

    def __getitem__(self, index):
        
        in_start = index
        in_end = index + self.seq_len
        out_start = in_end - self.label_len
        out_end = out_start + self.pred_len + self.label_len
        seq_x = self.data_x[in_start:in_end]
        seq_y = self.data_y[out_start:out_end]
        return seq_x, seq_y
        
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


class Dataset_earlywarning(Dataset) :
    def __init__(self, path, flag, size, split, splitval, scaler, fit_scaler=False):
        super().__init__()
        self.path = path
        self.flag = flag
        self.seq_len = size[0]
        self.split = split
        self.scaler = scaler
        self.fit_scaler = fit_scaler
        self.splitval = splitval
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.__read_data__()

    def __read_data__(self):
        df = pd.read_parquet(self.path)

        if self.split < 1 :   
            self.split = int(self.split*df.shape[0])
            self.splitval = int(self.splitval*df.shape[0])
        else :
            self.split = int(self.split)
            self.splitval = int(self.splitval)
        
        if self.set_type == 0 :
            df = df.iloc[:self.splitval]
        elif self.set_type == 1:
            df = df.iloc[self.splitval-self.seq_len:self.split]
        elif self.set_type == 2 :
            df = df.iloc[self.split-self.seq_len:]
        

        drop_cols = ["target"]
        feature_cols = [c for c in df.columns if c not in drop_cols]

        if self.scaler is not None:
            X = df[feature_cols].to_numpy(dtype=np.float32)

            if self.fit_scaler:
                self.scaler.fit(X)

            X = self.scaler.transform(X)
        else:
            X = df[feature_cols].to_numpy(dtype=np.float32)
        # Labels
        if self.set_type in (0, 1):  # train/val
            self.data_y = df["target"].to_numpy()
        else:             
            self.data_y = df[["target", "poolTick"]].to_numpy()
        self.data_x = X

    def __getitem__(self, index):
        in_start = index
        in_end = index + self.seq_len
        out_start = in_end 
        seq_x = self.data_x[in_start:in_end]
        seq_y = self.data_y[out_start]
        return seq_x, seq_y
        
    def __len__(self):
        return len(self.data_x) - self.seq_len

