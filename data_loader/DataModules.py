import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from data_loader.Datasets import Dataset_forecast, Dataset_earlywarning
from sklearn.preprocessing import StandardScaler, RobustScaler



class DataModule_forecast(L.LightningDataModule):
    def __init__(self, dataset_path, batch_size, test_split, val_split, seq_len, pred_len, label_len, test_batch_size, **kwargs):
        super().__init__()
        self.dataset_path = dataset_path
        self.input_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.batch_size = batch_size 
        self.test_batch_size = test_batch_size 
        self.split = test_split
        self.splitval = val_split
        print(self.splitval)
        print(self.split)
        # self.save_hyperparameters()
    def setup(self, stage : str):
        if stage == 'fit' :
            self.dataset_train = Dataset_forecast(self.dataset_path, flag = 'train', size = [self.input_len, self.pred_len, self.label_len],
                                                      split = self.split, splitval=self.splitval, scaler = None,
                                                      )
            self.dataset_val = Dataset_forecast(self.dataset_path, flag = 'val', size = [self.input_len, self.pred_len, self.label_len], 
                                                   split = self.split,  splitval=self.splitval, scaler = None, 
                                                   )
        if stage == 'test' :
            self.dataset_test = Dataset_forecast(self.dataset_path, flag = 'test', size = [self.input_len, self.pred_len, self.label_len],
                                                     split = self.split,  splitval=self.splitval, scaler = None,
                                                     )
        
    def train_dataloader(self):
        return DataLoader(
        self.dataset_train,
        batch_size = self.batch_size,
        shuffle = True,
        num_workers = 3,
        drop_last = True)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.test_batch_size, shuffle=False, num_workers=2)
        
    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.test_batch_size, shuffle=False, num_workers=2)

class DataModule_earlywarning(L.LightningDataModule):
    def __init__(self, dataset_path, batch_size, test_split, val_split, seq_len, pred_len, label_len, test_batch_size, scale_pos, scaler_type=None, **kwargs):
        super().__init__()
        self.dataset_path = dataset_path
        self.input_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.scale_pos = scale_pos
        self.batch_size = batch_size 
        self.test_batch_size = test_batch_size 
        self.split = test_split
        self.splitval = val_split
        print(self.splitval)
        print(self.split)
        if scaler_type is None:
            self.scaler = None
        elif scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()


        if self.scale_pos:
            dataset_train = Dataset_earlywarning(self.dataset_path, flag = 'train', size = [self.input_len, self.pred_len, self.label_len],
                                                      split = self.split, splitval=self.splitval, scaler = self.scaler, fit_scaler = True,
                                                      )
            y = np.asarray(dataset_train.data_y).reshape(-1)  

            n_pos = (y == 1).sum()
            n_neg = (y == 0).sum()

            self.pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        else:
            self.pos_weight = None

        # self.save_hyperparameters()
    def setup(self, stage : str):
        if stage == 'fit' :
            self.dataset_train = Dataset_earlywarning(self.dataset_path, flag = 'train', size = [self.input_len, self.pred_len, self.label_len],
                                                      split = self.split, splitval=self.splitval, scaler = self.scaler, fit_scaler = True,
                                                      )
                
            self.dataset_val = Dataset_earlywarning(self.dataset_path, flag = 'val', size = [self.input_len, self.pred_len, self.label_len], 
                                                   split = self.split,  splitval=self.splitval, scaler = self.scaler, fit_scaler = False,
                                                   )
        if stage == 'test' :
            self.dataset_test = Dataset_earlywarning(self.dataset_path, flag = 'test', size = [self.input_len, self.pred_len, self.label_len],
                                                     split = self.split,  splitval=self.splitval, scaler = self.scaler, fit_scaler = False,
                                                     )
        
    def train_dataloader(self):
        return DataLoader(
        self.dataset_train,
        batch_size = self.batch_size,
        shuffle = True,
        num_workers = 3,
        drop_last = True)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.test_batch_size, shuffle=False, num_workers=2)
        
    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.test_batch_size, shuffle=False, num_workers=2)