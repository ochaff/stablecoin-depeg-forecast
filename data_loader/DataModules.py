import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from data_loader.Datasets import Dataset_forecast, Dataset_earlywarning



class DataModule_forecast(L.LightningDataModule):
    def __init__(self, dataset_path, batch_size, test_split, val_split, seq_len, pred_len, label_len, test_batch_size, validation, exogenous, **kwargs):
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
        self.validation = validation
        self.exogenous = exogenous
        # self.save_hyperparameters()
    def setup(self, stage : str):
        if stage == 'fit' :
            self.dataset_train = Dataset_forecast(self.dataset_path, flag = 'train', size = [self.input_len, self.pred_len, self.label_len],
                                                      split = self.split, splitval=self.splitval, scaler = None,
                                                    exogenous=self.exogenous
                                                      )
            self.dataset_val = Dataset_forecast(self.dataset_path, flag = 'val', size = [self.input_len, self.pred_len, self.label_len], 
                                                   split = self.split,  splitval=self.splitval, scaler = None, 
                                                   exogenous=self.exogenous
                                                   )
        if stage == 'test' :
            self.dataset_test = Dataset_forecast(self.dataset_path, flag = 'test', size = [self.input_len, self.pred_len, self.label_len],
                                                     split = self.split,  splitval=self.splitval, scaler = None,
                                                    exogenous=self.exogenous
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
    def __init__(self, dataset_path, batch_size, test_split, val_split, seq_len, pred_len, label_len, test_batch_size, validation, exogenous, **kwargs):
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
        self.validation = validation
        self.exogenous = exogenous
        # self.save_hyperparameters()
    def setup(self, stage : str):
        if stage == 'fit' :
            self.dataset_train = Dataset_earlywarning(self.dataset_path, flag = 'train', size = [self.input_len, self.pred_len, self.label_len],
                                                      split = self.split, splitval=self.splitval, scaler = None,
                                                    exogenous=self.exogenous
                                                      )
            self.dataset_val = Dataset_earlywarning(self.dataset_path, flag = 'val', size = [self.input_len, self.pred_len, self.label_len], 
                                                   split = self.split,  splitval=self.splitval, scaler = None, 
                                                   exogenous=self.exogenous
                                                   )
        if stage == 'test' :
            self.dataset_test = Dataset_earlywarning(self.dataset_path, flag = 'test', size = [self.input_len, self.pred_len, self.label_len],
                                                     split = self.split,  splitval=self.splitval, scaler = None,
                                                    exogenous=self.exogenous
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