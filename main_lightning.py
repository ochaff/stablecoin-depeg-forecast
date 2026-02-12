import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from dotenv import load_dotenv
from lightning.pytorch.tuner import Tuner
import mlflow
from models.iTransformer import iTransformer_forecast, iTransformer_classifier
from models.TSMixer import TSMixer_forecast
from utils.argument_parser import parse_arguments
from utils.build_dataset import build_dataset
from data_loader.DataModules import DataModule_forecast, DataModule_earlywarning

load_dotenv()

CUDA_LAUNCH_BLOCKING=1.


if __name__ == "__main__":
    seed_everything(1233)
    parser = parse_arguments()
    temp_args = parser.parse_known_args()
    dict_args = vars(temp_args[0])
    if temp_args[0].method == 'earlywarning':
        dict_args['target'] = True
    elif temp_args[0].method == 'forecast':
        dict_args['target'] = False
    final_dataset_path = build_dataset(**dict_args)
    if temp_args[0].remote_logging:
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

    model_dict = {
        'iTransformer':[iTransformer_forecast, iTransformer_classifier],
        'TSMixer':[TSMixer_forecast, None]
    }
    if temp_args[0].method == "forecast":
        model = model_dict[temp_args[0].model_name][0]
    elif temp_args[0].method == "earlywarning":
        model = model_dict[temp_args[0].model_name][1]

    model.add_model_specific_args(parser)
    args = parser.parse_args()
    args.dataset_path = final_dataset_path
    if args.method == 'forecast':
        args.enc_in = pd.read_parquet(args.dataset_path).shape[1]
    elif args.method == 'earlywarning':
        args.enc_in = pd.read_parquet(args.dataset_path).shape[1] - 1 

    dict_args = vars(args)
    LModel = model(**dict_args)
    if args.method == 'forecast':
        data = DataModule_forecast(**dict_args)
    elif args.method == 'earlywarning':
        data = DataModule_earlywarning(**dict_args)
    if args.remote_logging:
        logger = MLFlowLogger(experiment_name = args.experiment_name, run_name = args.run_name, tracking_uri = os.getenv('MLFLOW_TRACKING_URI'), artifact_location = os.getenv('ARTIFACT_URI'), log_model = True)
    else:
        logger = TensorBoardLogger(save_dir='lightning_logs', name=args.experiment_name, version=args.run_name)
        
    checkpointing = ModelCheckpoint(monitor = 'val_loss', save_top_k = 1, mode= 'min')
    trainer = L.Trainer(
                devices = 1, accelerator = 'gpu',  max_epochs = args.n_epochs, logger = logger, deterministic = 'warn', 
                callbacks = [EarlyStopping(monitor = 'val_loss', patience = args.patience, verbose = args.verbose), checkpointing, LearningRateMonitor(logging_interval='epoch')],
                detect_anomaly = False,log_every_n_steps = 20
                )
    if args.check_lr == 1 :
                res = Tuner(trainer).lr_find(
                LModel, datamodule = data, min_lr=1e-5, max_lr=1e-2
                )
                print(f"suggested learning rate: {res.suggestion()}")
                fig = res.plot(show=True, suggest=True)
                fig.savefig('./figs/lr_search.png')
                LModel.hparams.learning_rate = res.suggestion()
    trainer.fit(LModel, data)
    trainer.test(LModel, data, ckpt_path="best")