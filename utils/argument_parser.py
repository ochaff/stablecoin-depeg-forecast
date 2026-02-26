import argparse 
from utils.build_dataset import add_dataset_args
def parse_arguments():
    parser = argparse.ArgumentParser(description='main file arguments')

    parser = add_dataset_args(parser)

    data = parser.add_argument_group('Data Loading arguments')
    data.add_argument('-b','--build_dataset',action='store_false', help='whether to build the dataset or not')
    data.add_argument('--seq_len', type=int, default=24)
    data.add_argument('--label_len', type=int, default=0)
    data.add_argument('--pred_len',type=int, default=6)
    data.add_argument('--val_split', type=float, default=0.7)
    data.add_argument('--test_split', type=float, default=0.85)
    data.add_argument('--batch_size', type=int, default=3)
    data.add_argument('--test_batch_size', type=int, default=1)
    data.add_argument('--scale_pos', action='store_true', help='whether to scale positive samples for BCEWithLogitsLoss')
    data.add_argument('--scaler_type', type=str, default=None, choices=[None, 'standard', 'robust'], help='type of scaler to use for data normalization')

    model = parser.add_argument_group('Model training arguments')
    model.add_argument('--remote_logging', action='store_true', help='whether to log to a remote MLFlow server')
    model.add_argument('--experiment_name', type=str, default='stablecoin-depeg', help='name of the MLFlow experiment')
    model.add_argument('--run_name', type=str, default=None, help='name of the MLFlow run')
    model.add_argument('--model_root_path', type=str, default='./models', help='root path of the pytorch models')
    model.add_argument('--method', type = str, default='forecast', choices=['forecast', 'earlywarning'], help='forecasting or early warning classification task')
    model.add_argument('--model_name', type=str, choices=['iTransformer', 'TSMixer', 'CNN', 'TimeXer'], default='iTransformer', help='name of the model to be trained')
    model.add_argument('--n_epochs', type=int, default=50, help = 'number of epochs for training')
    model.add_argument('--patience', type=int, default=10, help='number of epochs with no improvement after which training will be stopped')
    model.add_argument('--verbose', type=int, default=1, help='verbosity level for training')
    model.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate for training')
    model.add_argument('--check_lr', action='store_true', help='whether to check learning rate or not')
    model.add_argument('--validation_metric', type=str, default='loss', choices = ['loss','auc','auprc'], help='metric to monitor for validation during training')

    return parser
