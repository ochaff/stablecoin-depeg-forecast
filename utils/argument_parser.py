import argparse 

def parse_arguments():
    parser = argparse.ArgumentParser(description='main file arguments')

    dataset_building = parser.add_argument_group('Dataset building arguments')
    dataset_building.add_argument('--bypass', action='store_true', help='whether to bypass dataset building')
    dataset_building.add_argument('--dataset_path', type=str, default='./preprocessed_datasets', help='path to save the dataset')
    dataset_building.add_argument('--alpha', type=float, help='Gegenbauer polynomial alpha parameter', default=0.5)
    dataset_building.add_argument('--aave',action='store_false', help='remove AAVE metrics')
    dataset_building.add_argument('--aave_liq',action='store_false', help='remove AAVE liquidations')
    dataset_building.add_argument('--crv',action='store_false', help='remove Curve 3pool metrics')
    dataset_building.add_argument('--eth_price',action='store_false', help='remove ETH price oracle')
    dataset_building.add_argument('--eth_indicators',action='store_false', help='remove ETH price technical indicators')
    dataset_building.add_argument('--btc_price',action='store_false', help='remove BTC price oracle')
    dataset_building.add_argument('--btc_indicators',action='store_false', help='remove BTC price technical indicators')
    dataset_building.add_argument('--fear_greed',action='store_false', help='remove Fear and Greed index')
    dataset_building.add_argument('--gegen',action='store_false', help='remove Gegenbauer liquidity curve scores')
    class_target = parser.add_argument_group('Classification target arguments')
    class_target.add_argument('-w','--target_window', type=int, default=24, help='time window (in hours) for classification target')
    class_target.add_argument('-th','--target_threshold', type=int, default=25, help='threshold (in bps) for classification target')
    class_target.add_argument('-ds','--depeg_side', type=str, default='both', choices=['both', 'up', 'down'], help='depeg side for classification target')
    class_target.add_argument('-dt','--dynamic_threshold', action='store_true', help='use dynamic threshold for classification target')


    data = parser.add_argument_group('Data Loading arguments')
    data.add_argument('-b','--build_dataset',action='store_false', help='whether to build the dataset or not')
    data.add_argument('--seq_len', type=int, default=24)
    data.add_argument('--label_len', type=int, default=0)
    data.add_argument('--pred_len',type=int, default=6)
    data.add_argument('--val_split', type=float, default=0.7)
    data.add_argument('--test_split', type=float, default=0.85)
    data.add_argument('--batch_size', type=int, default=3)
    data.add_argument('--test_batch_size', type=int, default=1)



    model = parser.add_argument_group('Model training arguments')
    model.add_argument('--remote_logging', action='store_true', help='whether to log to a remote MLFlow server')
    model.add_argument('--experiment_name', type=str, default='stablecoin-depeg')
    model.add_argument('--run_name', type=str, default=None)
    model.add_argument('--model_root_path', type=str, default='./models')
    model.add_argument('--method', type = str, default='forecast', choices=['forecast', 'earlywarning'], help='forecasting or early warning classification task')
    model.add_argument('--model_name', type=str, default='iTransformer', help='name of the model to be trained')
    model.add_argument('--n_epochs', type=int, default=50)
    model.add_argument('--patience', type=int, default=10)
    model.add_argument('--verbose', type=int, default=1)
    model.add_argument('--check_lr', type=int, default=0)

    return parser
