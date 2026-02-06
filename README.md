# stablecoin-depeg-forecast

## Collect up-to-date data

Download latest release from [stablecoin-onchain-data](https://github.com/MSCA-DN-Digital-Finance/stablecoin-onchain-data).

````
bash update_data.sh
````

## Build preprocessed dataset 

### Script options
```
usage: build_dataset.py [-h] [--dataset_path DATASET_PATH] --alpha ALPHA [--aave] [--aave_liq] [--crv] [--eth_price] [--eth_indicators] [--fear_greed] [--gegen] [-t] [--target_window TARGET_WINDOW]
                        [--target_threshold TARGET_THRESHOLD] [--depeg_side {both,up,down}]

main file arguments

options:
  -h, --help            show this help message and exit

dataset building arguments:
  --dataset_path DATASET_PATH
                        path to save the dataset
  --alpha ALPHA         Gegenbauer polynomial alpha parameter
  --aave                remove AAVE metrics
  --aave_liq            remove AAVE liquidations
  --crv                 remove Curve 3pool metrics
  --eth_price           remove ETH price oracle
  --eth_indicators      remove ETH price technical indicators
  --fear_greed          remove Fear and Greed index
  --gegen               remove Gegenbauer liquidity curve scores

classification target arguments:
  -t, --target          add binary classification target for depegs
  --target_window TARGET_WINDOW
                        time window (in hours) for classification target
  --target_threshold TARGET_THRESHOLD
                        threshold (in bps) for classification target
  --depeg_side {both,up,down}
                        depeg side for classification target

```

### Examples

Build a dataset for forecasting with $\alpha =0.8$, using all available features.
```
python -m utils.build_dataset --alpha 0.8 
```

Build a dataset for binary classification (depeg early warning), with early warning window 24 hours, 20 bps depeg threshold and both upwards and downwards depegs. 

```
python -m utils.build_dataset --alpha 0.5 -t -w 24 -th 20 -ds 'both'
```
