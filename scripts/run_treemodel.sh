for alpha in  0.6 1.0 1.5 2.0 3.0 4.0 6.0 8.0
do 

python run_xgboost.py \
    --alpha $alpha \
    -th 15 \
    -w 24 \
    --remote_logging \
    --max_depth 6 \
    --n_estimators 1000 \
    --early_stopping_rounds 200 \
    --experiment_name stablecoin_treeclassifier \
    --run_name xgboost_alpha_$alpha\_fullfeatures \
    --eval_metric auc \
    --scaler robust \
    --objective binary:logistic \

done
