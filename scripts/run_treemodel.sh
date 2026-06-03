for alpha in 0.1
do
for model in xgboost lightgbm catboost random_forest
do

python run_xgboost.py \
    --alpha $alpha \
    --model_name $model \
    -th 15 \
    -w 24 \
    --remote_logging \
    --max_depth 8 \
    --n_estimators 1000 \
    --early_stopping_rounds 200 \
    --experiment_name stablecoin_treeclassifier \
    --run_name $model\_alpha_$alpha\_fullfeatures \
    --eval_metric auc \
    --scaler robust \

done
done
# python dl_artifacts.py ;