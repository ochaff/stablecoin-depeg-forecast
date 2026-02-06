for alpha in 0.4 0.8
do
for method in earlywarning
do

python main_lightning.py \
    --alpha $alpha \
    --model_name iTransformer \
    --method $method \
    --target_threshold 20 \
    --target_window 24 \
    --depeg_side both \
    --experiment_name stablecoin-depeg \
    --run_name "alpha_${alpha}_${method}" \
    --n_epochs 50 \
    --patience 10 \
    --verbose 1 \
    --check_lr 0 \
    --seq_len  168 \
    --pred_len 1 \
    --val_split 0.7 \
    --test_split 0.85 \
    --batch_size 64 \
    --test_batch_size 20 \
    --learning_rate 0.0001 \
    --scaler revin \
    --affine 1 \
    --remote_logging \

done
done