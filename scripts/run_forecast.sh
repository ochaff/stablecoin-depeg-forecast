for alpha in 1.0
do
for task in distribution
do

python main_lightning.py \
    --alpha $alpha \
    --forecast_task $task \
    --tau_pinball 0.025 \
    --model_name TSMixer \
    --method forecast \
    --experiment_name stablecoin-depeg \
    --run_name "alpha_${alpha}_${task}" \
    --n_epochs 50 \
    --patience 4 \
    --verbose 1 \
    --seq_len  64 \
    --pred_len 24 \
    --val_split 0.6 \
    --test_split 0.7 \
    --batch_size 256 \
    --test_batch_size 20 \
    --check_lr \
    --scaler revin \
    --affine 1 \
    --remote_logging \

done
done