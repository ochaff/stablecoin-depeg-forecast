for alpha in 1.5
do
for metric in auc auprc
do
for loss in bce focal
do
for model in TSMixer CNN
do

python main_lightning.py \
    --alpha $alpha \
    --model_name $model \
    --method earlywarning \
    --target_threshold 15 \
    --target_window 24 \
    --depeg_side both \
    --experiment_name stablecoin-earlywarning \
    --run_name "alpha_${alpha}_model_${model}" \
    --n_epochs 50 \
    --patience 10 \
    --verbose 1 \
    --seq_len  168 \
    --pred_len 1 \
    --val_split 0.55 \
    --test_split 0.7 \
    --batch_size 256 \
    --test_batch_size 20 \
    --learning_rate 0.0001 \
    --compute_shap 1 \
    --shap_background_size 64 \
    --shap_test_samples 256 \
    --class_loss $loss \
    --validation_metric $metric \
    --remote_logging \
    --scale_pos \
    --scaler_type robust \

done
done
done
done