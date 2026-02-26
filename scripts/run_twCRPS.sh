for alpha in 0.5 
do
for task in distribution
do
for model in TSMixer
do
for loss in crps twcrps
do
for decomp in chebyshev spline
do
for revin in robust
do

python main_lightning.py \
    --alpha $alpha \
    --forecast_task $task \
    --model_name $model \
    --method forecast \
    --experiment_name stablecoin-depeg \
    --run_name "${model}_alpha_${alpha}_${task}_${loss}_${decomp}_${revin}" \
    --n_epochs 100 \
    --patience 5 \
    --verbose 1 \
    --check_lr \
    --seq_len  64 \
    --pred_len 24 \
    --val_split 0.55 \
    --test_split 0.7 \
    --batch_size 256 \
    --test_batch_size 20 \
    --revin_type $revin \
    --affine 1 \
    --dist_loss $loss \
    --n_cheb 8 \
    --twcrps_side two_sided \
    --twcrps_smooth_h 1 \
    --u_grid_size 200 \
    --grid_density uniform \
    --quantile_decomp $decomp \
    --knot_kind uniform \
    --spline_degree 3 \

done
done
done
done
done
done
