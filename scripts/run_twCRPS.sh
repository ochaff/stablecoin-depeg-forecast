for alpha in 0.5 
do
for task in distribution
do
for model in TSMixer
do
for loss in twcrps
do
for decomp in spline
do
for revin in revin
do
for knot_p in 3.0
do
for grid in logit power-tail 
do

python main_lightning.py \
    --alpha $alpha \
    --forecast_task $task \
    --model_name $model \
    --method forecast \
    --experiment_name stablecoin-depeg \
    --run_name "${model}_alpha_${alpha}_${task}_${loss}_${decomp}_${revin}_${grid}" \
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
    --n_cheb 10 \
    --twcrps_side two_sided \
    --twcrps_smooth_h 1 \
    --u_grid_size 256 \
    --grid_density $grid \
    --quantile_decomp $decomp \
    --knot_kind uniform \
    --knot_p $knot_p \
    --spline_degree 3 \
    --remote_logging \

done
done
done
done
done
done
done
done
