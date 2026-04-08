for alpha in 0.3 
do
for task in distribution
do
for model in SAINT
do
for loss in twcrps
do
for tail in gpd
do
for decomp in spline
do
for revin in revin
do
for knot_p in 3.0
do
for grid in power-tail uniform
do
for selector in sparsemax
do
for gate in 1
do
for lam in 1e-4
do

python main_lightning.py \
    --alpha $alpha \
    --forecast_task $task \
    --model_name $model \
    --method forecast \
    --experiment_name stablecoin-models \
    --run_name "${model}_${selector}_gated_${gate}_lambda_${lam}" \
    --n_epochs 100 \
    --patience 5 \
    --verbose 1 \
    --check_lr \
    --seq_len  64 \
    --pred_len 24 \
    --val_split 0.55 \
    --test_split 0.7 \
    --batch_size 500 \
    --test_batch_size 50 \
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
    --tail_model $tail \
    --gpd_u_low 0.03 \
    --gpd_u_high 0.97 \
    --gpd_xi_min -0.25 \
    --gpd_xi_max 0.5 \
    --remote_logging \
    --selector_activation $selector \
    --use_hard_concrete $gate \
    --l0_lambda $lam \
    --save_test_diagnostics 1 \

done
done
done
done
done
done
done
done
done
done
done
done
