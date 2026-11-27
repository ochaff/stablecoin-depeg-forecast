"""Early warning from the benchmark probabilistic forecasters (GARCH-t, NeuralForecast TimeXer), refitted on every
CV fold.

Same folds, events, target and NP alert policy as ``cv_earlywarning.py`` / ``cv_forecaster_exceedance.py``; the
score is the forecaster's exceedance probability S_t = max_{h=2..24} P(|d_{t+h}| >= threshold), read from its
quantile functions on the paper's quantile grid exactly as in ``holdout_exceedance_eval.py``. Each model writes one
run (``<model>_fc_exceedance``) in the usual layout, so ``cv_earlywarning_report.py`` pools it with the others.

GARCH-t   GARCH(1,1), constant mean, Student-t innovations (``benchmark_scripts/GARCH_forecast.py``). Parameters are
          estimated once per fold on all rows before the end of the fold's training period (the holdout benchmark
          also fits once, ``refit_every=20000``); the conditional variance is then filtered forward with fixed
          parameters through the validation and test origins.
TimeXer   NeuralForecast TimeXer with every numeric feature as historical exogenous input, MQLoss on the paper's
          quantile levels (``benchmark_scripts/TimeXer_forecast.py`` configuration). Trained once per fold on the
          rows before the fold's first validation origin, with the last ``nf_val_share`` of them for early
          stopping, then applied with fixed weights to every validation and test origin.

    python cv_benchmark_exceedance.py --models garch_t timexer_nf --experiment_name <exp> --remote_logging
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

import cv_earlywarning as C
from early_warning.cv import RunLogger, make_folds, ranking_metrics, select_feature_set, split_manifest
from early_warning.events import build_target
from early_warning.pooled import discover_runs, run_cv_budget
from early_warning.threshold_selection import choose_neyman_pearson
from early_warning.tracking import MlflowTracker
from holdout_exceedance_eval import exceedance

MODELS = {"garch_t": ("GARCH_t_fc_exceedance", "price_only"), "timexer_nf": ("TimeXer_NF_fc_exceedance", "full"),
          "tide_nf": ("TiDE_NF_fc_exceedance", "full"), "arima": ("ARIMA_fc_exceedance", "price_only"),
          # the paper's neural benchmarks: SCQF's spline + GPD head and combined CRPS/twCRPS loss
          # (section 12 benchmark_scripts/{TiDE,TimeXer}_forecast.py --head spline_gpd)
          "tide_sg": ("TiDE_SG_fc_exceedance", "full"), "timexer_sg": ("TimeXer_SG_fc_exceedance", "full"),
          "naive": ("Naive_RW_fc_exceedance", "price_only")}
DEFAULT_SEED = {"timexer_nf": 1, "tide_nf": 1233, "timexer_sg": 1, "tide_sg": 1}             # random_seed of the holdout benchmark runs
H = 24


def u_grid():
    return np.asarray(pickle.load(open("benchmark_outputs/garch_student_t/preds_test_set.pkl", "rb"))["u_grid"])


# ---------------------------------------------------------------------------------------------------------------
# GARCH-t
# ---------------------------------------------------------------------------------------------------------------
def _garch_filter(eps2, omega, alpha, beta, backcast):
    """sigma^2_t = omega + alpha eps^2_{t-1} + beta sigma^2_{t-1}, sigma^2_0 = backcast (arch's convention)."""
    from scipy.signal import lfilter
    drive = np.empty_like(eps2)
    drive[0] = omega + (alpha + beta) * backcast          # equals omega + alpha*backcast + beta*backcast
    drive[1:] = omega + alpha * eps2[:-1]
    return lfilter([1.0], [1.0, -beta], drive)


def _backcast(eps2, tau=75):
    """arch's variance backcast: exponentially weighted (0.94) mean of the first tau squared residuals."""
    w = 0.94 ** np.arange(min(tau, len(eps2)))
    return float(np.sum(w / w.sum() * eps2[:len(w)]))


def fit_garch_t(y):
    """Gaussian-quasi-free MLE of a constant-mean GARCH(1,1) with standardised Student-t innovations (the arch
    ``arch_model(mean='Constant', vol='GARCH', dist='StudentsT')`` likelihood, implemented with scipy because arch
    does not import against the installed pandas)."""
    from scipy.optimize import minimize
    from scipy.special import gammaln
    y = np.asarray(y, np.float64)

    def nll(theta):
        mu, omega, alpha, beta, nu = theta
        eps2 = (y - mu) ** 2
        s2 = np.maximum(_garch_filter(eps2, omega, alpha, beta, _backcast(eps2)), 1e-12)
        c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
        return -np.sum(c - 0.5 * np.log(s2) - (nu + 1) / 2 * np.log1p(eps2 / (s2 * (nu - 2))))

    var = float(np.var(y))
    x0 = np.array([float(np.mean(y)), 0.05 * var, 0.1, 0.85, 8.0])
    bounds = [(-10 * abs(y).max(), 10 * abs(y).max()), (1e-8 * var, 10 * var), (0.0, 1.0), (0.0, 1.0), (2.05, 500.0)]
    cons = [{"type": "ineq", "fun": lambda t: 0.99999 - t[2] - t[3]}]
    r = minimize(nll, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
    mu, omega, alpha, beta, nu = r.x
    return dict(mu=mu, omega=omega, alpha=alpha, beta=beta, nu=nu, converged=bool(r.success), loglik=-float(r.fun))


def garch_quantiles(y, train_end, origins, u):
    """Q[B, H, J] for origins (row indices; forecast of rows a+1..a+H) from a GARCH(1,1)-t fitted on y[:train_end+1],
    variance filtered forward with fixed parameters (arch ``fix(...).forecast`` equivalent)."""
    from scipy.stats import t as student_t
    info = fit_garch_t(y[:train_end + 1])
    mu, omega, alpha, beta, nu = (info[k] for k in ("mu", "omega", "alpha", "beta", "nu"))
    last = int(origins.max())
    eps2 = (np.asarray(y[:last + 1], np.float64) - mu) ** 2
    sig2 = _garch_filter(eps2, omega, alpha, beta, _backcast(eps2))
    steps = np.empty((len(origins), H))
    steps[:, 0] = omega + alpha * eps2[origins] + beta * sig2[origins]            # sigma^2_{a+1 | a}
    for h in range(1, H):                                                          # E[sigma^2_{a+h+1} | a]
        steps[:, h] = omega + (alpha + beta) * steps[:, h - 1]
    z = np.sqrt((nu - 2.0) / nu) * student_t.ppf(u, df=nu)
    Q = mu + np.sqrt(np.maximum(steps, 1e-12))[:, :, None] * z[None, None, :]
    return Q, info


# ---------------------------------------------------------------------------------------------------------------
# NeuralForecast TimeXer
# ---------------------------------------------------------------------------------------------------------------
def central_levels(u):
    return sorted({round(100.0 * abs(2 * x - 1), 6) for x in u if not np.isclose(x, 0.5) and 0 < abs(2 * x - 1) < 1})


def nf_to_quantiles(cv_df, u, prefix="TimeXer"):
    cv_df = cv_df.sort_values(["cutoff", "ds"])
    cutoffs = pd.DatetimeIndex(cv_df["cutoff"].unique())
    B = len(cutoffs)
    cols = []
    for x in u:
        if np.isclose(x, 0.5):
            cols.append(f"{prefix}-median" if f"{prefix}-median" in cv_df else prefix)
        else:
            lvl = round(100.0 * abs(2 * x - 1), 6)
            side = "lo" if x < 0.5 else "hi"
            name = next((c for c in cv_df.columns if c.startswith(f"{prefix}-{side}-")
                         and abs(float(c.rsplit("-", 1)[1]) - lvl) < 1e-4), None)
            if name is None:
                raise KeyError(f"no {prefix}-{side}-{lvl} column")
            cols.append(name)
    Q = cv_df[cols].to_numpy(np.float64).reshape(B, H, len(u))
    return np.maximum.accumulate(Q, axis=-1), cutoffs


def _fold_frame(cv, exog, last):
    df = cv.iloc[:last + H + 1][["timestamp", "depeg_bps"] + exog].copy().rename(columns={"depeg_bps": "y"})
    df["ds"] = pd.DatetimeIndex(df.pop("timestamp")).tz_localize(None)
    df["unique_id"] = "stablecoin_depeg"
    if exog:
        df[exog] = df[exog].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return df


def _select_origins(cv_df, df, origins, u, prefix):
    """Windows cover every origin from the first validation to the last test hour (incl. the embargo gap)."""
    first, last = int(origins.min()), int(origins.max())
    Q, cutoffs = nf_to_quantiles(cv_df, u, prefix)
    expect = pd.DatetimeIndex(df.ds.iloc[first:last + 1])
    assert cutoffs.equals(expect), (cutoffs[:3], expect[:3])
    return Q[np.asarray(origins) - first]


S12_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "onchain-insights",
                           "12. Evaluation of Probabilistic Forecasts", "benchmark_scripts")


def nf_quantiles(name, cv, exog, origins, u, args, seed, head="mqloss"):
    """NeuralForecast TimeXer / TiDE (configurations of benchmark_scripts/{TimeXer,TiDE}_forecast.py).
    head="spline_gpd": the paper's benchmark head (section 12 spliced_quantile_head.py, PAPER_HEAD_CONFIG), with
    the section 12 settings (input_size 64; batch size 32 for TiDE, 64 for TimeXer), output on the grid u."""
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast import models as nfm
    first, last = int(origins.min()), int(origins.max())
    df = _fold_frame(cv, exog, last)
    val_size = int(args.nf_val_share * (first + 1))
    if head == "spline_gpd":
        import sys
        import torch
        if S12_SCRIPTS not in sys.path:
            sys.path.insert(0, S12_SCRIPTS)
        from spliced_quantile_head import SplicedSplineQuantileLoss, PAPER_HEAD_CONFIG
        loss = SplicedSplineQuantileLoss(output_quantiles=torch.as_tensor(u, dtype=torch.float32), **PAPER_HEAD_CONFIG)
    else:
        loss = MQLoss(level=central_levels(u))
    common = dict(h=H, hist_exog_list=exog, loss=loss, valid_loss=loss, max_steps=args.nf_max_steps,
                  scaler_type="standard", random_seed=seed, learning_rate=1e-3, windows_batch_size=1024,
                  val_check_steps=50, early_stop_patience_steps=10, enable_progress_bar=False,
                  enable_model_summary=False, logger=False)
    if name == "TimeXer":
        model = nfm.TimeXer(input_size=64, n_series=1, batch_size=64, **common)
    elif head == "spline_gpd":
        model = nfm.TiDE(input_size=64, batch_size=32, **common)
    else:
        model = nfm.TiDE(input_size=168, batch_size=200, **common)
    nf = NeuralForecast(models=[model], freq="h")
    cv_df = nf.cross_validation(df=df, n_windows=last - first + 1, step_size=1, val_size=val_size, refit=False,
                                verbose=False)
    info = dict(val_size=val_size, n_windows=last - first + 1, head=head)
    if head == "spline_gpd":
        from spliced_quantile_head import cv_df_to_quantile_pickle_dict_from_loss
        A = cv_df_to_quantile_pickle_dict_from_loss(cv_df, nf.models[0].loss, name)
        assert np.allclose(A["u_grid"], u, atol=1e-6), "spline head output grid differs from u"
        cutoffs = pd.DatetimeIndex(A["meta"]["cutoff"])
        expect = pd.DatetimeIndex(df.ds.iloc[first:last + 1])
        assert cutoffs.equals(expect), (cutoffs[:3], expect[:3])
        Q = np.asarray(A["Q"])
        assert len(Q) == last - first + 1, (Q.shape, last - first + 1)
        return Q[np.asarray(origins) - first], info
    return _select_origins(cv_df, df, origins, u, name), info


def sf_quantiles(name, cv, origins, u):
    """StatsForecast AutoARIMA (season 24, stepwise, approximation) / Naive, fitted once on the rows before the first
    validation origin and rolled forward without refitting (configurations of benchmark_scripts/{ARIMA,Naive}_forecast.py;
    Gaussian prediction intervals)."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, Naive
    first, last = int(origins.min()), int(origins.max())
    df = _fold_frame(cv, [], last)[["unique_id", "ds", "y"]]
    model = AutoARIMA(season_length=24, stepwise=True, approximation=True, alias="AutoARIMA") \
        if name == "AutoARIMA" else Naive(alias="Naive")
    cv_df = StatsForecast(models=[model], freq="h", n_jobs=1).cross_validation(
        df=df, h=H, n_windows=last - first + 1, step_size=1, level=central_levels(u), refit=False)
    info = {}
    if name == "AutoARIMA":
        try:
            from statsforecast.arima import arima_string
            info["order"] = arima_string(model.model_) if hasattr(model, "model_") else ""
        except Exception:
            pass
    return _select_origins(cv_df, df, origins, u, name), info


# ---------------------------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--dataset_file", default="preprocessed_datasets/dataset_alpha_0.1_full.parquet")
    p.add_argument("--experiment_name", default="earlywarning_pool_15bps_2026-11-19")
    p.add_argument("--log_dir", default="lightning_logs")
    p.add_argument("--random_state", type=int, default=None, help="default: the holdout benchmark's seed")
    p.add_argument("--nf_max_steps", type=int, default=1000)
    p.add_argument("--nf_val_share", type=float, default=0.15 / 0.7,
                   help="share of the pre-validation rows used for NF early stopping (holdout benchmark: 15%% of "
                        "the data with 70%% for training)")
    p.add_argument("--remote_logging", action="store_true")
    p.add_argument("--mlflow_experiment", default="ts-classifier")
    p.add_argument("--mlflow_tag", nargs="*", default=[])
    p.add_argument("--n_bootstrap", type=int, default=1000)
    a = p.parse_args()

    ew = C.parse_args(["--alpha", "0.1", "--dataset_file", a.dataset_file, "--feature_set", "full"])
    data, context, numeric, path = C.load_dataset(ew)
    n = len(data)
    holdout_start = data["timestamp"].iloc[n - int(ew.holdout_frac * n)]
    cv = data[data.timestamp < holdout_start].reset_index(drop=True)
    ctx = context[context.timestamp < holdout_start].reset_index(drop=True)
    times = pd.DatetimeIndex(cv.timestamp)
    labels, observed, _ = build_target(cv, ctx, ew.target_threshold, ew.min_lead_hours, ew.max_lead_hours,
                                       ew.depeg_event_reset_hours)
    folds = make_folds(times, ctx, threshold=ew.target_threshold, reset_hours=ew.depeg_event_reset_hours,
                       cv_start=ew.cv_start, cv_end=ew.cv_end, n_splits=ew.n_splits,
                       cv_embargo_hours=ew.cv_embargo_hours, cv_val_frac=ew.cv_val_frac)
    manifest = split_manifest(folds, times, holdout_start, data.timestamp.iloc[-1], ew.cv_val_frac)
    u = u_grid()
    y = cv.depeg_bps.to_numpy(np.float64)
    exog = [f for f in select_feature_set(numeric, "full") if f != "depeg_bps"]
    tracker = MlflowTracker(a.mlflow_experiment, enabled=a.remote_logging)
    policy = lambda *v, **kw: choose_neyman_pearson(*v, **kw, fpr=ew.np_fpr, delta=ew.np_delta,
                                                     negative_spacing_hours=ew.np_negative_spacing_hours,
                                                     reset_hours=ew.depeg_event_reset_hours,
                                                     cooldown_hours=ew.alert_cooldown_hours)

    for key in a.models:
        model_name, feature_set = MODELS[key]
        seed = a.random_state if a.random_state is not None else DEFAULT_SEED.get(key, 1)
        run_prefix = f"bench_{key}_seed{seed}"
        common = dict(model_name=model_name, dataset_path=str(path), feature_set=feature_set, alpha=0.1,
                      target_threshold=ew.target_threshold, random_state=seed, holdout_start=holdout_start,
                      depeg_event_reset_hours=ew.depeg_event_reset_hours, quantile_levels=len(u),
                      n_features=1 if key == "garch_t" else len(exog) + 1,
                      training_target=f"forecast {H}h ahead; alert target onset 2-24h",
                      threshold_policy=f"np_fpr_{ew.np_fpr:g}".replace(".", "p"),
                      **({"nf_max_steps": a.nf_max_steps, "nf_val_share": a.nf_val_share}
                         if key in ("timexer_nf", "tide_nf", "timexer_sg", "tide_sg") else {}))
        logger = RunLogger(a.log_dir, a.experiment_name, f"{run_prefix}_{model_name}_alpha_0.1")
        logger.log_params(common)
        logger.save_frame(manifest, "cv/split_manifest.csv")
        logger.save_frame(cv[["timestamp", "depeg_bps"]].astype({"depeg_bps": "float32"}),
                          "cv/realised_depeg_context.parquet")
        tags = {"local_experiment": a.experiment_name, "local_run_prefix": run_prefix, "feature_set": feature_set,
                "seed": seed, "target_threshold": ew.target_threshold, "task": "earlywarning_cv",
                "model_family": "benchmark_forecaster", "model_name": model_name,
                **dict(t.split("=", 1) for t in a.mlflow_tag)}
        run = tracker.start_run(logger.run_dir.name, tags)
        tracker.log_params(run, {**common, "local_run_dir": str(logger.run_dir.resolve())})
        scores, rows = [], []
        try:
            for f in folds:
                t0 = time.time()
                origins = np.concatenate([f["val"], f["test"]])
                if key == "garch_t":
                    Q, info = garch_quantiles(y, int(f["train"][-1]), origins, u)
                elif key in ("timexer_nf", "tide_nf", "timexer_sg", "tide_sg"):
                    Q, info = nf_quantiles("TimeXer" if key.startswith("timexer") else "TiDE", cv, exog, origins, u, a,
                                           seed + f["fold"] - 1, head="spline_gpd" if key.endswith("_sg") else "mqloss")
                else:
                    Q, info = sf_quantiles("AutoARIMA" if key == "arima" else "Naive", cv, origins, u)
                s = exceedance(Q, u, ew.target_threshold, int(ew.min_lead_hours), int(ew.max_lead_hours))
                s_val, s_test = s[:len(f["val"])], s[len(f["val"]):]
                part = [pd.DataFrame({"fold": f["fold"], "split": split, "model_name": model_name, "alpha": 0.1,
                                      "target_threshold": ew.target_threshold,
                                      "timestamp": cv.timestamp.iloc[idx].to_numpy(),
                                      "depeg_bps": y[idx].astype(np.float32), "probability": pp})
                        for split, idx, pp in (("validation", f["val"], s_val), ("test", f["test"], s_test))]
                scores += part
                tobs = observed[f["test"]]
                row = dict(fold=f["fold"], model_name=model_name, feature_set=feature_set,
                           fit_seconds=time.time() - t0, **{f"fit_{k}": v for k, v in info.items()},
                           **ranking_metrics(labels[f["test"]][tobs], s_test[tobs]),
                           **C.np_fold_evaluation(pd.concat(part, ignore_index=True), ctx, f["fold"], ew))
                rows.append(row)
                tracker.log_metrics(run, {k: v for k, v in row.items() if np.isscalar(v) and not isinstance(v, str)},
                                    step=f["fold"])
                print(f"  {model_name} fold {f['fold']}: AUC {row['fold_auc']:.3f} AUPRC {row['fold_auprc']:.3f} "
                      f"detected {row['np_detected']}/{row['np_events']} false alerts {row['np_false_alerts']} "
                      f"({row['fit_seconds']:.0f}s)", flush=True)
        except BaseException:
            tracker.end_run(run, "FAILED")
            raise
        logger.save_frame(pd.concat(scores, ignore_index=True), "cv/operational_scores.parquet")
        logger.save_frame(pd.DataFrame(rows), "cv/fold_metrics.csv")
        runs, _ = discover_runs(a.log_dir, a.experiment_name)
        runs = runs[runs.run_name == logger.run_dir.name]
        _, summary, _ = run_cv_budget(runs, ew.min_lead_hours, ew.max_lead_hours, n_bootstrap=a.n_bootstrap,
                                      cooldown_hours=ew.alert_cooldown_hours, threshold_selector=policy)
        rec = summary.to_dict("records")[0]
        tracker.log_metrics(run, {k: v for k, v in rec.items() if k.startswith("cv_")})
        tracker.log_artifacts(run, logger.run_dir)
        tracker.end_run(run)
        print(summary[["model_name", "cv_events_detected", "cv_eligible_events", "cv_false_alerts_per_month",
                       "cv_median_lead_hours", "cv_auc_mean", "cv_auprc_mean", "cv_brier_skill_score_mean",
                       "cv_breakeven_cost_loss", "cv_value_r0p1"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    main()
