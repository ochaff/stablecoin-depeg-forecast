
# Import models
import xgboost
from xgboost import XGBClassifier
from xgboost import plot_importance

import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier


import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt

import shap
from sklearn.preprocessing import StandardScaler, RobustScaler
from utils.build_dataset import build_dataset, add_dataset_args
import argparse

import os
import tempfile
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.catboost
import mlflow.xgboost

from mlflow.models.signature import infer_signature
from dotenv import load_dotenv
load_dotenv()

def log_fig(fig, artifact_path):
    # Works even if mlflow.log_figure is unavailable in your MLflow version
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, os.path.basename(artifact_path))
        fig.savefig(fp, dpi=150, bbox_inches="tight", transparent=True)
        mlflow.log_artifact(fp, os.path.dirname(artifact_path))


def _mlflow_log_current_fig(filename, artifact_subdir="plots", dpi=200):
    """Save the *current* matplotlib figure and log to MLflow artifacts."""
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, filename)
        plt.savefig(fp, dpi=dpi, bbox_inches="tight", transparent=True)
        mlflow.log_artifact(fp, artifact_subdir)
    plt.close()

def build_model(args, pos_weight):
    if args.model_name == "xgboost":
        return XGBClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=1.0,
            objective="binary:logistic",
            eval_metric="auc",
            early_stopping_rounds=args.early_stopping_rounds,
            scale_pos_weight=pos_weight,
            random_state=1233,
            n_jobs=args.n_jobs,
        )

    elif args.model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            num_leaves=args.num_leaves,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=1.0,
            objective="binary",
            scale_pos_weight=pos_weight,
            random_state=1233,
            n_jobs=args.n_jobs,
            verbosity=-1,
        )

    elif args.model_name == "catboost":
        return CatBoostClassifier(
            iterations=args.n_estimators,
            learning_rate=args.learning_rate,
            depth=args.max_depth,
            loss_function="Logloss",
            eval_metric="AUC",
            class_weights=[1.0, float(pos_weight)],
            random_seed=1233,
            verbose=False,
        )

    elif args.model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            class_weight={0: 1.0, 1: float(pos_weight)},
            random_state=1233,
            n_jobs=args.n_jobs,
            max_features=args.rf_max_features,
        )

    else:
        raise ValueError(f"Unsupported model_name: {args.model_name}")    
    

def fit_model(model, args, X_train, y_train, X_val, y_val):
    if args.model_name == "xgboost":
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )

    elif args.model_name == "lightgbm":
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)]
        )

    elif args.model_name == "catboost":
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=False
        )

    elif args.model_name == "random_forest":
        model.fit(X_train, y_train)

    else:
        raise ValueError(f"Unsupported model_name: {args.model_name}")

def get_best_iteration(model, model_name):
    if model_name == "xgboost":
        bi = getattr(model, "best_iteration", None)
    elif model_name == "lightgbm":
        bi = getattr(model, "best_iteration_", None)
    elif model_name == "catboost":
        bi = model.get_best_iteration()
    else:
        bi = None

    return -1 if bi is None else int(bi)

def log_model_to_mlflow(model, model_name, signature, input_example):
    if model_name == "xgboost":
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

    elif model_name == "lightgbm":
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

    elif model_name == "catboost":
        mlflow.catboost.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

    elif model_name == "random_forest":
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    
def get_native_feature_importance(model, model_name, feature_names):
    if model_name == "catboost":
        importances = model.get_feature_importance()
    else:
        importances = getattr(model, "feature_importances_", None)

    if importances is None:
        return None

    return pd.Series(importances, index=feature_names).sort_values(ascending=False)

def log_native_feature_importance(model, model_name, feature_names):
    imp = get_native_feature_importance(model, model_name, feature_names)
    if imp is None:
        return

    top_imp = imp.head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_imp.iloc[::-1].plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"{model_name} native feature importance")
    ax.set_xlabel("importance")
    fig.tight_layout()

    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "feature_importance_native.png")
        fig.savefig(fp, dpi=200, bbox_inches="tight")
        mlflow.log_artifact(fp, "plots/feature_importance")
    plt.close(fig)

def normalize_shap_output(shap_output, X, expected_value=None, positive_class_idx=1):
    feature_names = list(X.columns)
    data = X.values

    if isinstance(shap_output, shap.Explanation):
        values = shap_output.values
        base_values = shap_output.base_values

        # already 2D: (n_samples, n_features)
        if values.ndim == 2:
            return shap_output

        # binary/multiclass style: (n_samples, n_features, n_classes)
        if values.ndim == 3:
            values = values[:, :, positive_class_idx]

            if isinstance(base_values, np.ndarray):
                if base_values.ndim == 2:
                    base_values = base_values[:, positive_class_idx]
                elif base_values.ndim == 1 and len(base_values) > positive_class_idx and len(base_values) != len(values):
                    base_values = np.repeat(base_values[positive_class_idx], values.shape[0])

            return shap.Explanation(
                values=values,
                base_values=base_values,
                data=data,
                feature_names=feature_names
            )

    # old SHAP API: list of arrays
    if isinstance(shap_output, list):
        values = shap_output[positive_class_idx] if len(shap_output) > 1 else shap_output[0]

        if isinstance(expected_value, (list, np.ndarray)):
            base_value = expected_value[positive_class_idx] if len(expected_value) > positive_class_idx else expected_value[0]
        else:
            base_value = expected_value

        if np.isscalar(base_value):
            base_value = np.repeat(base_value, len(X))

        return shap.Explanation(
            values=values,
            base_values=base_value,
            data=data,
            feature_names=feature_names
        )

    arr = np.asarray(shap_output)
    if arr.ndim == 3:
        arr = arr[:, :, positive_class_idx]

    return shap.Explanation(
        values=arr,
        data=data,
        feature_names=feature_names
    )

def compute_shap_explanation(model, X_explain, shap_type):
    explainer = shap.TreeExplainer(model, feature_perturbation=shap_type)
    raw_shap = explainer(X_explain)
    shap_values = normalize_shap_output(
        raw_shap,
        X_explain,
        expected_value=getattr(explainer, "expected_value", None),
        positive_class_idx=1
    )
    return explainer, shap_values



os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')
uri = os.getenv('MLFLOW_TRACKING_URI')
artifact_uri = os.getenv('ARTIFACT_URI')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_dataset_args(parser)

    training_args = parser.add_argument_group('Training arguments')
    training_args.add_argument('--remote_logging', action='store_true', help='whether to log training metrics to remote MLFlow server')
    training_args.add_argument('--experiment_name', type=str, help='name of the MLflow experiment to log to')
    training_args.add_argument('--run_name', type=str, help='name of the MLflow run to log to')
    training_args.add_argument('--test_size', type=float, default=0.30, help='proportion of dataset to use as test set')
    training_args.add_argument('--val_size', type=float, default=0.15, help='proportion of dataset to use as validation set')
    training_args.add_argument('--scaler', type = str, default = 'standard', help='whether to scale features using StandardScaler')
    training_args.add_argument('--model_name',type=str, default='xgboost', choices=['xgboost', 'lightgbm', 'catboost', 'random_forest'], help='which model to train')


    model_args = parser.add_argument_group('Model arguments')
    model_args.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for XGBoost')
    model_args.add_argument('--early_stopping_rounds', type=int, default=200, help='early stopping rounds for XGBoost')
    model_args.add_argument('--eval_metric', type=str, default='auc', help='evaluation metric for XGBoost')
    model_args.add_argument('--n_estimators', type=int, default=800, help='number of trees for XGBoost')
    model_args.add_argument('--shap_type', type=str, default='tree_path_dependent', help='type of SHAP explainer to use')
    model_args.add_argument('--max_depth', type=int, default=6, help='maximum tree depth')
    model_args.add_argument('--num_leaves', type=int, default=31, help='LightGBM only')
    model_args.add_argument('--rf_max_features', type=str, default='sqrt', help='RandomForest only')
    model_args.add_argument('--n_jobs', type=int, default=-1, help='parallel jobs where supported')

    args = parser.parse_args()
    dict_args = vars(args)
    dict_args['target'] = True
    dataset_path = build_dataset(**dict_args)

        # ---- MLflow setup ----
    if args.remote_logging:
        mlflow.set_tracking_uri(uri)
        try:
            exp_id = mlflow.create_experiment(args.experiment_name, artifact_location=artifact_uri)
        except mlflow.exceptions.MlflowException:
            exp = mlflow.get_experiment_by_name(args.experiment_name)
            exp_id = exp.experiment_id
        mlflow.set_experiment(args.experiment_name)

    dataset = pd.read_parquet(dataset_path)
    dataset['timestamp'] = dataset.index
    for k in range(1, 8):
        dataset[f"poolTick_lag{k}h"] = dataset["poolTick"].shift(k)
    dataset = dataset.dropna()
    TIME_COL = "timestamp"   
    TARGET_COL = "target"    
    df = dataset.copy()


    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)


    FEATURES = [c for c in df.columns if c not in [TIME_COL, TARGET_COL]]

    X = df[FEATURES]
    y = df[TARGET_COL].astype(int)

    n = len(df)
    test_size = int(args.test_size * n)
    val_size  = int(args.val_size * n)

    train_end = n - test_size
    val_end = train_end
    train_end2 = train_end - val_size

    X_train, y_train = X.iloc[:train_end2], y.iloc[:train_end2]
    X_val,   y_val   = X.iloc[train_end2:val_end], y.iloc[train_end2:val_end]
    X_test,  y_test  = X.iloc[val_end:], y.iloc[val_end:]
    num_cols = X_train.columns 
    if args.scaler == 'standard':
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_val_scaled   = X_val.copy()
        X_test_scaled  = X_test.copy()
        X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_val_scaled[num_cols]   = scaler.transform(X_val[num_cols])
        X_test_scaled[num_cols]  = scaler.transform(X_test[num_cols])
        X_train, X_val, X_test = X_train_scaled, X_val_scaled, X_test_scaled
    elif args.scaler == 'robust':
        scaler = RobustScaler()
        X_train_scaled = X_train.copy()
        X_val_scaled   = X_val.copy()
        X_test_scaled  = X_test.copy()
        X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_val_scaled[num_cols]   = scaler.transform(X_val[num_cols])
        X_test_scaled[num_cols]  = scaler.transform(X_test[num_cols])
        X_train, X_val, X_test = X_train_scaled, X_val_scaled, X_test_scaled
    
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    w_pos = float(n_neg / max(n_pos, 1))
    print(f"Train positives={n_pos}, negatives={n_neg}, w_pos={w_pos:.3f}")

    model = build_model(args, pos_weight=w_pos)

    with mlflow.start_run(run_name=args.run_name):

        # Log params (a lot of them)
        mlflow.log_params({
            "model_name": args.model_name,
            "alpha": args.alpha,
            "dataset_path": dataset_path,
            "test_size": args.test_size,
            "val_size": args.val_size,
            "scaler": args.scaler,
            "learning_rate": args.learning_rate,
            "early_stopping_rounds": args.early_stopping_rounds,
            "eval_metric": args.eval_metric,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "num_leaves": args.num_leaves,
            "rf_max_features": args.rf_max_features,
            "target_window": args.target_window,
            "target_threshold": args.target_threshold,
            "depeg_side": args.depeg_side,
            "dynamic_threshold": int(args.dynamic_threshold),
            "scale_pos_weight_used": float(w_pos),
            "n_features": len(FEATURES),
        })

        # ---- train ----
        fit_model(model, args, X_train, y_train, X_val, y_val)
        
        mlflow.log_metric("best_iteration", get_best_iteration(model, args.model_name))

        # ---- evaluate ----
        proba_test = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba_test)
        auprc = average_precision_score(y_test, proba_test)

        mlflow.log_metric("test_roc_auc", float(auc))
        mlflow.log_metric("test_auprc", float(auprc))

        # threshold by Youden J
        fpr, tpr, thresholds = roc_curve(y_test, proba_test)
        j = tpr - fpr
        best_idx = int(np.argmax(j))
        thresh = float(thresholds[best_idx])
        mlflow.log_metric("best_threshold_youdenJ", thresh)
        mlflow.log_metric("tpr_at_best_threshold", float(tpr[best_idx]))
        mlflow.log_metric("fpr_at_best_threshold", float(fpr[best_idx]))

        # ---- plots (ROC + PR) ----
        prec, rec, _ = precision_recall_curve(y_test, proba_test)

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(fpr, tpr, label=f"AUC={auc:.3f}")
        ax[0].plot([0, 1], [0, 1], "--", color="gray")
        ax[0].set_title("ROC Curve")
        ax[0].set_xlabel("False Positive Rate")
        ax[0].set_ylabel("True Positive Rate")
        ax[0].legend()

        ax[1].plot(rec, prec, label=f"AUPRC={auprc:.3f}")
        ax[1].set_title("Precision-Recall Curve")
        ax[1].set_xlabel("Recall")
        ax[1].set_ylabel("Precision")
        ax[1].legend()

        log_fig(fig, "plots/roc_pr.png")
        plt.close(fig)

        # ---- confusion matrix at chosen threshold (optional) ----
        yhat = (proba_test >= thresh).astype(int)
        cm = confusion_matrix(y_test, yhat)

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        # ticks/labels (0,1) on both axes
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"])

        # keep full matrix visible and aligned
        ax.set_xlim(-0.5, cm.shape[1] - 0.5)
        ax.set_ylim(cm.shape[0] - 0.5, -0.5)

        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")

        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        log_fig(fig, "plots/confusion_matrix.png")
        plt.close(fig)

        # ---- model logging ----
        signature = infer_signature(X_train, model.predict_proba(X_train)[:, 1])

        # If you used StandardScaler, also log it so you can reproduce inference
        if args.scaler == "standard":
            mlflow.sklearn.log_model(scaler, name="preprocess_scaler")
        if args.scaler == "robust":
            mlflow.sklearn.log_model(scaler, name="preprocess_scaler")

        log_model_to_mlflow(
            model=model,
            model_name=args.model_name,
            signature=signature,
            input_example=X_train.head(5),
        )

        # save feature list for reproducibility
        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, "features.txt")
            with open(fp, "w") as f:
                f.write("\n".join(FEATURES))
            mlflow.log_artifact(fp, "meta")
        
        X_explain = X_test.copy()
        explainer, shap_values = compute_shap_explanation(
            model=model,
            X_explain=X_explain,
            shap_type=args.shap_type
        )

        # ---------- 1) SHAP beeswarm (global) ----------
        # (shap.plots.* creates a figure; we just save+log it)
        std_per_feature = shap_values.values.std(axis=0)
        order = np.argsort(std_per_feature)[::-1]
        shap.plots.beeswarm(shap_values, order=order, max_display=10, show=False)
        _mlflow_log_current_fig("shap_beeswarm_global.png", artifact_subdir="plots/shap")
        
        raw_train_shap = explainer(X_train)
        shap_values_train = normalize_shap_output(
            raw_train_shap,
            X_train,
            expected_value=getattr(explainer, "expected_value", None),
            positive_class_idx=1
        )
        std_per_feature_train = shap_values_train.values.std(axis=0)
        order = np.argsort(std_per_feature_train)[::-1]
        shap.plots.beeswarm(shap_values_train, order=order, max_display=10, show=False)
        _mlflow_log_current_fig("shap_beeswarm_global_insample.png", artifact_subdir="plots/shap")

        # ---------- 2) SHAP beeswarm for "warnings" subset ----------
        warn_mask = proba_test >= thresh
        sv_warn = shap_values[warn_mask]
        if sv_warn.values.shape[0] > 0:
            shap.plots.beeswarm(sv_warn, max_display=10, show=False)
            _mlflow_log_current_fig("shap_beeswarm_above_threshold.png", artifact_subdir="plots/shap")

            # ---------- 3) SHAP beeswarm ordered by mean positive contribution in warning regime ----------
            # (Your snippet referenced sv_warn_pos but didn't define it; use sv_warn here.)
            pos_mean = np.clip(sv_warn.values, 0, None).mean(axis=0)
            order_pos = np.argsort(pos_mean)[::-1]
            shap.plots.beeswarm(sv_warn, order=order_pos, max_display=10, show=False)
            _mlflow_log_current_fig("shap_beeswarm_above_threshold_ordered_by_pos_mean.png", artifact_subdir="plots/shap")

        # ---------- 4) Native feature importance (gain/weight/cover) ----------
        log_native_feature_importance(model, args.model_name, FEATURES)

        # ---------- 5) Predictions over time (probability + poolTick + shaded events) ----------
        test_dates = df["timestamp"].iloc[val_end:].reset_index(drop=True)
        y_test_reset = y_test.reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(13, 6))
        axi = ax.twinx()

        # Shade periods where y_test == 1
        for i in range(len(y_test_reset)):
            if int(y_test_reset.iloc[i]) == 1:
                ax.axvspan(
                    test_dates.iloc[i],
                    test_dates.iloc[min(i + 1, len(test_dates) - 1)],
                    color="lightgrey",
                    alpha=0.2,
                    zorder=0,
                )

        ax.plot(test_dates, proba_test, label="Predicted Probability", color="royalblue", lw=2, zorder=2)
        axi.plot(test_dates, X_test["poolTick"].reset_index(drop=True), label="Pool price", color="crimson", lw=1.5, alpha=0.9, zorder=1)

        ax.set_ylabel("probability of depeg in the next 24 hours", color="royalblue")
        axi.set_ylabel("Pool price", color="crimson")
        ax.tick_params(axis="y", labelcolor="royalblue")
        axi.tick_params(axis="y", labelcolor="crimson")

        fig.tight_layout()
        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, "predictions_over_time.png")
            fig.savefig(fp, dpi=300, transparent=True, bbox_inches="tight")
            mlflow.log_artifact(fp, "plots/timeseries")
        plt.close(fig)

        # ---- choose top 10 features by std(|SHAP|) ----
        mean_abs = np.abs(shap_values.values).std(axis=0)
        top10_idx = np.argsort(mean_abs)[::-1][:10]
        feature_names = list(shap_values.feature_names) if shap_values.feature_names is not None else list(X_test.columns)

        # ---- shap scatter per top feature ----
        for idx in top10_idx:
            fname = feature_names[idx]
            shap.plots.scatter(shap_values[:, idx], color = shap_values, show=False)
            plt.title(f"SHAP scatter: {fname}")
            _mlflow_log_current_fig(f"shap_scatter_{fname}.png")

            shap.plots.scatter(shap_values[:, idx], color=proba_test, show=False)
            plt.title(f"SHAP scatter (colored by proba): {fname}")
            _mlflow_log_current_fig(f"shap_scatter_{fname}_colored_by_proba.png")
        
        # Build a tidy artifact table
        pred_df = pd.DataFrame({
            "timestamp": df["timestamp"].iloc[val_end:].reset_index(drop=True),
            "y_true": y_test.reset_index(drop=True).astype(int),
            "proba_depeg": proba_test.astype(float),
        })

        with tempfile.TemporaryDirectory() as d:
            pq_path  = os.path.join(d, "test_pred_proba.parquet")
            pred_df.to_parquet(pq_path, index=False)
            mlflow.log_artifact(pq_path,  artifact_path="predictions")
        # ------------------ end MLflow logging block ------------------
