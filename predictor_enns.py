#!/usr/bin/env python
"""
HDLSS feature‑importance pipeline for psychopathy data (ENNS integrated)
=======================================================================

This version utilitza l’**Evidential Neural Network Screening (ENNS)** que
m’has passat íntegrament —incloent les variants de regressió i classificació—
per seleccionar variables estables abans d’entrenar un Random‑Forest amb
permutation importance. El flux extern amb 60 resplits, mètriques de
rendiment, gràfics i resum de comptatges continua idèntic.

Outputs (CSV)
-------------
* `variable_selection_counts.csv` – nombre de vegades que cada variable apareix
  seleccionada en totes les repeticions (internes + externes).
* `rf_mean_importances.csv` – importància mitjana de permutació.
* `model_performance_metrics.csv` – RMSE/MAE de cada split extern.

Dependències
------------
```bash
pip install pandas numpy pyreadstat scikit-learn joblib tqdm matplotlib tensorflow
```
"""

from __future__ import annotations

import pathlib
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pyreadstat
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---- TensorFlow / Keras ------------------------------------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input as KInput
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# -----------------------------------------------------------------------------
# Project‑specific helpers (assumed available)
# -----------------------------------------------------------------------------
from func_sel import *
from funcions_net import *
from crear_dataset import *
from preprocessing import *

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
N_ITER = 5                # Internal CV iterations per external split
N_BOOTSTRAPS_ENNS = 20    # Bootstraps inside ENNS
IMPORTANCE_THRESHOLD = 0.5
MIN_SELECTION_RATIO = 0.65
TEST_SIZE = 0.2           # External test size
CAP_TARGET = 1_095        # Days (3 years)
RANDOM_STATE = 10         # Seed
N_JOBS = -1               # Parallel cores
VERBOSE = True

# -----------------------------------------------------------------------------
# ENNS functions (exactes del fragment que has proporcionat)
# -----------------------------------------------------------------------------

def create_model(input_dim):
    model = Sequential([
        KInput(shape=(input_dim,)),
        Dense(30, activation='relu'),
        Dropout(0.4),
        Dense(15, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mse'])
    return model


def get_feature_importance(model):
    weights = model.layers[0].get_weights()[0]
    importance = np.sum(np.abs(weights), axis=1)
    return importance


def enns(
    X,
    y,
    n_bootstraps: int = N_BOOTSTRAPS_ENNS,
    importance_threshold: float = IMPORTANCE_THRESHOLD,
    min_selection_ratio: float = MIN_SELECTION_RATIO,
):
    print("Iniciant ENNS amb EarlyStopping i validació interna…")
    print(
        f"Paràmetres: n_bootstraps={n_bootstraps}, "
        f"importance_threshold={importance_threshold}, "
        f"min_selection_ratio={min_selection_ratio}"
    )
    feature_counts = np.zeros(X.shape[1])

    for _ in range(n_bootstraps):
        X_bs, y_bs = resample(X, y, replace=True)
        model = create_model(X_bs.shape[1])

        early_stop = EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        model.fit(
            X_bs,
            y_bs,
            epochs=100,
            batch_size=16,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0,
        )

        importance = get_feature_importance(model)
        importance = importance / np.max(importance)
        feature_counts += importance >= importance_threshold

        tf.keras.backend.clear_session()

    selected_features = np.where(feature_counts >= (n_bootstraps * min_selection_ratio))[0]
    return selected_features.tolist()

# Optional classification ENNS (not used here but included for completeness)

def create_classification_model(input_dim):
    model = Sequential([
        KInput(shape=(input_dim,)),
        Dense(30, activation='relu'),
        Dropout(0.2),
        Dense(15, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def enns_classification(
    X,
    y,
    n_bootstraps: int = 40,
    importance_threshold: float = 0.7,
    min_selection_ratio: float = 0.7,
):

    feature_counts = np.zeros(X.shape[1])

    for _ in tqdm(range(n_bootstraps)):
        X_bs, y_bs = resample(X, y, replace=True)
        X_bs = X_bs.astype(np.float32)
        y_bs = y_bs.astype(np.float32)

        model = create_classification_model(X_bs.shape[1])
        early_stop = EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        model.fit(
            X_bs,
            y_bs,
            epochs=50,
            batch_size=16,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0,
        )

        importance = get_feature_importance(model)
        importance = importance / np.max(importance)
        feature_counts += importance >= importance_threshold

        tf.keras.backend.clear_session()

    selected_features = np.where(feature_counts >= (n_bootstraps * min_selection_ratio))[0]
    return selected_features.tolist()

# -----------------------------------------------------------------------------
# Core pipeline
# -----------------------------------------------------------------------------

def run_single_split(
    X_trainval: pd.DataFrame,
    y_trainval: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    external_i: int,
    internal_j: int,
):
    print(f"    [↳] External {external_i+1:02d} – Internal {internal_j+1:02d}")
    """Internal CV split: ENNS selection + RF permutation importance."""

    X_train = X_trainval.iloc[train_idx].values.astype(np.float32)
    y_train = y_trainval.iloc[train_idx].values.astype(np.float32)
    X_test = X_trainval.iloc[test_idx]
    y_test = y_trainval.iloc[test_idx]

    # ENNS variable selection
    sel_indices = enns(X_train, y_train)

    if not sel_indices:
        zeros = pd.Series(0.0, index=X_trainval.columns, name="importance")
        return [], zeros, np.nan, np.nan

    selected_vars = X_trainval.columns[sel_indices].tolist()

    # Random‑Forest model & permutation importance
    rf = RandomForestRegressor(
        n_estimators=250,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    ).fit(X_trainval.iloc[train_idx][selected_vars], y_trainval.iloc[train_idx])

    perm = permutation_importance(
        rf,
        X_trainval.iloc[train_idx][selected_vars],
        y_trainval.iloc[train_idx],
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    importances = (
        pd.Series(perm.importances_mean, index=selected_vars, name="importance")
        .reindex(X_trainval.columns)
        .fillna(0.0)
    )

    # Hold‑out performance
    y_pred = rf.predict(X_test[selected_vars])
    rmse_val = root_mean_squared_error(y_test, y_pred)
    mae_val = mean_absolute_error(y_test, y_pred)

    return selected_vars, importances, rmse_val, mae_val


def run_external_split(i: int, X_full: pd.DataFrame, y_full: pd.Series, k: int):
    print(f"[+] External repetition {i+1}")

    X_trainval_raw, X_test_raw, y_trainval, y_test = train_test_split(
        X_full,
        y_full,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE + i,
    )

    scaler = StandardScaler().fit(X_trainval_raw)
    X_trainval = pd.DataFrame(
        scaler.transform(X_trainval_raw), columns=X_full.columns
    )
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X_full.columns)

    rkf = RepeatedKFold(
        n_splits=k,
        n_repeats=int(np.ceil(N_ITER / k)),
        random_state=RANDOM_STATE,
    )

    results = [
    run_single_split(X_trainval, y_trainval, tr, te, i, j)
    for j, (tr, te) in enumerate(list(rkf.split(X_trainval))[:N_ITER])
    ]

    selected_lists, importances_list, rmses, maes = map(list, zip(*results))

    sel_counts_split = pd.Series(0, index=X_full.columns, dtype=int)
    for sel in selected_lists:
        sel_counts_split[sel] += 1

    mean_importances_split = pd.concat(importances_list, axis=1).mean(axis=1)

    vars_any_selected = sel_counts_split[sel_counts_split > 0].index.tolist()
    if vars_any_selected:
        rf_final = RandomForestRegressor(
            n_estimators=250,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ).fit(X_trainval[vars_any_selected], y_trainval)

        y_test_pred = rf_final.predict(X_test[vars_any_selected])
        test_rmse = root_mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
    else:
        test_rmse = np.nan
        test_mae = np.nan

    return {
        "cv_rmse_mean": float(np.mean(rmses)),
        "cv_rmse_std": float(np.std(rmses)),
        "cv_mae_mean": float(np.mean(maes)),
        "cv_mae_std": float(np.std(maes)),
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "selected_counts": sel_counts_split.to_dict(),
        "mean_importances": mean_importances_split.to_dict(),
    }



# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main(path_sav: str = "CEJFEAjut2015Updated.sav", path_vars: str = "variables3.csv") -> None:
    t0 = datetime.now()
    print("[+] Loading raw data …")

    # --- Load data (project‑specific helpers) ---
    df_orig, meta = pyreadstat.read_sav(path_sav)
    df_variables = pd.read_csv(path_vars, sep=";")
    dict_vars = create_dict(meta)

    # Psychology‑specific datasets (domain‑specific helpers)
    df_psico = dataset_psicologia(df_orig, dict_vars, df_variables)
    df_psico = drop_all_columns(df_psico, meta, df_variables, dict_vars)

    df_inicial = dataset_inicial(df_orig, dict_vars)
    df_global = create_dataset_global(df_orig, dict_vars, df_variables, meta)

    df = df_global.copy()
    print("   Dataset shape:", df.shape)

    # --- Target capping & preprocessing ---
    df["temps_fins_reincidencia1a"] = df_orig["temps_fins_reincidencia1a"].clip(upper=CAP_TARGET)

    _, df = neteja_na_columns(df, llindar=0.6)
    df = omple_nans(df)
    df = label_encoding(df)

    X_full = df.drop(columns="temps_fins_reincidencia1a")
    y_full = df["temps_fins_reincidencia1a"]

    # ---------------------------------------------------------------------
    # Repeated external splits (parallelised)
    # ---------------------------------------------------------------------
    N_SPLITS_EXTERN = 36
    k = int(round(1 / TEST_SIZE))

    all_results = Parallel(n_jobs=N_JOBS, verbose=VERBOSE)(
        delayed(run_external_split)(i, X_full, y_full, k)
        for i in range(N_SPLITS_EXTERN)
    )

    # ---------------------------------------------------------------------
    # Collect performance metrics & aggregated statistics
    # ---------------------------------------------------------------------
    perf_dicts = [
        {k: v for k, v in d.items() if k not in ("selected_counts", "mean_importances")}
        for d in all_results
    ]
    df_results = pd.DataFrame(perf_dicts)

    # Global selection counts & permutation importances
    sel_counts_global = pd.Series(0, index=X_full.columns, dtype=int)
    importances_accum = pd.Series(0.0, index=X_full.columns, dtype=float)
    for d in all_results:
        sel_counts_global += pd.Series(d["selected_counts"])
        importances_accum += pd.Series(d["mean_importances"])

    importances_mean_global = importances_accum / N_SPLITS_EXTERN

    # ---------------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------------
    outdir = pathlib.Path("outputs_overfit_enns")
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Performance metrics
    df_results.round(2).to_csv(outdir / "model_performance_metrics.csv", index=False)

    # 2. Mean permutation importances across all externals
    importances_mean_global.sort_values(ascending=False).to_csv(
        outdir / "rf_mean_importances.csv", header=["mean_importance"]
    )

    # 3. Variable selection counts
    sel_counts_global.sort_values(ascending=False).to_csv(
        outdir / "variable_selection_counts.csv", header=["times_selected"]
    )

    # ------------------------------------------------------------------
    # Diagnostic plots (RMSE/MAE vs. external split, etc.)
    # ------------------------------------------------------------------
    # Gràfica 1: CV RMSE vs Test RMSE
    plt.figure(figsize=(10, 6))
    plt.plot(df_results["cv_rmse_mean"], label="CV RMSE", marker="o")
    plt.plot(df_results["test_rmse"], label="Test RMSE", marker="x")
    plt.xlabel("External repetition")
    plt.ylabel("RMSE")
    plt.title("CV vs Test RMSE")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(outdir / "rmse_comparison.png"); plt.close()

    # Gràfica 2: CV MAE vs Test MAE
    plt.figure(figsize=(10, 6))
    plt.plot(df_results["cv_mae_mean"], label="CV MAE", marker="o")
    plt.plot(df_results["test_mae"], label="Test MAE", marker="x")
    plt.xlabel("External repetition")
    plt.ylabel("MAE")
    plt.title("CV vs Test MAE")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(outdir / "mae_comparison.png"); plt.close()

    # Distribution plots
    plt.figure(figsize=(8, 5))
    plt.hist(df_results["test_rmse"], bins=10, edgecolor="black", alpha=0.85)
    plt.xlabel("Test RMSE"); plt.ylabel("Frequency"); plt.title("Test RMSE distribution")
    plt.tight_layout(); plt.savefig(outdir / "test_rmse_distribution.png"); plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df_results["test_mae"], bins=10, edgecolor="black", alpha=0.85)
    plt.xlabel("Test MAE"); plt.ylabel("Frequency"); plt.title("Test MAE distribution")
    plt.tight_layout(); plt.savefig(outdir / "test_mae_distribution.png"); plt.close()

    # ------------------------------------------------------------------
    print("\n[✓] Finished. Key outputs saved in", outdir)


if __name__ == "__main__":
    try:
        main(*sys.argv[1:])
    except KeyboardInterrupt:
        print("Interrupted by user. Bye!")
