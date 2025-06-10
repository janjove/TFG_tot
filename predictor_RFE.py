#!/usr/bin/env python
"""
HDLSS feature‑importance pipeline for psychopathy data (RFE variant)
===================================================================

This script replaces the previous Lasso‑based sparse selection with a
**Recursive Feature Elimination (RFE)** procedure that retains exactly 50
variables in every resampling split. The rest of the workflow (Random‑Forest
permutation importance, repeated external resampling, overfitting checks) is
unchanged, except that it now also *counts* how many times **each** variable
is retained by RFE across **all** internal & external splits.  A final CSV
(`variable_selection_counts.csv`) summarises these tallies.

Outputs (CSV):
    • rf_mean_importances.csv        ― Mean permutation importance per variable
    • model_performance_metrics.csv  ― RMSE & MAE for each external split
    • variable_selection_counts.csv  ― Times each variable was chosen by RFE

Design notes
------------
1. **RFE(50)** is performed with a `RandomForestRegressor` base estimator so the
   ranking criterion naturally captures non‑linear effects.
2. Each internal CV split thus selects *exactly* 50 features; over *K* internal
   folds × *N* external repetitions this yields a robust stability estimate.
3. Final counts are simply accumulated across all inner selections (e.g. with
   60 externals × 5 internals ⇒ max count = 300).
4. The target is still capped at 1 095 days.

Dependencies
------------
    pip install pandas numpy pyreadstat scikit‑learn joblib tqdm matplotlib
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
from sklearn.feature_selection import RFE
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Project‑specific helpers (assumed to exist in local modules)
# -----------------------------------------------------------------------------
from func_sel import *
from funcions_net import *
from crear_dataset import *
from preprocessing import *

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
N_ITER = 5            # Internal CV iterations per external split
N_FEATURES_RFE = 50   # Features retained by RFE
TEST_SIZE = 0.2       # Proportion for external test split
CAP_TARGET = 1_095    # Days (3 years) – clip upper outliers
RANDOM_STATE = 10     # Reproducibility
N_JOBS = -1           # Parallel cores (‑1 ⇒ all)
VERBOSE = True

# -----------------------------------------------------------------------------
# Core pipeline functions
# -----------------------------------------------------------------------------

def run_single_split(
    X_trainval: pd.DataFrame,
    y_trainval: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[list[str], pd.Series, float, float]:
    """One internal CV split: RFE(50) selection + RF permutation importance."""

    X_train, X_test = X_trainval.iloc[train_idx], X_trainval.iloc[test_idx]
    y_train, y_test = y_trainval.iloc[train_idx], y_trainval.iloc[test_idx]

    # --- Recursive Feature Elimination to keep N_FEATURES_RFE variables ---
    base_estimator = RandomForestRegressor(
        n_estimators=250,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    rfe = RFE(
        estimator=base_estimator,
        n_features_to_select=N_FEATURES_RFE,
        step=0.1,
        importance_getter="auto",
    ).fit(X_train, y_train)

    selected_mask = rfe.support_
    selected_vars = X_train.columns[selected_mask].tolist()

    # If for some reason none were selected (shouldn't happen) ⇒ skip
    if not selected_vars:
        zeros = pd.Series(0.0, index=X_trainval.columns, name="importance")
        return selected_vars, zeros, np.nan, np.nan

    # --- Train RF on selected vars & compute permutation importance ---
    rf = RandomForestRegressor(
        n_estimators=250,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    ).fit(X_train[selected_vars], y_train)

    perm = permutation_importance(
        rf,
        X_train[selected_vars],
        y_train,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    importances = pd.Series(
        perm.importances_mean, index=selected_vars, name="importance"
    ).reindex(X_trainval.columns).fillna(0.0)

    # --- Performance on hold‑out fold ---
    y_pred = rf.predict(X_test[selected_vars])
    rmse_val = root_mean_squared_error(y_test, y_pred)
    mae_val = mean_absolute_error(y_test, y_pred)

    return selected_vars, importances, rmse_val, mae_val


def run_external_split(i: int, X_full: pd.DataFrame, y_full: pd.Series, k: int) -> dict:
    """One complete external repetition: train/val/test + internal CV."""

    print(f"[+] External repetition {i+1}")

    X_trainval_raw, X_test_raw, y_trainval, y_test = train_test_split(
        X_full, y_full, test_size=TEST_SIZE, random_state=RANDOM_STATE + i
    )

    scaler = StandardScaler().fit(X_trainval_raw)
    X_trainval = pd.DataFrame(scaler.transform(X_trainval_raw), columns=X_full.columns)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X_full.columns)

    rkf = RepeatedKFold(
        n_splits=k,
        n_repeats=int(np.ceil(N_ITER / k)),
        random_state=RANDOM_STATE,
    )

    # --- Run internal CV splits ---
    results = [
        run_single_split(X_trainval, y_trainval, train_idx, test_idx)
        for train_idx, test_idx in list(rkf.split(X_trainval))[:N_ITER]
    ]

    selected_lists, importances_list, rmses, maes = map(list, zip(*results))

    # --- Aggregate selection counts within this external repetition ---
    sel_counts_split = pd.Series(0, index=X_full.columns, dtype=int)
    for sel in selected_lists:
        sel_counts_split[sel] += 1

    # --- Aggregate permutation importances ---
    mean_importances_split = pd.concat(importances_list, axis=1).mean(axis=1)

    # --- Refit RF on *any* variable chosen at least once internally ---
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
        # Performance metrics
        "cv_rmse_mean": float(np.mean(rmses)),
        "cv_rmse_std": float(np.std(rmses)),
        "cv_mae_mean": float(np.mean(maes)),
        "cv_mae_std": float(np.std(maes)),
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        # Aggregates for later collection
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
    N_SPLITS_EXTERN = 60
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
    outdir = pathlib.Path("outputs_overfit_rfe")
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
