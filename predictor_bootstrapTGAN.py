#!/usr/bin/env python
"""
HDLSS feature‑importance pipeline for psychopathy data
=====================================================

This script implements a **repeated‑split, stability‑aware** workflow that
combines Lasso for sparse selection and Random‑Forest permutation importance
for non‑linear effects.

Outputs (CSV):
    • lasso_mean_coefficients.csv   ― Mean Lasso coefficient per variable
    • rf_mean_importances.csv       ― Mean permutation importance per variable
    • model_performance_metrics.csv ― MSE & R² for each split

The design follows these principles
-----------------------------------
1. Keep **all** observations available via repeated resampling rather than a
   single train/test split.
2. Quantify selection stability by averaging over many resamples.
3. Use permutation importance (not MDI) to avoid split‑based bias.
4. Cap the target at 1 095 days as indicated in the source notebook.

Dependencies
------------
    pip install pandas numpy pyreadstat scikit‑learn joblib tqdm

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
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error,root_mean_squared_error
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.model_selection import train_test_split


from func_sel import *
from funcions_net import *
from crear_dataset import *
from funcions_net import *
from preprocessing import *
import matplotlib.pyplot as plt
from sklearn.utils import resample

from ctgan import CTGAN
import torch                          # opcional, per saber si tens GPU


# ╭────────────────────── User‑defined helpers ───────────────────────╮
# These should already exist in your project. If not, implement or
# import them accordingly. They are *only* stubbed here for completeness.
# ╰────────────────────────────────────────────────────────────────────╯

# ╭──────────────────────────── Parameters ────────────────────────────╮
N_ITER = 5           # Number of resampling iterations
ALPHA_THRESH = 0.01    # |coef| threshold for Lasso variable retention
TEST_SIZE = 0.2        # Approximate proportion for each test split
CAP_TARGET = 1_095     # Days (3 years) – clip upper outliers
RANDOM_STATE = 10      # Reproducibility
N_JOBS = -1            # Parallel cores (‑1 ⇒ all)
VERBOSE = True
# ╰────────────────────────────────────────────────────────────────────╯


# ╭───────────────────────────── Pipeline ─────────────────────────────╮
def afegir_soroll_discret(df, prob_mutacio=0.1):
    df_noisy = df.copy()
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            # Soroll numèric discret: sumem o restem 1 amb probabilitat
            muta = np.random.rand(len(df)) < prob_mutacio
            increments = np.random.choice([-1, 1], size=len(df))
            df_noisy[col] += muta * increments
        else:
            # Soroll categòric: canvi a un altre valor aleatori de la columna
            muta = np.random.rand(len(df)) < prob_mutacio
            valors_possibles = df[col].unique()
            nous_valors = np.random.choice(valors_possibles, size=len(df))
            df_noisy.loc[muta, col] = nous_valors[muta]
    return df_noisy

def run_single_split(
    X_trainval: pd.DataFrame,
    y_trainval: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[pd.Series, pd.Series, float, float]:
    """Execute Lasso selection + RF importance on one data split."""

    X_train, X_test = X_trainval.iloc[train_idx], X_trainval.iloc[test_idx]
    y_train, y_test = y_trainval.iloc[train_idx], y_trainval.iloc[test_idx]

    # Lasso with internal 5‑fold CV to pick alpha
    alphas = np.logspace(0, 1, 50)
    lasso = LassoCV(
        alphas=alphas,
        cv=5,
        max_iter=10_000,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    ).fit(X_train, y_train)

    coefs = pd.Series(lasso.coef_, index=X_trainval.columns, name="coef")

    selected = coefs[coefs.abs() > ALPHA_THRESH].index.tolist()
    if not selected:
        # Return zeros; Skip RF to save time.
        zeros = pd.Series(0.0, index=X_trainval.columns, name="importance")
        return coefs, zeros, np.nan, np.nan

    # Random‑Forest model on selected features only
    rf = RandomForestRegressor(
        n_estimators=250,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    ).fit(X_train[selected], y_train)

    # Permutation importance (on train subset)
    perm = permutation_importance(
        rf,
        X_train[selected],
        y_train,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    importances = pd.Series(
        perm.importances_mean, index=selected, name="importance"
    ).reindex(X_trainval.columns).fillna(0.0)

    # Performance on hold‑out set
    y_pred = rf.predict(X_test[selected])
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return coefs, importances, rmse, mae

def tgan_sample(df_train: pd.DataFrame,
                n_samples: int,
                epochs: int = 300,
                seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Entrena un CTGAN sobre df_train i retorna n_samples files sintètiques.
    df_train ha d’incloure la variable objectiu.
    """
    model = CTGAN(
        epochs=epochs,
        cuda=torch.cuda.is_available(),   # aprofita GPU si n’hi ha
    )
    model.fit(df_train)
    return model.sample(n_samples)
def run_external_split(i: int, X_full: pd.DataFrame, y_full: pd.Series, k: int) -> dict:
    """Una repetició externa completa (train/test + CV + test final)"""
    print(f"[+] Repetició externa {i+1}")

    # 1. Split inicial
    X_trainval_raw, X_test_raw, y_trainval, y_test = train_test_split(
        X_full, y_full, test_size=TEST_SIZE, random_state=RANDOM_STATE + i
    )

    # 2. Fit scaler amb dades reals
    scaler = StandardScaler().fit(X_trainval_raw)
    X_trainval = pd.DataFrame(scaler.transform(X_trainval_raw), columns=X_full.columns)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X_full.columns)

    # 3. Augmentació amb bootstrap + soroll discret
    n_bootstrap = int(len(X_trainval) * 0.5)
    Xy_trainval = pd.concat([X_trainval, y_trainval.reset_index(drop=True)], axis=1)
    Xy_bootstrap = resample(Xy_trainval, replace=True, n_samples=n_bootstrap, random_state=RANDOM_STATE + i)
    X_trainval_bootstrap = Xy_bootstrap[X_full.columns]
    y_trainval_bootstrap = Xy_bootstrap[y_trainval.name]
    X_trainval_bootstrap_noisy = afegir_soroll_discret(X_trainval_bootstrap, prob_mutacio=0.1)

    # 4. Augmentació amb dades sintètiques
    train_table = pd.concat([X_trainval_raw.reset_index(drop=True), y_trainval.reset_index(drop=True)], axis=1)
    n_synth = int(len(X_trainval_raw) * 0.5)
    synth_table = tgan_sample(train_table, n_synth)
    X_synth = pd.DataFrame(scaler.transform(synth_table[X_full.columns]), columns=X_full.columns)
    y_synth = synth_table[y_trainval.name]

    # 5. Concatenem tot: real + bootstrap sorollós + sintètic
    X_trainval_augmented = pd.concat([X_trainval, X_trainval_bootstrap_noisy, X_synth], ignore_index=True)
    y_trainval_augmented = pd.concat([y_trainval, y_trainval_bootstrap, y_synth], ignore_index=True)

    # 6. Validació creuada repetida sobre el conjunt augmentat
    rkf = RepeatedKFold(
        n_splits=k,
        n_repeats=int(np.ceil(N_ITER / k)),
        random_state=RANDOM_STATE
    )

    results = [
        run_single_split(X_trainval_augmented, y_trainval_augmented, train_idx, test_idx)
        for train_idx, test_idx in list(rkf.split(X_trainval_augmented))[:N_ITER]
    ]

    lasso_coefs, importances, rmses, maes = map(list, zip(*results))
    mean_coefs = pd.concat(lasso_coefs, axis=1).mean(axis=1)
    selected_vars = mean_coefs[mean_coefs.abs() > ALPHA_THRESH].index.tolist()

    if selected_vars:
        rf_final = RandomForestRegressor(
            n_estimators=250,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ).fit(X_trainval_augmented[selected_vars], y_trainval_augmented)

        y_test_pred = rf_final.predict(X_test[selected_vars])
        test_rmse = root_mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
    else:
        test_rmse = np.nan
        test_mae = np.nan

    return {
        "cv_rmse_mean": np.mean(rmses),
        "cv_rmse_std": np.std(rmses),
        "cv_mae_mean": np.mean(maes),
        "cv_mae_std": np.std(maes),
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "n_selected_vars": len(selected_vars),
    }



def main(path_sav: str = "CEJFEAjut2015Updated.sav", path_vars: str = "variables3.csv") -> None:
    t0 = datetime.now()
    print("[+] Loading raw data …")

    # --- Càrrega i preprocessat (igual que abans) ---
    t0 = datetime.now()
    print("[+] Loading raw data …")

    df_orig, meta = pyreadstat.read_sav(path_sav)
    df_variables = pd.read_csv(path_vars, sep=";")
    dict_vars = create_dict(meta)


    ## datset psicologia
    df_psico = dataset_psicologia(df_orig, dict_vars, df_variables)
    df_psico = drop_all_columns(df_psico, meta, df_variables, dict_vars)

    ## dataset inicial
    df_inicial = dataset_inicial(df_orig, dict_vars)

    ### dataset global
    df_global = create_dataset_global(df_orig, dict_vars,df_variables,meta)

    df = df_global.copy()

    print("Mida del dataset global:", df.shape)

    df["temps_fins_reincidencia1a"] = df_orig["temps_fins_reincidencia1a"].clip(upper=CAP_TARGET)


    _, df = neteja_na_columns(df, llindar=0.6)
    df = omple_nans(df)
    df = label_encoding(df)


    X_full = df.drop(columns="temps_fins_reincidencia1a")
    y_full = df["temps_fins_reincidencia1a"]

    # --- Paràmetres per repetits splits externs ---
    N_SPLITS_EXTERN = 60
    k = int(round(1 / TEST_SIZE))
    all_results = []

    all_results = Parallel(n_jobs=N_JOBS, verbose=VERBOSE)(
    delayed(run_external_split)(i, X_full, y_full, k)
    for i in range(N_SPLITS_EXTERN)
    )

    # --- Guarda resultats
    outdir = pathlib.Path("outputs_overfit_TGAN")
    outdir.mkdir(parents=True, exist_ok=True)

    df_results = pd.DataFrame(all_results)

    # 1. Calcular la mitjana i la desviació estàndard
    row_mitjanes = df_results.mean(numeric_only=True).round(2)
    row_mitjanes.name = "Mitjana"

    row_std = df_results.std(numeric_only=True).round(2)
    row_std.name = "Desviació estàndard"

    # 2. Crear nou DataFrame amb ambdues files
    df_stats = pd.DataFrame([row_mitjanes, row_std])

    # 3. Guardar el DataFrame original
    df_results.round(2).to_csv(outdir / "overfitting_evidence.csv", index=False)

    # 4. Guardar les estadístiques en un fitxer separat
    df_stats.to_csv(outdir / "resum_estadistiques.csv")



    # --- Gràfica 1: CV RMSE vs Test RMSE ---
    plt.figure(figsize=(10, 6))
    plt.plot(df_results["cv_rmse_mean"], label="CV RMSE", marker='o')
    plt.plot(df_results["test_rmse"], label="Test RMSE", marker='x')
    plt.xlabel("Repetició externa")
    plt.ylabel("RMSE")
    plt.title("Comparació entre CV RMSE i Test RMSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "rmse_comparison.png")
    plt.close()

    # --- Gràfica 2: Diferència (Test - CV) ---
    plt.figure(figsize=(10, 6))
    diff = df_results["test_rmse"] - df_results["cv_rmse_mean"]
    plt.bar(range(len(diff)), diff)
    plt.xlabel("Repetició externa")
    plt.ylabel("Test RMSE - CV RMSE")
    plt.title("Diferència d'error: test - validació")
    plt.axhline(0, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig(outdir / "rmse_difference.png")
    plt.close()

    # --- Gràfica 3: CV MAE vs Test MAE ---
    plt.figure(figsize=(10, 6))
    plt.plot(df_results["cv_mae_mean"], label="CV MAE", marker='o')
    plt.plot(df_results["test_mae"], label="Test MAE", marker='x')
    plt.xlabel("Repetició externa")
    plt.ylabel("MAE")
    plt.title("Comparació entre CV MAE i Test MAE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "mae_comparison.png")
    plt.close()

    # --- Gràfica 4: Diferència (Test - CV) ---
    plt.figure(figsize=(10, 6))
    diff_mae = df_results["test_mae"] - df_results["cv_mae_mean"]
    plt.bar(range(len(diff_mae)), diff_mae, color='orange')
    plt.xlabel("Repetició externa")
    plt.ylabel("Test MAE - CV MAE")
    plt.title("Diferència d'error: test - validació (MAE)")
    plt.axhline(0, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig(outdir / "mae_difference.png")
    plt.close()

    # --- Gràfica 5: Distribució de rmse del test ---
    plt.figure(figsize=(8, 5))
    plt.hist(df_results["test_rmse"], bins=10, color="orange", edgecolor="black", alpha=0.85)
    plt.xlabel("Test RMSE")
    plt.ylabel("Freqüència")
    plt.title("Distribució del Test RMSE")
    plt.tight_layout()
    plt.savefig(outdir / "test_rmse_distribution.png")
    plt.close()

    # --- Gràfica 6: Distribució de mae del test ---
    plt.figure(figsize=(8, 5))
    plt.hist(df_results["test_mae"], bins=10, color="skyblue", edgecolor="black", alpha=0.85)
    plt.xlabel("Test MAE")
    plt.ylabel("Freqüència")
    plt.title("Distribució del Test MAE")
    plt.tight_layout()
    plt.savefig(outdir / "test_mae_distribution.png")
    plt.close()





    print("   • outputs_overfit/rmse_comparison.png")
    print("   • outputs_overfit/rmse_difference.png")
    print("   • outputs_overfit/test_r2_distribution.png")

    print("\n[✓] Finalitzat. Resultats guardats a:")
    print("   • outputs_overfit/overfitting_evidence.csv")



if __name__ == "__main__":
    try:
        main(*sys.argv[1:])
    except KeyboardInterrupt:
        print("Interrupted by user. Bye!")
