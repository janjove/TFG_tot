#!/usr/bin/env python
"""
HDLSS feature-importance pipeline per a cada test psicològic
===========================================================

Per cada dataset que genera `dataset_dif2`, executem:
  • LassoCV + selecció (coeficients per iteració)
  • RF + permutation importance (importàncies per iteració)
  • Càlcul de RMSE i MAE per iteració

I guardem 3 CSVs per test:
    – lasso_coefficients_iterations.csv
    – rf_importances_iterations.csv
    – performance_metrics.csv (conté RMSE i MAE)
"""

from __future__ import annotations
import pathlib, sys
from datetime import datetime

import numpy as np
import pandas as pd
import pyreadstat
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from func_sel import *
from funcions_net import *
from crear_dataset import *
from preprocessing import *
from datasets_separats  import dataset_dif2       # importa aquí el teu funció

# ───────────────────────── Paràmetres ──────────────────────────
N_ITER      = 120
ALPHA_THRESH= 0.01
TEST_SIZE   = 0.2
CAP_TARGET  = 1_095
RANDOM_STATE= 42
N_JOBS      = -1
VERBOSE     = True
# ──────────────────────────────────────────────────────────────

def run_single_split(X_scaled, y, train_idx, test_idx):
    """Realitza una única divisió train/test i retorna coeficients Lasso, importàncies RF,
    RMSE i MAE."""
    X_tr, X_te = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    # LassoCV
    alphas = np.logspace(0, 1, 50)
    lasso = LassoCV(alphas=alphas, cv=5, max_iter=10_000,
                    random_state=RANDOM_STATE, n_jobs=N_JOBS).fit(X_tr, y_tr)
    coef = pd.Series(lasso.coef_, index=X_scaled.columns)

    # Selecció
    selected = coef[coef.abs() > ALPHA_THRESH].index.tolist()
    if not selected:
        zeros = pd.Series(0.0, index=X_scaled.columns)
        rmse = np.nan; mae = np.nan
        return coef, zeros, rmse, mae

    # RF + permutation importance
    rf = RandomForestRegressor(n_estimators=250,
                               random_state=RANDOM_STATE,
                               n_jobs=N_JOBS).fit(X_tr[selected], y_tr)
    perm = permutation_importance(rf, X_tr[selected], y_tr,
                                  n_repeats=10,
                                  random_state=RANDOM_STATE,
                                  n_jobs=N_JOBS)
    imp = pd.Series(perm.importances_mean, index=selected)\
             .reindex(X_scaled.columns).fillna(0.0)

    # Mètriques en test
    y_pred = rf.predict(X_te[selected])
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae  = mean_absolute_error(y_te, y_pred)

    return coef, imp, rmse, mae


def process_dataset(name, df_full, dict_vars, df_variables):
    """Aplica tot el pipeline a un sol DataFrame i desa els resultats, retornant
    un resum de mitjanes per al test."""
    # 1) Neteges i encode (igual que al teu main original)
    df = drop_all_columns(df_full, None, df_variables, dict_vars)  # o l’equivalent
    # ... (neteja_na_columns, omple_nans, label_encoding…)
    df["temps_fins_reincidencia1a"] = df_full["temps_fins_reincidencia1a"].clip(upper=CAP_TARGET)

    _, df = neteja_na_columns(df, llindar=0.6)
    df = omple_nans(df)
    df = label_encoding(df)

    # 2) X / y i escalat
    X = df.drop(columns="temps_fins_reincidencia1a")
    y = df["temps_fins_reincidencia1a"]
    scaler = StandardScaler().fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, copy=False)

    # 3) RepeatedKFold
    k = int(round(1 / TEST_SIZE))
    rkf = RepeatedKFold(n_splits=k,
                        n_repeats=int(np.ceil(N_ITER / k)),
                        random_state=RANDOM_STATE)

    # 4) Paral·lelització
    results = Parallel(n_jobs=N_JOBS, verbose=VERBOSE)(
        delayed(run_single_split)(X_scaled, y, tr, te)
        for (_, (tr, te)) in tqdm(
            enumerate(rkf.split(X_scaled), 1),
            total=N_ITER,
            disable=not VERBOSE
        )
    )[:N_ITER]

    # 5) Desem resultats per iteració
    coefs, imps, rmses, maes = map(list, zip(*results))

    # Crear carpeta per aquest test
    outdir = pathlib.Path("outputs2") / name
    outdir.mkdir(parents=True, exist_ok=True)

    # 5a) Coeficients per iteració
    pd.concat(coefs, axis=1).to_csv(
        outdir / "lasso_coefficients_iterations.csv",
        index=True,
        header=[f"iter_{i+1}" for i in range(len(coefs))]
    )

    # 5b) Importàncies per iteració
    pd.concat(imps, axis=1).to_csv(
        outdir / "rf_importances_iterations.csv",
        index=True,
        header=[f"iter_{i+1}" for i in range(len(imps))]
    )

    # 5c) Mètriques per iteració
    pd.DataFrame({"rmse": rmses, "mae": maes})\
      .to_csv(outdir / "performance_metrics.csv", index_label="iter")

    print(f"[✓] Resultats per '{name}' desats a {outdir}")

    # 6) Càlcul de mitjanes
    mean_coefs = pd.concat(coefs, axis=1).mean(axis=1)
    mean_imps  = pd.concat(imps, axis=1).mean(axis=1)
    mean_rmse  = np.nanmean(rmses)
    mean_mae   = np.nanmean(maes)
    mean_features = df[X.columns].mean()  # abans d’escalar

    summary = pd.concat([
        mean_coefs.rename("mean_lasso_coef"),
        mean_imps.rename("mean_rf_importance"),
        mean_features.rename("mean_original_feature")
    ], axis=1)

    # Afegim les mètriques globals a part (mateixes per totes les files)
    summary.reset_index(inplace=True)
    summary["mean_rmse"] = mean_rmse
    summary["mean_mae"] = mean_mae
    summary["test_name"] = name

    print(f"[✓] Mitjanes calculades per '{name}'")
    return summary


def main(path_sav="CEJFEAjut2015Updated.sav", path_vars="variables3.csv"):
    t0 = datetime.now()
    print("[+] Carregant dades…")
    df_orig, meta      = pyreadstat.read_sav(path_sav)
    df_variables       = pd.read_csv(path_vars, sep=";")
    dict_vars          = create_dict(meta)

    # Dataset per test
    datasets = dataset_dif2(df_orig, dict_vars, df_variables)

    summary_list = []

    for name, df_test in datasets.items():
        summary = process_dataset(name, df_test, dict_vars, df_variables)
        summary_list.append(summary)

    # Combina tots els resums
    all_summary = pd.concat(summary_list, axis=0)

    # Arrodonim a 3 decimals
    all_summary_rounded = all_summary.copy()
    numeric_cols = all_summary_rounded.select_dtypes(include=[np.number]).columns
    all_summary_rounded[numeric_cols] = all_summary_rounded[numeric_cols].round(3)

    # Crea la carpeta de sortida
    output_dir = pathlib.Path("outputs2")
    output_dir.mkdir(exist_ok=True)

    # 1. Desa només les mitjanes de Lasso
    lasso_df = all_summary_rounded[["index", "mean_lasso_coef", "test_name"]]
    lasso_df.to_csv(output_dir / "all_tests_lasso.csv", index=False)

    # 2. Desa només les importàncies de Random Forest
    imp_df = all_summary_rounded[["index", "mean_rf_importance", "test_name"]]
    imp_df.to_csv(output_dir / "all_tests_importances.csv", index=False)

    # 3. Desa mètriques de rendiment a part (sense repetir per cada feature)
    perf_df = all_summary_rounded[["test_name", "mean_rmse", "mean_mae"]].drop_duplicates()
    perf_df.to_csv(output_dir / "all_tests_performance.csv", index=False)

    print(f"\n[✓] Lasso desat a {output_dir / 'all_tests_lasso.csv'}")
    print(f"[✓] Importàncies desades a {output_dir / 'all_tests_importances.csv'}")
    print(f"[✓] Mètriques de rendiment desades a {output_dir / 'all_tests_performance.csv'}")

    elapsed = datetime.now() - t0
    print(f"[✓] Procés completat en {elapsed.total_seconds():.1f}s")


if __name__ == "__main__":
    try:
        main(*sys.argv[1:])
    except KeyboardInterrupt:
        print("Interrupted by user. Bye!")
