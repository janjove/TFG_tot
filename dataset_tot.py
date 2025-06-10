
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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from func_sel import *
from funcions_net import *
from crear_dataset import *
from funcions_net import *
from preprocessing import *

# ╭────────────────────── User‑defined helpers ───────────────────────╮
# These should already exist in your project. If not, implement or
# import them accordingly. They are *only* stubbed here for completeness.
# ╰────────────────────────────────────────────────────────────────────╯

# ╭──────────────────────────── Parameters ────────────────────────────╮
N_ITER = 200           # Number of resampling iterations
ALPHA_THRESH = 0.01    # |coef| threshold for Lasso variable retention
TEST_SIZE = 0.2        # Approximate proportion for each test split
CAP_TARGET = 1_095     # Days (3 years) – clip upper outliers
RANDOM_STATE = 42      # Reproducibility
N_JOBS = -1            # Parallel cores (‑1 ⇒ all)
VERBOSE = True
# ╰────────────────────────────────────────────────────────────────────╯


# ╭───────────────────────────── Pipeline ─────────────────────────────╮

def run_single_split(
    X_scaled: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[pd.Series, pd.Series, float, float]:
    """Execute Lasso selection + RF importance on one data split."""

    X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Lasso with internal 5‑fold CV to pick alpha
    alphas = np.logspace(-4, 1, 50)
    lasso = LassoCV(
        alphas=alphas,
        cv=5,
        max_iter=10_000,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    ).fit(X_train, y_train)

    coefs = pd.Series(lasso.coef_, index=X_scaled.columns, name="coef")

    selected = coefs[coefs.abs() > ALPHA_THRESH].index.tolist()
    if not selected:
        # Return zeros; Skip RF to save time.
        zeros = pd.Series(0.0, index=X_scaled.columns, name="importance")
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
    ).reindex(X_scaled.columns).fillna(0.0)

    # Performance on hold‑out set
    y_pred = rf.predict(X_test[selected])
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return coefs, importances, mse, r2


def main(path_sav: str = "CEJFEAjut2015Updated.sav", path_vars: str = "variables3.csv") -> None:
    t0 = datetime.now()
    print("[+] Loading raw data …")

    df_orig, meta = pyreadstat.read_sav(path_sav)
    df_variables = pd.read_csv(path_vars, sep=";")

    dict_vars = create_dict(meta)
    df_psico = dataset_psicologia(df_orig, dict_vars, df_variables)
    df = drop_all_columns(df_psico, meta, df_variables, dict_vars)

    # Cap target
    df["temps_fins_reincidencia1a"] = (
        df_orig["temps_fins_reincidencia1a"].clip(upper=CAP_TARGET)
    )

    # Clean & encode
    _, df = neteja_na_columns(df, llindar=0.6)
    df = omple_nans(df)
    df = label_encoding(df)

    # Split features / target
    X = df.drop(columns="temps_fins_reincidencia1a")
    y = df["temps_fins_reincidencia1a"]

    # Standardize once for all splits (safer for HDLSS)
    scaler = StandardScaler().fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, copy=False)

    # Resampling strategy: pick k so that test≈TEST_SIZE
    k = int(round(1 / TEST_SIZE))
    rkf = RepeatedKFold(
        n_splits=k,
        n_repeats=int(np.ceil(N_ITER / k)),
        random_state=RANDOM_STATE,
    )

    # Run splits in parallel
    results = Parallel(n_jobs=N_JOBS, verbose=VERBOSE)(
        delayed(run_single_split)(X_scaled, y, train_idx, test_idx)
        for (_, (train_idx, test_idx)) in tqdm(
            enumerate(rkf.split(X_scaled), 1), total=N_ITER, disable=not VERBOSE
        )
    )[:N_ITER]  # Truncate in case of overshoot

    # Unpack
    lasso_coefs, importances, mses, r2s = map(list, zip(*results))

    mean_coefs = pd.concat(lasso_coefs, axis=1).mean(axis=1)
    mean_imps = pd.concat(importances, axis=1).mean(axis=1)

    # Persist
    outdir = pathlib.Path("outputs2").mkdir(parents=True, exist_ok=True) or pathlib.Path("outputs")
    mean_coefs.to_csv(outdir / "lasso_mean_coefficients.csv", header=["mean_coef"])
    mean_imps.to_csv(outdir / "rf_mean_importances.csv", header=["mean_importance"])
    pd.DataFrame({"mse": mses, "r2": r2s}).to_csv(
        outdir / "model_performance_metrics.csv", index=False
    )

    print("\n[✓] Pipeline completed in", datetime.now() - t0)
    print("   • outputs/lasso_mean_coefficients.csv")
    print("   • outputs/rf_mean_importances.csv")
    print("   • outputs/model_performance_metrics.csv")


if __name__ == "__main__":
    try:
        main(*sys.argv[1:])
    except KeyboardInterrupt:
        print("Interrupted by user. Bye!")
