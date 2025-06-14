from sklearn.model_selection import LeaveOneOut
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

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
from sklearn.model_selection import train_test_split


from func_sel import *
from funcions_net import *
from crear_dataset import *
from funcions_net import *
from preprocessing import *
import matplotlib.pyplot as plt

# Suposem que aquest dataset ja ha passat per la pipeline i la selecció de features
# Substitueix 'filtered_datasets' per la variable que tinguis amb les dades ja tractades
# I substitueix 'TRFM' pel nom concret del dataset que vols provar

df_orig, meta = pyreadstat.read_sav("CEJFEAjut2015Updated.sav")
df_variables = pd.read_csv("variables3.csv", sep=";")
dict_vars = create_dict(meta)
df_psico = dataset_psicologia(df_orig, dict_vars, df_variables)
df = drop_all_columns(df_psico, meta, df_variables, dict_vars)
df_inicial = dataset_inicial(df_orig, dict_vars)
df_inicial.columns = [str(col) for col in df_inicial.columns]
df = df.reset_index(drop=True)
df_inicial = df_inicial.reset_index(drop=True)
df_inicial = df_inicial.drop(columns=df.columns.intersection(df_inicial.columns))


df["temps_fins_reincidencia1a"] = df_orig["temps_fins_reincidencia1a"].clip(upper=1095)
_, df = neteja_na_columns(df, llindar=0.6)
df = omple_nans(df)
df = label_encoding(df)

X_full = df.drop(columns="temps_fins_reincidencia1a")
y_full = df["temps_fins_reincidencia1a"]

X_trainval_raw, X_test_raw, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=8
    )

scaler = StandardScaler().fit(X_trainval_raw)
X_train = pd.DataFrame(scaler.transform(X_trainval_raw), columns=X_full.columns)
X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X_full.columns)


loo = LeaveOneOut()
prediccions = []
y_reals = []
X_tests = []

for train_index, test_index in tqdm(loo.split(X_full), total=len(X_full)):
    X_train_raw, X_test_raw = X_full.iloc[train_index], X_full.iloc[test_index]
    y_train, y_test_ = y_full.iloc[train_index], y_full.iloc[test_index]

    scaler = StandardScaler().fit(X_train_raw)
    X_train = pd.DataFrame(scaler.transform(X_train_raw), columns=X_full.columns)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X_full.columns)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)[0]

    prediccions.append(y_pred)
    y_reals.append(y_test_.values[0])
    X_tests.append(X_test.iloc[0].values)

# Convertir a arrays per ordenar i fer servir LIME
prediccions = np.array(prediccions)
y_reals = np.array(y_reals)
X_tests = np.array(X_tests)

# Seleccionem les 5 prediccions més baixes
top_5_idx = np.argsort(prediccions)[:25]  # de més baixa a més alta

# Preparem l'explainer de LIME
scaler = StandardScaler().fit(X_full)
X_scaled = pd.DataFrame(scaler.transform(X_full), columns=X_full.columns)
explainer = LimeTabularExplainer(
    training_data=np.array(X_scaled),
    feature_names=X_scaled.columns.tolist(),
    mode="regression"
)

# Mostrar les explicacions amb LIME per als top-5
for i, idx in enumerate(top_5_idx):
    sample = X_tests[idx].reshape(1, -1)
    
    exp = explainer.explain_instance(
        data_row=sample.flatten(),
        predict_fn=lambda x: model.predict(x),
        num_features=10
    )

    print(f"Mostra {i+1}:")
    print("Predicció del model:", prediccions[idx])
    print("Valor real:", y_reals[idx])
    
    fig = exp.as_pyplot_figure()
    plt.tight_layout()
    plt.show()
