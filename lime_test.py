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
df = pd.concat([df, df_inicial], axis=1)

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


# 1. Entrenar el model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 2. Seleccionem un exemple concret (p. ex., el primer del test set)
i = 0
sample = X_test.iloc[i].values.reshape(1, -1)

# 3. Preparem LIME
explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns.tolist(),
    mode="regression"
)
# Obtenir totes les prediccions sobre el test set
prediccions = model.predict(X_test)

# Agafem els índexs dels 5 casos amb predicció més alta
top_5_idx = np.argsort(prediccions)  # de més baixa a més alta

# Bucle per mostrar les explicacions de LIME
for i, idx in enumerate(top_5_idx):
    sample = X_test.iloc[idx].values.reshape(1, -1)
    exp = explainer.explain_instance(
        data_row=sample.flatten(),
        predict_fn=model.predict,
        num_features=10
    )
    
    print("Predicció del model:", model.predict(sample)[0])
    print("Valor real de y_test:", y_test.iloc[idx])
    
    # Mostrar gràfic de LIME
    fig = exp.as_pyplot_figure()
    plt.tight_layout()
    plt.show()
