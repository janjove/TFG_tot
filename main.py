
import pandas as pd
import pyreadstat
import funcions_net
import numpy as np
from preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA







df, meta = pyreadstat.read_sav("CEJFEAjut2015Updated.sav")

df_variables = pd.read_csv("variables.csv", sep=';')

dict = funcions_net.create_dict(meta)

print("Creem diccionari")


df = funcions_net.drop_columns(df, funcions_net.find_drop_columns(meta, df_variables,dict))

print("Columnes eliminades")

df = funcions_net.drop_columns(df, funcions_net.eliminem_preguntes(meta, df_variables,dict))

print("Columnes eliminades")

print(df['temps_fins_reincidencia1a'])

## mirem quants nans tenim a cada columna
## ordenem per els que més nans tenen

print("Mirem quants nans tenim a cada columna")
netejat,df = neteja_na_columns(df, llindar=0.7)

df = omple_nans(df)

print("Columnes netejades")
## fem label encoding de les columnes categòriques
df = label_encoding(df)




## separem entre train i test
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='temps_fins_reincidencia1a'), df['temps_fins_reincidencia1a'], test_size=0.2, random_state=42)

## normalitzem les dades
scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)

## fem un pca

pca = PCA(n_components=100)

pca.fit(X_train)
X_train = pca.transform(X_train)

## Provem de fer una regressió de Lasso

## Fem reg
param_grid = {
    'alpha': [10, 100, 1000, 10000, 100000]
}

# Crea el model Lasso
lasso = Lasso(max_iter=100000)
# Aplica GridSearchCV per trobar el millor alpha
grid_search = GridSearchCV(lasso, param_grid, cv=10)
grid_search.fit(X_train, y_train)

millor_alpha = grid_search.best_params_['alpha']
# Mostra el millor valor d'alpha
print("Millor valor d'alpha:", millor_alpha)
print("Coeficients:", grid_search.cv_results_['mean_test_score'])

