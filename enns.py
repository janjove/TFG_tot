import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from tensorflow.keras import Input

def create_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(50, activation='relu'),
        Dropout(0.3),
        Dense(30, activation='relu'),
        Dropout(0.3),
        Dense(15, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mse'])
    return model

# Funció per calcular importància de variables (aproximació basada en pesos)
def get_feature_importance(model):
    # Agafem només els pesos de la primera capa
    weights = model.layers[0].get_weights()[0]
    importance = np.sum(np.abs(weights), axis=1)
    return importance

# Implementació simplificada de ENNS
def enns(X, y, n_bootstraps=50, importance_threshold=0.6, min_selection_ratio=0.5):
    feature_counts = np.zeros(X.shape[1])

    #for i in tqdm(range(n_bootstraps), desc="Bootstrap iterations"):
    for i in tqdm(range(n_bootstraps)):
        X_bs, y_bs = resample(X, y, replace=True)
        model = create_model(X_bs.shape[1])
        model.fit(X_bs, y_bs, epochs=50, batch_size=16, verbose=0)

        importance = get_feature_importance(model)
        importance = importance / np.max(importance)
        feature_counts += (importance >= importance_threshold)

    selected_features = np.where(feature_counts >= (n_bootstraps * min_selection_ratio))[0]
    return selected_features

