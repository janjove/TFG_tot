from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import resample
import numpy as np
from tqdm import tqdm


def create_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(30, activation='relu'),       # Menys neurones per evitar sobreajust
        Dropout(0.4),                       # Dropout més alt per més regularització
        Dense(15, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mse'])
    return model


# Càlcul d'importància basada només en primera capa (pot millorar-se però deixem així per ara)
def get_feature_importance(model):
    weights = model.layers[0].get_weights()[0]
    importance = np.sum(np.abs(weights), axis=1)
    return importance


# Implementació millorada de ENNS amb EarlyStopping i validació interna
def enns(X, y, n_bootstraps=75, importance_threshold=0.5, min_selection_ratio=0.5):
    print("Iniciant ENNS amb EarlyStopping i validació interna...")
    print(f"Paràmetres: n_bootstraps={n_bootstraps}, importance_threshold={importance_threshold}, min_selection_ratio={min_selection_ratio}")
    feature_counts = np.zeros(X.shape[1])

    for _ in tqdm(range(n_bootstraps)):
        X_bs, y_bs = resample(X, y, replace=True)
        model = create_model(X_bs.shape[1])

        # Early stopping amb validació interna per evitar sobreajust
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_bs, y_bs, epochs=100, batch_size=16,
                  validation_split=0.2, callbacks=[early_stop], verbose=0)

        importance = get_feature_importance(model)
        importance = importance / np.max(importance)
        feature_counts += (importance >= importance_threshold)

    selected_features = np.where(feature_counts >= (n_bootstraps * min_selection_ratio))[0]
    return selected_features



def create_classification_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(30, activation='relu'),
        Dropout(0.2),
        Dense(15, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # activació per classificació binària
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_feature_importance(model):
    weights = model.layers[0].get_weights()[0]
    importance = np.sum(np.abs(weights), axis=1)
    return importance

def enns_classification(X, y, n_bootstraps=40, importance_threshold=0.7, min_selection_ratio=0.7):
    print("Iniciant ENNS per classificació amb EarlyStopping...")
    print(f"Paràmetres: n_bootstraps={n_bootstraps}, importance_threshold={importance_threshold}, min_selection_ratio={min_selection_ratio}")
    feature_counts = np.zeros(X.shape[1])

    for _ in tqdm(range(n_bootstraps)):
        X_bs, y_bs = resample(X, y, replace=True)

        X_bs = X_bs.astype(np.float32)
        y_bs = y_bs.astype(np.float32)

        model = create_classification_model(X_bs.shape[1])

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_bs, y_bs, epochs=100, batch_size=32,
                  validation_split=0.2, callbacks=[early_stop], verbose=0)

        importance = get_feature_importance(model)
        importance = importance / np.max(importance)
        feature_counts += (importance >= importance_threshold)

    selected_features = np.where(feature_counts >= (n_bootstraps * min_selection_ratio))[0]
    return selected_features
