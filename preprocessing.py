import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def neteja_na_columns(df, llindar=0.6):

    # Calculem el percentatge de NaNs per columna
    percentatge_nans = df.isnull().mean()
    
    # Seleccionem les columnes amb més del llindar de NaNs
    columnes_a_eliminar = percentatge_nans[percentatge_nans > llindar].index.tolist()
    
    # Eliminem les columnes seleccionades
    df_net = df.drop(columns=columnes_a_eliminar)
    
    return columnes_a_eliminar, df_net

## veure que hem tre

def columnes_netejades(netejat, dict):
    
    print(netejat)
    print(len(netejat))

    claus_corresponents = [clau for clau, valor in dict.items() if valor in netejat]

    print(claus_corresponents)


def omple_nans(df):
    df_net = df.copy()
    
    for col in df_net.columns:
        if df_net[col].dtype in ['int64', 'float64']:
            no_nuls = df_net[col].notnull().sum()
            if no_nuls < 2:
                df_net[col] = df_net[col].fillna(0)
            else:
                mitjana = df_net[col].mean()
                df_net[col] = df_net[col].fillna(mitjana)
        else:
            if not df_net[col].mode().empty:
                moda = df_net[col].mode().iloc[0]
                df_net[col] = df_net[col].fillna(moda)
    
    return df_net


def label_encoding(df):
    """
    Aplica Label Encoding a totes les columnes categòriques d'un DataFrame.
    
    Paràmetres:
        - df: DataFrame original.
    
    Retorna:
        - df_encoded: DataFrame amb les columnes categòriques codificades.
    """
    # Còpia del DataFrame per no modificar l'original
    df_encoded = df.copy()
    
    # Crear una instància del LabelEncoder
    le = LabelEncoder()
    
    # Recorre totes les columnes del DataFrame
    for col in df_encoded.columns:
        # Si la columna és categòrica o objecte
        if df_encoded[col].dtype == 'object' or df_encoded[col].dtype.name == 'category':
            # Aplica Label Encoding
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    return df_encoded

