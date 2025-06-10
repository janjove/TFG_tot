from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

class NetejaNAColumns(BaseEstimator, TransformerMixin):
    def __init__(self, llindar=0.6):
        self.llindar = llindar
    
    def fit(self, X, y=None):
        self.cols_to_drop = X.columns[X.isnull().mean() > self.llindar]
        return self
    
    def transform(self, X):
        return X.drop(columns=self.cols_to_drop, errors='ignore')

class OmpleNans(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in X_copy.columns:
            if X_copy[col].dtype in ['int64', 'float64']:
                no_nuls = X_copy[col].notnull().sum()
                if no_nuls < 2:
                    X_copy[col].fillna(0, inplace=True)
                else:
                    X_copy[col].fillna(X_copy[col].mean(), inplace=True)
            else:
                moda = X_copy[col].mode().iloc[0] if not X_copy[col].mode().empty else None
                X_copy[col].fillna(moda, inplace=True)
        return X_copy

class LabelEncoding(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.encoders = {col: LabelEncoder().fit(X[col].astype(str)) for col in X.select_dtypes(include=['object', 'category']).columns}
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col, encoder in self.encoders.items():
            X_copy[col] = encoder.transform(X_copy[col].astype(str))
        return X_copy

