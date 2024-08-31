# src/features.py
import pandas as pd
from sklearn.model_selection import train_test_split

def define_features_labels(df):
    X = df.drop(columns='Churn', axis=1)
    y = df['Churn']
    return X, y

def perform_train_test_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
