from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt

def create_preprocessor(df):
        # Split the dataframe
    X = df.drop(columns='Churn', axis = 1)
    y = df['Churn']
    
    # Select numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    # Define the preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(), categorical_cols)
    ])
    return preprocessor

def fit_kmeans(n_clusters, df):
    X = df.drop(columns='Churn', axis = 1)
    y = df['Churn']
    kmeans_pipeline = Pipeline([
        ("preprocessor", create_preprocessor(df)),
        ("cluster", KMeans(n_clusters=n_clusters, random_state=9, verbose=0))
    ])
    kmeans_pipeline.fit(X)
    return kmeans_pipeline.named_steps["cluster"].inertia_

from sklearn.metrics import silhouette_samples, silhouette_score

def silhuette_score(n_clusters, df):
    X = df.drop(columns='Churn', axis = 1)
    y = df['Churn']
    silhouette_s = []

    for n_clusters in range(2, 11):
        kmeans_pipeline = Pipeline([
            ("preprocessor", create_preprocessor(df)),
            ("cluster", KMeans(n_clusters=n_clusters, random_state=9, verbose=0))
        ])

        # Fit the pipeline and get the cluster labels
        cluster_labels = kmeans_pipeline.fit_predict(X)
        
        # Get the preprocessed data
        X_tr = kmeans_pipeline.named_steps["preprocessor"].transform(X)
        
        silhouette_avg = silhouette_score(X_tr, cluster_labels).round(4)
        print(f"For n_clusters = {n_clusters}, The average silhouette_score is : {silhouette_avg}")
        
        silhouette_s.append(silhouette_avg)
        #print(silhouette_s)
    return silhouette_s

# src/model.py
import pandas as pd
import pickle
from pycaret.classification import setup, compare_models, tune_model
from sklearn.model_selection import train_test_split
from features import define_features_labels

def train_and_evaluate_model(df):
    X, y = define_features_labels(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clsfr = setup(
        data=pd.concat([X_train, y_train], axis=1),
        target='Churn',
        session_id=9,
        fix_imbalance=True,
        fix_imbalance_method='SMOTE',
        transformation=True,
        transformation_method='yeo-johnson',
        normalize=True,
        normalize_method='zscore',
        n_jobs=-1
    )
    
    best_model = compare_models(fold=5, n_select=1, sort='f1')
    
    tuned_model = tune_model(best_model)

    # Save the tuned model using pickle
    with open('model.pkl', 'wb') as file:
        pickle.dump(tuned_model, file)
    
    return tuned_model, X_test, y_test
