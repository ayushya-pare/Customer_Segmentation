import pandas as pd

def preprocess_data(df):
    df.rename(columns={'Partner': 'Married', 'Dependents': 'Children'}, inplace=True)
    df.replace({" ": '0'}, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    #df['SeniorCitizen'] = df['SeniorCitizen'].replace({1: 'yes', 0: 'no'})
    #df['gender'] = df['gender'].replace({'Male': 1, 'Female': 0})
    #df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})
    df.drop(columns=['customerID', 'PaperlessBilling'], inplace=True)
    
    internet_service_features = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for feature in internet_service_features:
        df[feature] = df[feature].replace('No internet service', 'No')
    df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')
    
    return df

def save_data(df, path):
    df.to_csv(path, index=False)
