import pandas as pd

def download_data(file_path):
    return pd.read_csv(file_path, index_col=0)

